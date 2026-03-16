"""
LLM Agent: GPT-based agent using function calling API.
Implements ReAct-style prompting with memory.
"""

import json
import time
import os
from typing import Dict, List, Optional, Any
from openai import OpenAI
from game.environment import Player, GameState, Phase, Role, ADJACENCY, TASK_LOCATIONS
from game.mechanics import get_available_actions
from agents.memory import AgentMemory
from agents.personalities import get_personality


# OpenAI function definitions for Among Us actions
TASK_PHASE_TOOLS_CREW = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move to an adjacent room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Name of the adjacent room to move to."}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Complete a task in the current room (only if the task is available here).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Name of the task to complete."}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_body",
            "description": "Report a dead body in the current room to trigger an emergency meeting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "Name of the dead player to report."}
                },
                "required": ["body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Stay in the current room and do nothing this turn.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

TASK_PHASE_TOOLS_IMP = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move to an adjacent room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Name of the adjacent room."}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kill",
            "description": "Kill a crewmate in the same room. Only do this when you won't be caught.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Name of the crewmate to kill."}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Stay in the current room. Useful for waiting for isolated targets.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

MEETING_PHASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Make a statement during the meeting discussion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speech": {"type": "string", "description": "Your statement to the group (1-3 sentences)."}
                },
                "required": ["speech"]
            }
        }
    }
]

VOTE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vote",
            "description": "Cast your vote to eject a player, or skip voting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name of the player to vote out, or 'SKIP' to abstain."
                    }
                },
                "required": ["target"]
            }
        }
    }
]


class LLMAgent:
    def __init__(
        self,
        player: Player,
        memory_size: int = 10,
        use_react: bool = True,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        self.player = player
        self.memory = AgentMemory(max_size=memory_size)
        self.use_react = use_react
        self.model = model
        self.personality_name, self.personality_desc = get_personality(player.role.value)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Inject LLM caller into memory for compression
        self.memory._llm_caller = self._simple_llm_call

    def _simple_llm_call(self, prompt: str) -> str:
        """Simple LLM call for memory compression."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[Summary unavailable: {e}]"

    def _build_system_prompt(self) -> str:
        role = self.player.role.value
        if role == "Impostor":
            goal = ("Kill all Crewmates without being voted out. Blend in, complete fake tasks, "
                    "and deceive others in meetings.")
        else:
            goal = ("Complete all your tasks and identify the Impostor. Vote out suspicious players "
                    "in meetings.")

        return f"""You are playing Among Us as a {role}.
Your name: {self.player.name}
Your goal: {goal}

=== YOUR PERSONALITY ===
{self.personality_desc}

=== GAME RULES ===
- Task phase: Move between rooms, complete tasks (Crewmates), or kill isolated targets (Impostors).
- Meeting phase: Discuss observations, then vote to eject a player (or skip).
- Crewmates win by completing all tasks OR ejecting all Impostors.
- Impostors win when they equal or outnumber Crewmates, or time runs out.
- You can only see players in the SAME room as you.
"""

    def _build_user_prompt(self, observation: Dict, available_actions: Dict, context: str = "") -> str:
        obs_text = self._observation_to_text(observation)
        memory_text = self.memory.to_context_string()

        if self.use_react:
            instruction = (
                "Analyze the situation and decide your action.\n"
                "Format your thinking as:\n"
                "[THOUGHT] What is my goal? What are my options?\n"
                "[PLAN] What should I do next and why?\n"
                "[ACTION] (Use the function tool to execute)"
            )
        else:
            instruction = "Choose your next action using the available function tools."

        prompt = f"""=== YOUR RECENT MEMORY ===
{memory_text}

=== CURRENT SITUATION ===
{obs_text}

{context}

{instruction}"""
        return prompt

    def _observation_to_text(self, obs: Dict) -> str:
        lines = [
            f"Phase: {obs['phase']} | Timestep: {obs['timestep']}/{obs['max_timesteps']}",
            f"Your location: {obs['my_location']}",
            f"Players here: {obs.get('co_located_players', [])}",
            f"Adjacent rooms: {obs.get('adjacent_rooms', [])}",
            f"Alive players: {obs.get('alive_players', [])}",
        ]
        if obs.get('dead_bodies_here'):
            lines.append(f"⚠️  Dead bodies here: {obs['dead_bodies_here']}")
        if obs.get('remaining_tasks'):
            lines.append(f"Remaining tasks: {obs['remaining_tasks']}")
            lines.append(f"Tasks here: {[t for t in obs.get('remaining_tasks', []) if TASK_LOCATIONS.get(t) == obs['my_location']]}")
        return "\n".join(lines)

    def act(self, observation: Dict, state: GameState, retry: int = 3) -> Dict:
        """Decide and return an action dict."""
        available = get_available_actions(self.player, state)
        tools = self._get_tools_for_phase(state.phase)

        system = self._build_system_prompt()
        user = self._build_user_prompt(observation, available)

        for attempt in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tools=tools,
                    tool_choice="required",
                    max_tokens=500,
                )
                msg = resp.choices[0].message

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    fn_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    action = self._parse_function_call(fn_name, args, state)
                    return action

            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(1)
                else:
                    print(f"[{self.player.name}] API error after {retry} attempts: {e}")

        # Fallback: random valid action
        return self._fallback_action(state)

    def speak(self, observation: Dict, state: GameState, discussion_history: List[str]) -> str:
        """Generate a meeting speech."""
        system = self._build_system_prompt()
        history_text = "\n".join(discussion_history[-10:]) if discussion_history else "No one has spoken yet."

        context = f"=== MEETING DISCUSSION SO FAR ===\n{history_text}"
        user = self._build_user_prompt(observation, {}, context)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=MEETING_PHASE_TOOLS,
                tool_choice="required",
                max_tokens=300,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                args = json.loads(msg.tool_calls[0].function.arguments)
                return args.get("speech", f"I have nothing to say right now.")
        except Exception as e:
            print(f"[{self.player.name}] speak error: {e}")

        return f"I've been watching everyone carefully."

    def vote(self, observation: Dict, state: GameState, discussion_history: List[str]) -> str:
        """Cast a vote. Returns player name or 'SKIP'."""
        alive_names = [p.name for p in state.alive_players() if p.name != self.player.name]
        system = self._build_system_prompt()
        history_text = "\n".join(discussion_history)
        context = (
            f"=== MEETING DISCUSSION ===\n{history_text}\n\n"
            f"=== VOTE ===\n"
            f"You must vote to eject someone or SKIP. Alive players: {alive_names}\n"
            f"Vote for the player you most suspect, or SKIP if unsure."
        )
        user = self._build_user_prompt(observation, {}, context)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=VOTE_TOOLS,
                tool_choice="required",
                max_tokens=100,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                args = json.loads(msg.tool_calls[0].function.arguments)
                target = args.get("target", "SKIP")
                if target not in alive_names and target != "SKIP":
                    target = "SKIP"
                return target
        except Exception as e:
            print(f"[{self.player.name}] vote error: {e}")

        return "SKIP"

    def _get_tools_for_phase(self, phase: Phase) -> List[Dict]:
        if phase == Phase.TASK:
            if self.player.role == Role.CREWMATE:
                return TASK_PHASE_TOOLS_CREW
            else:
                return TASK_PHASE_TOOLS_IMP
        else:
            return MEETING_PHASE_TOOLS

    def _parse_function_call(self, fn_name: str, args: Dict, state: GameState) -> Dict:
        """Convert function call to action dict."""
        action_map = {
            "move": {"type": "MOVE", "destination": args.get("destination", "")},
            "complete_task": {"type": "COMPLETE_TASK", "task": args.get("task", "")},
            "kill": {"type": "KILL", "target": args.get("target", "")},
            "report_body": {"type": "REPORT_BODY", "body": args.get("body", "")},
            "wait": {"type": "WAIT"},
            "speak": {"type": "SPEAK", "speech": args.get("speech", "")},
            "vote": {"type": "VOTE", "vote": args.get("target", "SKIP")},
        }
        return action_map.get(fn_name, {"type": "WAIT"})

    def _fallback_action(self, state: GameState) -> Dict:
        """Random valid action as fallback."""
        available = get_available_actions(self.player, state)
        if state.phase == Phase.TASK:
            adj = ADJACENCY[self.player.location]
            return {"type": "MOVE", "destination": __import__('random').choice(adj)}
        return {"type": "WAIT"}

    def update_memory(self, timestep: int, observation: Dict, action: Dict, result: str):
        self.memory.add(timestep, observation, action, result)


class RandomAgent:
    """Random-action baseline agent."""

    def __init__(self, player: Player):
        self.player = player

    def act(self, observation: Dict, state: GameState, **kwargs) -> Dict:
        available = get_available_actions(self.player, state)
        adj = ADJACENCY[self.player.location]

        if state.phase == Phase.TASK:
            # Impostors: occasionally kill if possible
            if self.player.role == Role.IMPOSTOR:
                targets = available.get("KILL", {}).get("targets", [])
                if targets and __import__('random').random() < 0.5:
                    return {"type": "KILL", "target": __import__('random').choice(targets)}

            # Crewmates: try to do tasks
            if self.player.role == Role.CREWMATE:
                doable = available.get("COMPLETE_TASK", {}).get("tasks", [])
                if doable:
                    return {"type": "COMPLETE_TASK", "task": __import__('random').choice(doable)}

                # Report bodies
                bodies = available.get("REPORT_BODY", {}).get("bodies", [])
                if bodies:
                    return {"type": "REPORT_BODY", "body": bodies[0]}

            return {"type": "MOVE", "destination": __import__('random').choice(adj)}

        return {"type": "WAIT"}

    def speak(self, observation: Dict, state: GameState, discussion_history: List[str]) -> str:
        templates = [
            "I was doing my tasks, nothing suspicious.",
            "I haven't seen anything weird yet.",
            "We should focus on the evidence.",
            "I don't know who it is yet.",
        ]
        return __import__('random').choice(templates)

    def vote(self, observation: Dict, state: GameState, discussion_history: List[str]) -> str:
        alive = [p.name for p in state.alive_players() if p.name != self.player.name]
        if alive and __import__('random').random() < 0.5:
            return __import__('random').choice(alive)
        return "SKIP"

    def update_memory(self, *args, **kwargs):
        pass
