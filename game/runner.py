"""
Game runner: orchestrates a complete Among Us game.
Handles task phase, meeting phase, and logging.
"""

import random
from typing import List, Dict, Optional, Any, Union
from game.environment import GameState, Player, Phase, Role, create_game
from game.mechanics import execute_action, resolve_votes, ActionResult
from agents.llm_agent import LLMAgent, RandomAgent


AgentType = Union[LLMAgent, RandomAgent]


class GameLog:
    def __init__(self):
        self.events: List[Dict] = []
        self.meetings: List[Dict] = []
        self.final_state: Optional[Dict] = None

    def log_event(self, timestep: int, player: str, action: Dict, result: str):
        self.events.append({
            "timestep": timestep,
            "player": player,
            "action": action,
            "result": result,
        })

    def log_meeting(self, meeting_num: int, trigger: str, speeches: List[Dict],
                    votes: Dict, ejected: Optional[str], tally: Dict):
        self.meetings.append({
            "meeting_num": meeting_num,
            "trigger": trigger,
            "speeches": speeches,
            "votes": votes,
            "ejected": ejected,
            "vote_tally": tally,
        })

    def to_summary(self) -> Dict:
        return {
            "total_events": len(self.events),
            "total_meetings": len(self.meetings),
            "events": self.events,
            "meetings": self.meetings,
            "final_state": self.final_state,
        }


class GameRunner:
    def __init__(
        self,
        agents: Dict[str, AgentType],
        state: GameState,
        meeting_interval: int = 5,
        verbose: bool = True,
    ):
        self.agents = agents  # name -> agent
        self.state = state
        self.meeting_interval = meeting_interval
        self.verbose = verbose
        self.log = GameLog()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self) -> Dict:
        """Run a full game and return results."""
        self._log(f"\n{'='*60}")
        self._log("GAME START")
        self._log(f"Players: {[p.name for p in self.state.players]}")
        for p in self.state.players:
            self._log(f"  {p.name}: {p.role.value} | Tasks: {p.tasks}")
        self._log(f"{'='*60}\n")

        while not self.state.game_over:
            self.state.timestep += 1

            # Check time limit
            if self.state.timestep > self.state.max_timesteps:
                self.state.game_over = True
                self.state.winner = "Impostor"
                self._log("Time ran out! Impostors win.")
                break

            if self.state.phase == Phase.TASK:
                self._run_task_phase()
            else:
                self._run_meeting_phase()

            # Check victory
            winner = self.state.check_victory()
            if winner:
                self.state.game_over = True
                self.state.winner = winner
                self._log(f"\n🏆 {winner}s WIN! (Timestep {self.state.timestep})")
                break

            # Scheduled meeting every N timesteps
            if (self.state.phase == Phase.TASK and
                    self.state.timestep % self.meeting_interval == 0):
                self.state.phase = Phase.MEETING
                self.state.meeting_trigger = "scheduled"

        # Final stats
        result = self._compile_results()
        self.log.final_state = result
        return result

    def _run_task_phase(self):
        self._log(f"\n--- TASK PHASE | T={self.state.timestep} ---")
        meeting_triggered = False
        trigger_player = None

        for player in self.state.alive_players():
            if self.state.game_over:
                break

            agent = self.agents.get(player.name)
            if not agent:
                continue

            obs = self.state.observe(player)
            action = agent.act(obs, self.state)

            result = execute_action(player, action, self.state)
            self._log(f"  {player.name} [{player.role.value[:3]}]: {result.message}")

            agent.update_memory(self.state.timestep, obs, action, result.message)
            self.log.log_event(self.state.timestep, player.name, action, result.message)

            if result.trigger_meeting:
                meeting_triggered = True
                trigger_player = result.meeting_trigger_player

            # Check victory after each kill
            winner = self.state.check_victory()
            if winner:
                self.state.game_over = True
                self.state.winner = winner
                return

        if meeting_triggered and not self.state.game_over:
            self.state.phase = Phase.MEETING
            self.state.meeting_trigger = trigger_player or "body_report"

    def _run_meeting_phase(self):
        self.state.meeting_count += 1
        trigger = self.state.meeting_trigger or "scheduled"
        self._log(f"\n{'='*40}")
        self._log(f"MEETING #{self.state.meeting_count} triggered by: {trigger}")
        self._log(f"{'='*40}")

        alive = self.state.alive_players()
        discussion_history: List[str] = []
        speeches: List[Dict] = []

        # 2 discussion rounds
        for round_num in range(2):
            self._log(f"\n  [Discussion Round {round_num + 1}]")
            for player in alive:
                agent = self.agents.get(player.name)
                if not agent:
                    continue
                obs = self.state.observe(player)
                speech = agent.speak(obs, self.state, discussion_history)
                statement = f"{player.name}: {speech}"
                discussion_history.append(statement)
                speeches.append({"player": player.name, "speech": speech, "round": round_num + 1})
                self._log(f"    {statement}")

        # Voting round
        self._log(f"\n  [Vote]")
        votes: Dict[str, str] = {}
        for player in alive:
            agent = self.agents.get(player.name)
            if not agent:
                votes[player.name] = "SKIP"
                continue
            obs = self.state.observe(player)
            vote = agent.vote(obs, self.state, discussion_history)
            votes[player.name] = vote
            self._log(f"    {player.name} votes: {vote}")

        # Resolve
        result = resolve_votes(votes, [p.name for p in alive])
        if result.ejected:
            ejected_player = self.state.get_player(result.ejected)
            if ejected_player:
                ejected_player.alive = False
                self._log(f"\n  ❌ {result.ejected} was ejected! (Role: {ejected_player.role.value})")
        else:
            self._log(f"\n  No ejection (tie or all skips).")

        self.log.log_meeting(
            self.state.meeting_count, trigger, speeches, votes,
            result.ejected, result.tally
        )

        self.state.phase = Phase.TASK
        self.state.meeting_trigger = None

    def _compile_results(self) -> Dict:
        alive_crew = len(self.state.alive_crewmates())
        alive_imp = len(self.state.alive_impostors())
        total_kills = sum(1 for e in self.log.events if e["action"].get("type") == "KILL" and "killed" in e["result"])
        impostor_name = next((p.name for p in self.state.players if p.role == Role.IMPOSTOR), None)

        return {
            "winner": self.state.winner,
            "timesteps": self.state.timestep,
            "total_meetings": self.state.meeting_count,
            "kills": total_kills,
            "alive_crewmates": alive_crew,
            "alive_impostors": alive_imp,
            "impostor_ejected": impostor_name and not self.state.get_player(impostor_name).alive,
            "log": self.log.to_summary(),
        }


def build_agents(
    state: GameState,
    config: str,
    memory_size: int = 10,
    use_react: bool = True,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, AgentType]:
    """
    Build agent dict for a given configuration.
    config: "all_random" | "all_llm" | "llm_crew" | "llm_imp"
    """
    agents = {}
    for player in state.players:
        use_llm = False
        if config == "all_llm":
            use_llm = True
        elif config == "llm_crew" and player.role == Role.CREWMATE:
            use_llm = True
        elif config == "llm_imp" and player.role == Role.IMPOSTOR:
            use_llm = True

        if use_llm:
            agents[player.name] = LLMAgent(
                player=player,
                memory_size=memory_size,
                use_react=use_react,
                model=model,
                api_key=api_key,
            )
        else:
            agents[player.name] = RandomAgent(player=player)

    return agents


def run_single_game(
    config: str,
    player_names: List[str] = None,
    memory_size: int = 10,
    use_react: bool = True,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run a single game with given configuration."""
    if player_names is None:
        player_names = ["Alice", "Bob", "Charlie", "Diana", "Evan"]

    state = create_game(player_names, n_impostors=1)
    agents = build_agents(state, config, memory_size, use_react, model, api_key)
    runner = GameRunner(agents, state, verbose=verbose)
    return runner.run()
