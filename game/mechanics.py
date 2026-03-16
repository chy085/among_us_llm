"""
Game mechanics: action execution and phase transitions.
"""

from typing import List, Dict, Optional, Tuple
from game.environment import GameState, Player, Phase, Role, ADJACENCY, TASK_LOCATIONS
import random


# ─── Action Definitions ──────────────────────────────────────────────────────

TASK_PHASE_ACTIONS_CREW = ["MOVE", "COMPLETE_TASK", "REPORT_BODY"]
TASK_PHASE_ACTIONS_IMP  = ["MOVE", "KILL"]
MEETING_PHASE_ACTIONS   = ["SPEAK", "VOTE"]


def get_available_actions(player: Player, state: GameState) -> Dict:
    """Return available actions and their parameters."""
    actions = {}

    if state.phase == Phase.TASK:
        # MOVE: go to adjacent room
        actions["MOVE"] = {"destinations": ADJACENCY[player.location]}

        if player.role == Role.CREWMATE:
            # COMPLETE_TASK: tasks available in current room
            doable = [t for t in player.tasks if TASK_LOCATIONS.get(t) == player.location]
            actions["COMPLETE_TASK"] = {"tasks": doable}

            # REPORT_BODY
            bodies = [n for n, loc in state.dead_bodies.items() if loc == player.location]
            if bodies:
                actions["REPORT_BODY"] = {"bodies": bodies}

        elif player.role == Role.IMPOSTOR:
            # KILL: crewmates in same room
            targets = [
                p.name for p in state.players_in_room(player.location)
                if p.name != player.name and p.role == Role.CREWMATE
            ]
            actions["KILL"] = {"targets": targets}

    return actions


# ─── Action Execution ─────────────────────────────────────────────────────────

class ActionResult:
    def __init__(self, success: bool, message: str, trigger_meeting: bool = False,
                 meeting_trigger_player: Optional[str] = None):
        self.success = success
        self.message = message
        self.trigger_meeting = trigger_meeting
        self.meeting_trigger_player = meeting_trigger_player


def execute_action(player: Player, action: Dict, state: GameState) -> ActionResult:
    """Execute an action and update game state."""
    action_type = action.get("type", "WAIT")

    if action_type == "MOVE":
        dest = action.get("destination")
        if dest and dest in ADJACENCY[player.location]:
            player.location = dest
            return ActionResult(True, f"{player.name} moved to {dest}.")
        return ActionResult(False, f"{player.name} tried invalid move to {dest}.")

    elif action_type == "COMPLETE_TASK":
        task = action.get("task")
        if task and player.complete_task(task):
            return ActionResult(True, f"{player.name} completed task: {task}.")
        return ActionResult(False, f"{player.name} could not complete task: {task}.")

    elif action_type == "KILL":
        target_name = action.get("target")
        target = state.get_player(target_name)
        if target and target.alive and target.location == player.location and target.role == Role.CREWMATE:
            target.alive = False
            state.dead_bodies[target_name] = player.location
            return ActionResult(True, f"{player.name} killed {target_name} in {player.location}.")
        return ActionResult(False, f"{player.name} kill attempt failed.")

    elif action_type == "REPORT_BODY":
        body = action.get("body")
        if body in state.dead_bodies and state.dead_bodies[body] == player.location:
            return ActionResult(True, f"{player.name} reported body of {body}.",
                                trigger_meeting=True, meeting_trigger_player=player.name)
        return ActionResult(False, f"{player.name} could not report body.")

    elif action_type == "WAIT":
        return ActionResult(True, f"{player.name} waited.")

    return ActionResult(False, f"Unknown action: {action_type}")


# ─── Meeting Phase ─────────────────────────────────────────────────────────────

class MeetingResult:
    def __init__(self, votes: Dict[str, str], ejected: Optional[str], tally: Dict[str, int]):
        self.votes = votes      # voter -> voted_for (or "SKIP")
        self.ejected = ejected  # player name or None
        self.tally = tally


def resolve_votes(votes: Dict[str, str], alive_players: List[str]) -> MeetingResult:
    """Count votes and determine ejection."""
    tally: Dict[str, int] = {}
    for voter, target in votes.items():
        if target != "SKIP":
            tally[target] = tally.get(target, 0) + 1

    if not tally:
        return MeetingResult(votes, None, tally)

    max_votes = max(tally.values())
    top = [p for p, v in tally.items() if v == max_votes]

    # Tie → no ejection
    if len(top) > 1:
        return MeetingResult(votes, None, tally)

    ejected = top[0]
    return MeetingResult(votes, ejected, tally)
