"""
Among Us Game Environment
Implements the 14-room map and game state management.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import random


class Phase(Enum):
    TASK = "TASK"
    MEETING = "MEETING"


class Role(Enum):
    CREWMATE = "Crewmate"
    IMPOSTOR = "Impostor"


# 14-room map adjacency graph
ROOMS = [
    "Cafeteria", "Weapons", "O2", "Navigation",
    "MedBay", "Upper Engine", "Admin", "Shields",
    "Comms", "Electrical", "Storage", "Lower Engine",
    "Reactor", "Security"
]

ADJACENCY = {
    "Cafeteria":     ["Weapons", "MedBay", "Admin", "Storage"],
    "Weapons":       ["Cafeteria", "O2"],
    "O2":            ["Weapons", "Navigation", "Shields"],
    "Navigation":    ["O2", "Shields"],
    "MedBay":        ["Cafeteria", "Upper Engine"],
    "Upper Engine":  ["MedBay", "Reactor", "Security"],
    "Admin":         ["Cafeteria", "Storage", "Comms"],
    "Shields":       ["O2", "Navigation", "Comms"],
    "Comms":         ["Admin", "Shields", "Electrical"],
    "Electrical":    ["Comms", "Storage", "Lower Engine"],
    "Storage":       ["Cafeteria", "Admin", "Electrical", "Lower Engine"],
    "Lower Engine":  ["Electrical", "Storage", "Reactor"],
    "Reactor":       ["Upper Engine", "Lower Engine", "Security"],
    "Security":      ["Upper Engine", "Reactor"],
}

# Task locations (Crewmate tasks)
TASK_LOCATIONS = {
    "Fix Wiring":      "Electrical",
    "Download Data":   "Admin",
    "Clear Asteroids": "Weapons",
    "Calibrate Distributor": "Navigation",
    "Empty Garbage":   "O2",
    "Submit Scan":     "MedBay",
    "Fuel Engines":    "Upper Engine",
    "Align Engine Output": "Lower Engine",
    "Stabilize Steering": "Navigation",
    "Prime Shields":   "Shields",
    "Monitor Tree":    "O2",
    "Fix Weather Node": "Comms",
    "Divert Power":    "Electrical",
    "Inspect Sample":  "MedBay",
}


@dataclass
class Player:
    name: str
    role: Role
    location: str = "Cafeteria"
    alive: bool = True
    tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)

    def assign_tasks(self, n: int = 3):
        task_list = list(TASK_LOCATIONS.keys())
        random.shuffle(task_list)
        self.tasks = task_list[:n]

    def complete_task(self, task: str) -> bool:
        if task in self.tasks and self.location == TASK_LOCATIONS[task]:
            self.tasks.remove(task)
            self.completed_tasks.append(task)
            return True
        return False

    def all_tasks_done(self) -> bool:
        return len(self.tasks) == 0


@dataclass
class GameState:
    players: List[Player]
    phase: Phase = Phase.TASK
    timestep: int = 0
    max_timesteps: int = 20
    dead_bodies: Dict[str, str] = field(default_factory=dict)  # name -> room
    meeting_trigger: Optional[str] = None  # who triggered the meeting
    ejected_player: Optional[str] = None
    game_over: bool = False
    winner: Optional[str] = None  # "Crewmate" or "Impostor"
    meeting_count: int = 0

    def get_player(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def alive_players(self) -> List[Player]:
        return [p for p in self.players if p.alive]

    def alive_crewmates(self) -> List[Player]:
        return [p for p in self.players if p.alive and p.role == Role.CREWMATE]

    def alive_impostors(self) -> List[Player]:
        return [p for p in self.players if p.alive and p.role == Role.IMPOSTOR]

    def players_in_room(self, room: str) -> List[Player]:
        return [p for p in self.players if p.alive and p.location == room]

    def observe(self, player: Player) -> Dict:
        """Return what a player can observe."""
        room = player.location
        co_located = [p.name for p in self.players_in_room(room) if p.name != player.name]
        bodies_here = [name for name, loc in self.dead_bodies.items() if loc == room]
        adj_rooms = ADJACENCY[room]

        obs = {
            "phase": self.phase.value,
            "timestep": self.timestep,
            "max_timesteps": self.max_timesteps,
            "my_name": player.name,
            "my_role": player.role.value,
            "my_location": room,
            "co_located_players": co_located,
            "dead_bodies_here": bodies_here,
            "adjacent_rooms": adj_rooms,
            "alive_players": [p.name for p in self.alive_players() if p.name != player.name],
            "all_alive_players": [p.name for p in self.alive_players()],
        }
        if player.role == Role.CREWMATE:
            obs["remaining_tasks"] = player.tasks
            obs["completed_tasks"] = player.completed_tasks

        return obs

    def check_victory(self) -> Optional[str]:
        alive_crew = self.alive_crewmates()
        alive_imp = self.alive_impostors()

        # Impostors win if they equal or outnumber crewmates
        if len(alive_imp) >= len(alive_crew):
            return "Impostor"

        # Impostors win if time runs out
        if self.timestep >= self.max_timesteps:
            return "Impostor"

        # Crewmates win if all impostors ejected
        if len(alive_imp) == 0:
            return "Crewmate"

        # Crewmates win if all tasks completed
        all_tasks_done = all(
            p.all_tasks_done() for p in self.alive_crewmates()
        )
        if all_tasks_done and len(alive_crew) > 0:
            return "Crewmate"

        return None


def create_game(player_names: List[str], n_impostors: int = 1, tasks_per_player: int = 3) -> GameState:
    """Initialize a new game."""
    assert len(player_names) >= n_impostors + 2, "Need at least impostor + 2 crewmates"

    roles = [Role.IMPOSTOR] * n_impostors + [Role.CREWMATE] * (len(player_names) - n_impostors)
    random.shuffle(roles)

    players = []
    for name, role in zip(player_names, roles):
        p = Player(name=name, role=role)
        if role == Role.CREWMATE:
            p.assign_tasks(tasks_per_player)
        players.append(p)

    return GameState(players=players)
