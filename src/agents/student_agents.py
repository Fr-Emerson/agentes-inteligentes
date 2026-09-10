from __future__ import annotations

import random
from collections import defaultdict
from heapq import heappop, heappush
from typing import Any

from src.agents.base import Agent
from src.types import Action, Cell, Perception, Transition


MOVEMENT_ACTIONS = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
DELTAS = {
    Action.NORTH: (-1, 0), Action.SOUTH: (1, 0),
    Action.EAST: (0, 1), Action.WEST: (0, -1),
}


def _neighbor(position: tuple[int, int], action: Action) -> tuple[int, int]:
    dr, dc = DELTAS[action]
    return position[0] + dr, position[1] + dc


def _moves(perception: Perception) -> list[Action]:
    # Nos cenários fornecidos, todos os vizinhos válidos são visíveis.
    return [action for action in MOVEMENT_ACTIONS
            if perception.cell_at(_neighbor(perception.position, action))
            not in (None, Cell.WALL)]


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SimpleReflexAgent(Agent):
    """OBRIGATÓRIO: usar somente a percepção atual."""

    name = "simple"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        if perception.on_victim:
            return Action.RESCUE
        if perception.on_charger and perception.battery < perception.max_battery:
            return Action.RECHARGE
        moves = _moves(perception)
        if not moves:
            return Action.WAIT
        safe = [a for a in moves if perception.cell_at(
            _neighbor(perception.position, a)) != Cell.HAZARD]
        moves = safe or moves
        target_cell = (Cell.EXIT if perception.rescued == perception.total_victims
                       else Cell.VICTIM)
        if perception.battery < perception.max_battery * 0.3:
            target_cell = Cell.CHARGER
        targets = [p for p, cell in perception.visible_cells.items()
                   if cell == target_cell]
        # Um pouco de aleatoriedade evita ficar preso numa regra gulosa.
        if targets and self.rng.random() >= 0.2:
            scores = {a: min(_distance(_neighbor(perception.position, a), p)
                             for p in targets) for a in moves}
            best = min(scores.values())
            moves = [a for a in moves if scores[a] == best]
        return self.rng.choice(moves)

    def reset(self) -> None:
        pass


class ModelBasedAgent(Agent):
    """OBRIGATÓRIO: manter estado interno, sem exigir busca."""

    name = "model"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.known_map: dict[tuple[int, int], Cell] = {}
        self.visit_count: dict[tuple[int, int], int] = defaultdict(int)
        self.last_action: Action | None = None

    def _update_model(self, perception: Perception) -> None:
        # Sobrescreve inclusive vítimas que já foram resgatadas.
        self.known_map.update(perception.visible_cells)
        self.visit_count[perception.position] += 1

    def _routes(self, start: tuple[int, int]) -> dict:
        """Rotas no mapa observado, penalizando perigos conhecidos."""
        routes = {start: (0, None)}
        queue = [(0, start)]
        while queue:
            cost, position = heappop(queue)
            if cost != routes[position][0]:
                continue
            for action in MOVEMENT_ACTIONS:
                target = _neighbor(position, action)
                cell = self.known_map.get(target)
                if cell in (None, Cell.WALL):
                    continue
                # Custo heurístico de risco, sem consultar parâmetros ocultos.
                new_cost = cost + (5 if cell == Cell.HAZARD else 1)
                if new_cost < routes.get(target, (float("inf"), None))[0]:
                    first = action if position == start else routes[position][1]
                    routes[target] = (new_cost, first)
                    heappush(queue, (new_cost, target))
        return routes

    def act(self, perception: Perception) -> Action:
        self._update_model(perception)
        if perception.on_victim:
            action = Action.RESCUE
        elif perception.on_charger and perception.battery < perception.max_battery:
            action = Action.RECHARGE
        else:
            routes = self._routes(perception.position)
            reachable = [p for p in routes if p != perception.position]
            chargers = [p for p in reachable if self.known_map[p] == Cell.CHARGER]
            targets = []
            if chargers:
                nearest = min(chargers, key=lambda p: routes[p][0])
                reserve = max(perception.max_battery * 0.3, routes[nearest][0] + 5)
                if perception.battery <= reserve:
                    targets = [nearest]
            if not targets:
                desired = (Cell.EXIT if perception.rescued == perception.total_victims
                           else Cell.VICTIM)
                targets = [p for p in reachable if self.known_map[p] == desired]
            if not targets:
                # Explora células conhecidas ainda não visitadas.
                targets = [p for p in reachable if self.visit_count.get(p, 0) == 0]
            if targets:
                target = min(targets, key=lambda p: routes[p][0])
                action = routes[target][1]
            else:
                moves = _moves(perception)
                if moves:
                    score = {a: self.visit_count.get(_neighbor(perception.position, a), 0)
                             + (5 if perception.cell_at(_neighbor(perception.position, a))
                                == Cell.HAZARD else 0) for a in moves}
                    best = min(score.values())
                    action = self.rng.choice([a for a in moves if score[a] == best])
                else:
                    action = Action.WAIT
        self.last_action = action
        return action

    def diagnostics(self) -> dict[str, Any]:
        return {
            "known_cells": len(self.known_map),
            "visited_cells": len(self.visit_count),
        }


class LearningAgent(Agent):
    """OBRIGATÓRIO: sugestão de implementação com Q-learning tabular."""

    name = "learning"

    def __init__(
        self,
        seed: int = 0,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        evaluation_epsilon: float = 0.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.evaluation_epsilon = evaluation_epsilon
        self.training = True
        self.q: dict[tuple[Any, Action], float] = defaultdict(float)

    @property
    def exploration_rate(self) -> float:
        """Epsilon usado na fase atual do experimento."""
        return self.epsilon if self.training else self.evaluation_epsilon

    def reset(self) -> None:
        # Não apague self.q: o conhecimento deve persistir entre episódios.
        pass

    def _state(self, perception: Perception) -> Any:
        battery_band = min(4, 5 * perception.battery // max(1, perception.max_battery))
        neighborhood = tuple(perception.cell_at(_neighbor(perception.position, a))
                             for a in MOVEMENT_ACTIONS)
        return (perception.position, battery_band, neighborhood,
                perception.on_victim, perception.on_charger, perception.on_exit,
                perception.rescued, perception.total_victims)

    def _available_actions(self, perception: Perception) -> list[Action]:
        actions = _moves(perception)
        if perception.on_victim:
            actions.append(Action.RESCUE)
        if perception.on_charger and perception.battery < perception.max_battery:
            actions.append(Action.RECHARGE)
        return actions or [Action.WAIT]

    def act(self, perception: Perception) -> Action:
        actions = self._available_actions(perception)
        if self.rng.random() < self.exploration_rate:
            return self.rng.choice(actions)
        state = self._state(perception)
        values = {a: self.q.get((state, a), 0.0) for a in actions}
        best = max(values.values())
        return self.rng.choice([a for a in actions if values[a] == best])

    def observe_transition(self, transition: Transition) -> None:
        if not self.training:
            return
        state = self._state(transition.perception)
        key = (state, transition.action)
        future = 0.0
        if not transition.done:
            next_state = self._state(transition.next_perception)
            future = max(self.q.get((next_state, a), 0.0)
                         for a in self._available_actions(transition.next_perception))
        old = self.q.get(key, 0.0)
        self.q[key] = old + self.alpha * (transition.reward + self.gamma * future - old)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "q_entries": len(self.q),
            "q_states": len({state for state, _action in self.q}),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "evaluation_epsilon": self.evaluation_epsilon,
            "training": int(self.training),
        }


class GoalBasedAgent(Agent):
    """OPCIONAL: recomendado após busca/planejamento."""

    name = "goal"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        raise NotImplementedError("Extensão opcional: agente baseado em objetivos.")


class UtilityBasedAgent(Agent):
    """OPCIONAL: recomendado após busca e funções de utilidade."""

    name = "utility"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, perception: Perception) -> Action:
        raise NotImplementedError("Extensão opcional: agente baseado em utilidade.")
