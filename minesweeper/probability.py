import numpy as np

from minesweeper.board import FLAGGED, HIDDEN, Board

_NOT_HIDDEN_OR_FLAG = 99

MAX_COMPONENT_SIZE = 30


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _constraints(board: Board) -> list[tuple[list[int], int]]:
    """One constraint per opened number tile that still has hidden neighbours:
    (hidden neighbour tile ids, how many of them must be mines).
    """
    opened = np.nonzero(board.state > 0)[0]
    candidates = opened[board.values[opened] > 0]

    constraints = []
    for c in candidates:
        neighbours = board.neighbour_table[c]
        mask = board.neighbour_mask[c]
        neighbour_state = np.where(mask, board.state[neighbours], _NOT_HIDDEN_OR_FLAG)

        hidden = neighbours[neighbour_state == HIDDEN].tolist()
        flagged_count = int(np.count_nonzero(neighbour_state == FLAGGED))
        remaining = int(board.values[c]) - flagged_count

        if hidden and remaining > 0:
            constraints.append((hidden, remaining))
    return constraints


def _group_components(constraints: list[tuple[list[int], int]]) -> list[dict]:
    """Partition constraints into independent groups - tiles in one group
    never share a constraint with tiles in another, so they can be solved separately.
    """
    uf = _UnionFind()
    for hidden, _ in constraints:
        for t in hidden[1:]:
            uf.union(hidden[0], t)

    groups: dict[int, dict] = {}
    for hidden, target in constraints:
        root = uf.find(hidden[0])
        group = groups.setdefault(root, {"tiles": set(), "constraints": []})
        group["tiles"].update(hidden)
        group["constraints"].append((hidden, target))
    return list(groups.values())


def _enumerate_component(
    tiles: list[int], constraints: list[tuple[list[int], int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustively enumerate every mine/safe assignment to `tiles` consistent
    with `constraints`, pruning as soon as a partial assignment can no longer
    satisfy some constraint. Returns:
      totals[j]    = number of valid assignments using exactly j mines
      per_tile[t][j] = number of those assignments where tile t is a mine
    """
    n = len(tiles)
    index = {t: i for i, t in enumerate(tiles)}

    local_constraints = [([index[t] for t in hidden], target) for hidden, target in constraints]
    tile_constraints: list[list[int]] = [[] for _ in range(n)]
    for ci, (members, _) in enumerate(local_constraints):
        for t in members:
            tile_constraints[t].append(ci)

    sizes = [len(members) for members, _ in local_constraints]
    targets = [target for _, target in local_constraints]
    sum_assigned = [0] * len(local_constraints)
    assigned_count = [0] * len(local_constraints)

    assignment = [0] * n
    totals = np.zeros(n + 1, dtype=np.int64)
    per_tile = np.zeros((n, n + 1), dtype=np.int64)

    def assign(i: int, value: int) -> bool:
        assignment[i] = value
        valid = True
        for ci in tile_constraints[i]:
            sum_assigned[ci] += value
            assigned_count[ci] += 1
            needed = targets[ci] - sum_assigned[ci]
            remaining_unassigned = sizes[ci] - assigned_count[ci]
            if needed < 0 or needed > remaining_unassigned:
                valid = False
        return valid

    def unassign(i: int, value: int) -> None:
        for ci in tile_constraints[i]:
            sum_assigned[ci] -= value
            assigned_count[ci] -= 1

    def record() -> None:
        mines = sum(assignment)
        totals[mines] += 1
        for t, v in enumerate(assignment):
            if v:
                per_tile[t][mines] += 1

    def dfs(i: int) -> None:
        if i == n:
            record()
            return
        for value in (0, 1):
            if assign(i, value):
                dfs(i + 1)
            unassign(i, value)

    dfs(0)
    return totals, per_tile


def compute_probabilities(board: Board, max_component_size: int = MAX_COMPONENT_SIZE) -> dict[int, float]:
    """Exact mine probability for every tile bordering a numbered tile,
    computed independently per connected component of constraints.

    Note: this ignores the global "only N mines left on the whole board"
    coupling across components/unconstrained tiles - it treats each component
    as if mines were freely available. That has zero effect on which tiles
    come out as exactly 0.0 or 1.0 (those are true regardless), it can only
    bias the in-between probabilities slightly on boards with very few mines
    left relative to the board size. Components bigger than
    `max_component_size` are skipped (left out of the result) rather than
    risking a combinatorial blow-up.
    """
    constraints = _constraints(board)
    if not constraints:
        return {}

    probabilities: dict[int, float] = {}
    for group in _group_components(constraints):
        tiles = sorted(group["tiles"])
        if len(tiles) > max_component_size:
            continue

        totals, per_tile = _enumerate_component(tiles, group["constraints"])
        total_leaves = int(totals.sum())
        if total_leaves == 0:
            continue

        for local_i, tile in enumerate(tiles):
            probabilities[tile] = int(per_tile[local_i].sum()) / total_leaves

    return probabilities
