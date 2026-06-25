import numpy as np

from minesweeper import probability
from minesweeper.board import HIDDEN, FLAGGED, Board

_NOT_HIDDEN_OR_FLAG = 99  # sentinel for out-of-bounds neighbours: must not match HIDDEN/FLAGGED


def _opened_number_tiles(board: Board) -> np.ndarray:
    """Indices of opened tiles showing a number > 0 (the only tiles that
    constrain their neighbours).
    """
    opened = np.nonzero(board.state > 0)[0]
    return opened[board.values[opened] > 0]


def step(board: Board) -> bool:
    """Run a single deterministic constraint-propagation pass: open tiles that
    must be safe, flag tiles that must be mines. Returns True if anything changed.
    """
    candidates = _opened_number_tiles(board)
    if candidates.size == 0:
        return False

    neighbours = board.neighbour_table[candidates]
    mask = board.neighbour_mask[candidates]
    neighbour_state = np.where(mask, board.state[neighbours], _NOT_HIDDEN_OR_FLAG)

    hidden_mask = neighbour_state == HIDDEN
    flag_mask = neighbour_state == FLAGGED

    numbers = board.values[candidates]
    flag_count = flag_mask.sum(axis=1)
    hidden_count = hidden_mask.sum(axis=1)

    safe_rows = flag_count == numbers
    mine_rows = (hidden_count > 0) & (hidden_count + flag_count == numbers)

    to_open = np.unique(neighbours[hidden_mask & safe_rows[:, None]])
    to_flag = np.unique(neighbours[hidden_mask & mine_rows[:, None]])

    changed = False
    if to_flag.size:
        board.state[to_flag] = FLAGGED
        changed = True
    if to_open.size:
        for idx in to_open:
            board.open_tile(int(idx))
        changed = True

    return changed


def subset_step(board: Board) -> bool:
    """Cross-tile subset deduction: if numbered tile A's hidden neighbours are
    a subset of numbered tile B's hidden neighbours, the extra mines B needs
    beyond A's must lie entirely in B's "extra" tiles (the set difference).
    If that count is 0, the extras are all safe; if it equals the size of the
    difference, the extras are all mines. This catches cases plain per-tile
    counting (`step`) can't, e.g. the classic "1-2-1" pattern.
    """
    candidates = _opened_number_tiles(board)
    if candidates.size == 0:
        return False

    neighbours = board.neighbour_table[candidates]
    mask = board.neighbour_mask[candidates]
    neighbour_state = np.where(mask, board.state[neighbours], _NOT_HIDDEN_OR_FLAG)

    hidden_mask = neighbour_state == HIDDEN
    flag_mask = neighbour_state == FLAGGED
    remaining = board.values[candidates] - flag_mask.sum(axis=1)

    hidden_sets = [frozenset(neighbours[i][hidden_mask[i]].tolist()) for i in range(len(candidates))]

    # Group candidate tiles by which hidden tile they constrain, so we only
    # ever compare pairs that could plausibly share a subset relationship.
    groups: dict[int, list[int]] = {}
    for i, hs in enumerate(hidden_sets):
        if not hs or remaining[i] <= 0:
            continue
        for tile in hs:
            groups.setdefault(tile, []).append(i)

    to_open: set[int] = set()
    to_flag: set[int] = set()
    checked_pairs: set[tuple[int, int]] = set()

    for members in groups.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                pair = (i, j) if i < j else (j, i)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                hs_i, hs_j = hidden_sets[i], hidden_sets[j]
                if hs_i == hs_j:
                    continue
                if hs_i < hs_j:
                    small, big = i, j
                elif hs_j < hs_i:
                    small, big = j, i
                else:
                    continue

                diff = hidden_sets[big] - hidden_sets[small]
                diff_mines = remaining[big] - remaining[small]
                if diff_mines == 0:
                    to_open.update(diff)
                elif diff_mines == len(diff):
                    to_flag.update(diff)

    changed = False
    if to_flag:
        board.state[list(to_flag)] = FLAGGED
        changed = True
    if to_open:
        for idx in to_open:
            board.open_tile(int(idx))
        changed = True

    return changed


def solve_deterministic(board: Board, max_passes: int = 1000) -> None:
    """Repeatedly apply `step` (cheap, per-tile) and `subset_step` (cross-tile)
    until neither makes progress or the board is finished. `step` always runs
    first since it's cheaper and any subset deduction can unlock new basic
    deductions.
    """
    for _ in range(max_passes):
        if board.done:
            return
        if step(board):
            continue
        if not subset_step(board):
            return


def solve(board: Board, max_passes: int = 1000) -> dict[int, float]:
    """Full solve: cheap deterministic logic first, then exact CSP enumeration
    (`probability.compute_probabilities`) to squeeze out every remaining
    provably-safe/mine tile - that's strictly more powerful than `step`/
    `subset_step`, just more expensive, which is why it only runs once those
    two are stuck.

    Returns the mine probability of every tile that's still genuinely
    ambiguous after all of that (empty dict if the board ends up fully resolved).
    """
    for _ in range(max_passes):
        if board.done:
            return {}

        if step(board) or subset_step(board):
            continue

        probabilities = probability.compute_probabilities(board)
        certain_safe = [tile for tile, p in probabilities.items() if p == 0.0]
        certain_mine = [tile for tile, p in probabilities.items() if p == 1.0]

        if not certain_safe and not certain_mine:
            return {tile: p for tile, p in probabilities.items() if 0.0 < p < 1.0}

        if certain_mine:
            board.state[certain_mine] = FLAGGED
        for idx in certain_safe:
            board.open_tile(int(idx))

    return {}
