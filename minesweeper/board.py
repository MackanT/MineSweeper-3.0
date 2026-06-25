import numpy as np

HIDDEN = 0
FLAGGED = -2
OUT_OF_BOUNDS = -1

_NEIGHBOUR_OFFSETS = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]
_WINDOW_OFFSETS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)]


def _build_offset_table(rows: int, cols: int, offsets: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lookup table mapping each tile to its offset neighbours.

    Returns (table, mask), both shape (rows*cols, len(offsets)). `table` holds
    flat tile indices (0 where invalid); `mask` marks which entries are in-bounds.
    """
    n = rows * cols
    row = np.arange(n) // cols
    col = np.arange(n) % cols

    table = np.zeros((n, len(offsets)), dtype=np.int32)
    mask = np.zeros((n, len(offsets)), dtype=bool)

    for k, (dr, dc) in enumerate(offsets):
        new_row = row + dr
        new_col = col + dc
        valid = (new_row >= 0) & (new_row < rows) & (new_col >= 0) & (new_col < cols)
        table[valid, k] = new_row[valid] * cols + new_col[valid]
        mask[:, k] = valid

    return table, mask


_TABLE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_tables(rows: int, cols: int):
    """Neighbour/window offset tables, cached per board shape (they're shape-only)."""
    key = (rows, cols)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        neighbour_table, neighbour_mask = _build_offset_table(rows, cols, _NEIGHBOUR_OFFSETS)
        window_table, window_mask = _build_offset_table(rows, cols, _WINDOW_OFFSETS)
        cached = (neighbour_table, neighbour_mask, window_table, window_mask)
        _TABLE_CACHE[key] = cached
    return cached


class Board:
    """Headless, vectorized Minesweeper board - everything is numpy array ops.
    """

    def __init__(self, rows: int, cols: int, n_mines: int, rng: np.random.Generator | None = None):
        self.rows = rows
        self.cols = cols
        self.n_tiles = rows * cols
        self.n_mines = n_mines
        self.rng = rng if rng is not None else np.random.default_rng()

        self.neighbour_table, self.neighbour_mask, self.window_table, self.window_mask = _get_tables(rows, cols)

        self.values = np.zeros(self.n_tiles, dtype=np.int8)
        self.state = np.full(self.n_tiles, HIDDEN, dtype=np.int8)
        self.is_mine = np.zeros(self.n_tiles, dtype=bool)
        self.done = False
        self.won = False

    def generate(self, safe_idx: int) -> None:
        """Place mines and compute neighbour counts, guaranteeing `safe_idx` is clear."""
        candidates = np.delete(np.arange(self.n_tiles), safe_idx)
        mine_idx = self.rng.choice(candidates, size=self.n_mines, replace=False)

        self.is_mine[:] = False
        self.is_mine[mine_idx] = True

        mine_grid = self.is_mine.reshape(self.rows, self.cols).astype(np.int8)
        padded = np.pad(mine_grid, 1)

        counts = np.zeros((self.rows, self.cols), dtype=np.int8)
        for dr, dc in _NEIGHBOUR_OFFSETS:
            counts += padded[1 + dr : 1 + dr + self.rows, 1 + dc : 1 + dc + self.cols]

        self.values = counts.reshape(self.n_tiles)
        self.values[mine_idx] = -1

    def _neighbour_state(self, idx: np.ndarray, oob_value: int) -> np.ndarray:
        neighbours = self.neighbour_table[idx]
        mask = self.neighbour_mask[idx]
        return np.where(mask, self.state[neighbours], oob_value), neighbours, mask

    def open_tile(self, idx: int) -> np.ndarray:
        """Open a tile (and flood-fill connected zero tiles). Returns flat indices opened.

        Sets `done`/`won` if this hits a mine or completes the board.
        """
        if self.done or self.state[idx] != HIDDEN:
            return np.array([], dtype=np.int64)

        if self.is_mine[idx]:
            self.state[idx] = self.values[idx] + 1
            self.done = True
            self.won = False
            return np.array([idx], dtype=np.int64)

        opened = self._flood_open(np.array([idx], dtype=np.int64))
        self._check_win()
        return opened

    def _flood_open(self, frontier: np.ndarray) -> np.ndarray:
        opened_chunks = []
        while frontier.size:
            frontier = frontier[self.state[frontier] == HIDDEN]
            if frontier.size == 0:
                break

            self.state[frontier] = self.values[frontier] + 1
            opened_chunks.append(frontier)

            zero_frontier = frontier[self.values[frontier] == 0]
            if zero_frontier.size == 0:
                break

            neighbours = self.neighbour_table[zero_frontier]
            mask = self.neighbour_mask[zero_frontier]
            candidates = np.unique(neighbours[mask])
            frontier = candidates[self.state[candidates] == HIDDEN]

        if not opened_chunks:
            return np.array([], dtype=np.int64)
        return np.concatenate(opened_chunks)

    def flag_tile(self, idx: int) -> bool:
        """Toggle a flag on a hidden tile. Returns True if state changed."""
        if self.done:
            return False
        if self.state[idx] == HIDDEN:
            self.state[idx] = FLAGGED
        elif self.state[idx] == FLAGGED:
            self.state[idx] = HIDDEN
        else:
            return False
        self._check_win()
        return True

    def _check_win(self) -> None:
        unopened = self.state <= HIDDEN
        if np.count_nonzero(unopened) == self.n_mines and np.all(self.is_mine[unopened]):
            self.done = True
            self.won = True

    def hidden_tiles(self) -> np.ndarray:
        return np.nonzero(self.state == HIDDEN)[0]

    def border_tiles(self) -> np.ndarray:
        """Hidden tiles adjacent to at least one opened tile - the only ones
        worth reasoning about (logically or via the model).
        """
        hidden = self.hidden_tiles()
        if hidden.size == 0:
            return hidden
        neighbour_state, _, mask = self._neighbour_state(hidden, OUT_OF_BOUNDS)
        has_opened_neighbour = np.any((neighbour_state > 0) & mask, axis=1)
        return hidden[has_opened_neighbour]

    def views(self, idx: np.ndarray) -> np.ndarray:
        """5x5 state window centered on each tile in `idx`, shape (len(idx), 25).

        Encoding: hidden=0, opened-with-value=value+1, flagged=-2, out-of-bounds=-1.
        """
        table = self.window_table[idx]
        mask = self.window_mask[idx]
        safe_table = np.where(mask, table, 0)
        return np.where(mask, self.state[safe_table], OUT_OF_BOUNDS)

    def as_grid(self, values: np.ndarray | None = None) -> np.ndarray:
        return (self.state if values is None else values).reshape(self.rows, self.cols)
