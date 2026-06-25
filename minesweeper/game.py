import numpy as np
import torch
from torch.utils.data import TensorDataset

from minesweeper.board import Board
from minesweeper.network import TileClassifier
from minesweeper.solver import solve_deterministic

VIEW_SIZE = 25


def new_game(rows: int, cols: int, n_mines: int, rng: np.random.Generator) -> Board:
    board = Board(rows, cols, n_mines, rng)
    start = int(rng.integers(board.n_tiles))
    board.generate(start)
    board.open_tile(start)
    return board


def generate_dataset(
    rows: int, cols: int, n_mines: int, num_samples: int, seed: int | None = None
) -> TensorDataset:
    """Play random games out to the point logic gets stuck, and label every
    remaining border tile (mine=0, safe=1). These ambiguous tiles are exactly
    what the model needs to learn - the deterministic solver already handles
    everything else, both here and at inference time.
    """
    rng = np.random.default_rng(seed)
    view_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    collected = 0

    while collected < num_samples:
        board = new_game(rows, cols, n_mines, rng)
        solve_deterministic(board)

        border = board.border_tiles()
        if border.size == 0:
            continue

        view_chunks.append(board.views(border))
        label_chunks.append((~board.is_mine[border]).astype(np.int64))
        collected += border.size

    views = np.concatenate(view_chunks)[:num_samples]
    labels = np.concatenate(label_chunks)[:num_samples]

    data = torch.tensor(views, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(data, targets)


def play_episode(
    rows: int,
    cols: int,
    n_mines: int,
    model: TileClassifier | None = None,
    rng: np.random.Generator | None = None,
) -> bool:
    """Play one game to completion. Logic solves what it can; remaining
    border tiles are picked by the model (most confident prediction first),
    or randomly if no model is given. Returns True on a win.
    """
    rng = rng if rng is not None else np.random.default_rng()
    board = new_game(rows, cols, n_mines, rng)

    while not board.done:
        solve_deterministic(board)
        if board.done:
            break

        border = board.border_tiles()
        if border.size == 0:
            hidden = board.hidden_tiles()
            if hidden.size == 0:
                break
            board.open_tile(int(rng.choice(hidden)))
            continue

        if model is None:
            board.open_tile(int(rng.choice(border)))
            continue

        views = board.views(border)
        with torch.no_grad():
            logits = model(torch.tensor(views, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()

        tile_local, cls = divmod(int(np.argmax(probs)), 2)
        target = int(border[tile_local])
        if cls == 1:
            board.open_tile(target)
        else:
            board.flag_tile(target)

    return board.won


def evaluate(
    rows: int,
    cols: int,
    n_mines: int,
    num_games: int,
    model: TileClassifier | None = None,
    seed: int | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    wins = sum(play_episode(rows, cols, n_mines, model, rng) for _ in range(num_games))
    return wins / num_games
