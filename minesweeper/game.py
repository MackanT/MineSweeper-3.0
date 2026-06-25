import numpy as np
import torch
from torch.utils.data import TensorDataset

from minesweeper.board import Board
from minesweeper.network import TileClassifier
from minesweeper.solver import solve

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
    """Play random games out to the point the solver - cheap logic plus exact
    CSP enumeration - can prove nothing more, and label every tile it's left
    genuinely undecided on (mine=0, safe=1). These are the only tiles where
    no amount of reasoning helps, which is exactly what the model has a
    chance of adding value on, both here and at inference time.
    """
    rng = np.random.default_rng(seed)
    view_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    collected = 0

    while collected < num_samples:
        board = new_game(rows, cols, n_mines, rng)
        probabilities = solve(board)

        if not probabilities:
            continue

        tiles = np.array(list(probabilities.keys()))
        view_chunks.append(board.views(tiles))
        label_chunks.append((~board.is_mine[tiles]).astype(np.int64))
        collected += tiles.size

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
    """Play one game to completion. The solver (logic + exact CSP enumeration)
    resolves everything it provably can; whatever's left is genuinely
    ambiguous. Without a model, the best play there is just the
    lowest-probability tile the solver already computed - no model can beat
    that on tiles the solver has already proven are irreducibly uncertain.
    With a model, defer to its most confident prediction instead. Returns
    True on a win.
    """
    rng = rng if rng is not None else np.random.default_rng()
    board = new_game(rows, cols, n_mines, rng)

    while not board.done:
        probabilities = solve(board)
        if board.done:
            break

        if not probabilities:
            hidden = board.hidden_tiles()
            if hidden.size == 0:
                break
            board.open_tile(int(rng.choice(hidden)))
            continue

        tiles = np.array(list(probabilities.keys()))

        if model is None:
            safest = min(probabilities, key=probabilities.get)
            board.open_tile(int(safest))
            continue

        views = board.views(tiles)
        with torch.no_grad():
            logits = model(torch.tensor(views, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()

        tile_local, cls = divmod(int(np.argmax(probs)), 2)
        target = int(tiles[tile_local])
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
