import argparse

import torch
from torch.utils.data import DataLoader, random_split

from minesweeper.game import evaluate, generate_dataset
from minesweeper.network import TileClassifier, default_device, train

DIFFICULTIES = {
    "easy": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "hard": (16, 30, 99),
}

HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
INPUT_SIZE = 25


def add_difficulty_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default="hard")


def cmd_generate_data(args: argparse.Namespace) -> None:
    rows, cols, mines = DIFFICULTIES[args.difficulty]
    dataset = generate_dataset(rows, cols, mines, args.num_samples, seed=args.seed)
    torch.save(dataset, args.output)
    print(f"Saved {len(dataset)} samples to {args.output}")


def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else default_device()
    print(f"Training on: {device}")

    dataset = torch.load(args.dataset, weights_only=False)
    pin_memory = device.type == "cuda"

    val_dataloader = None
    if args.val_split > 0:
        n_val = int(len(dataset) * args.val_split)
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, pin_memory=pin_memory)
    else:
        train_set = dataset

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)

    model = TileClassifier(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
    model = train(
        model,
        dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        step_size=args.step_size,
        device=device,
        val_dataloader=val_dataloader,
    )

    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


def cmd_play(args: argparse.Namespace) -> None:
    rows, cols, mines = DIFFICULTIES[args.difficulty]

    model = None
    if args.model:
        model = TileClassifier(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        model.load_state_dict(torch.load(args.model, weights_only=True))
        model.eval()

    win_rate = evaluate(rows, cols, mines, args.num_games, model=model, seed=args.seed)
    print(f"Win rate over {args.num_games} games: {win_rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minesweeper solver: logic + ML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_gen = subparsers.add_parser("generate-data", help="play random games, label border tiles")
    add_difficulty_arg(p_gen)
    p_gen.add_argument("--num-samples", type=int, default=100_000)
    p_gen.add_argument("--output", default="dataset.pt")
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=cmd_generate_data)

    p_train = subparsers.add_parser("train", help="train the tile classifier on a dataset")
    p_train.add_argument("--dataset", default="dataset.pt")
    p_train.add_argument("--output", default="model.pth")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--lr-decay", type=float, default=0.5, help="StepLR gamma")
    p_train.add_argument("--step-size", type=int, default=10, help="StepLR step size, in epochs")
    p_train.add_argument("--batch-size", type=int, default=2048)
    p_train.add_argument("--val-split", type=float, default=0.1, help="fraction held out for validation, 0 to disable")
    p_train.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p_train.add_argument("--seed", type=int, default=None, help="seed for the train/val split")
    p_train.set_defaults(func=cmd_train)

    p_play = subparsers.add_parser("play", help="evaluate win rate (logic solver + optional model)")
    add_difficulty_arg(p_play)
    p_play.add_argument("--model", default=None, help="path to a trained model .pth (omit for logic-only / random)")
    p_play.add_argument("--num-games", type=int, default=100)
    p_play.add_argument("--seed", type=int, default=None)
    p_play.set_defaults(func=cmd_play)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
