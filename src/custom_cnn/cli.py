from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras

from . import data as data_mod
from . import model as model_mod
from . import training as training_mod
from . import evaluation as evaluation_mod


def cmd_train(args: argparse.Namespace) -> None:
    (x_train, y_train), (x_test, y_test) = data_mod.load_cifar10()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_mod.preprocess(
        x_train, y_train, x_test, y_test, val_size=args.val_size
    )

    model = model_mod.build_model()
    training_mod.compile_model(model)

    print("Fit model on training data")
    training_mod.fit(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.es_patience,
        reduce_lr_patience=args.rlrop_patience,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    model.save(model_path)
    print(f"Saved model to: {model_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    model = keras.models.load_model(args.model)
    (x_train, y_train), (x_test, y_test) = data_mod.load_cifar10()
    (_, _), (_, _), (x_test, y_test) = data_mod.preprocess(
        x_train, y_train, x_test, y_test, val_size=args.val_size
    )

    print("Evaluate on test data")
    results = evaluation_mod.evaluate(model, x_test, y_test, batch_size=args.batch_size)
    print("test loss, test acc:", results)

    if args.report:
        preds = evaluation_mod.predict(model, x_test)
        report = evaluation_mod.classification_report_from_predictions(y_test, preds)
        print("\nClassification Report:\n", report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="custom-cnn", description="Train and evaluate the custom CNN on CIFAR-10")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--val-size", type=int, default=10_000)
    p_train.add_argument("--es-patience", type=int, default=5)
    p_train.add_argument("--rlrop-patience", type=int, default=3)
    p_train.add_argument("--output-dir", type=str, default="models")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate a saved model")
    p_eval.add_argument("--model", type=str, required=True, help="Path to saved .keras or .h5 model")
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--val-size", type=int, default=10_000)
    p_eval.add_argument("--report", action="store_true", help="Print sklearn classification report")
    p_eval.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
