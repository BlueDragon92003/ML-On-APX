from pathlib import Path
import argparse


def handle_training_process(args: argparse.Namespace):
    print("training!")
    pass


def handle_data_management(args: argparse.Namespace):
    print("data manager")
    pass


def handle_model_management(args: argparse.Namespace):
    print("model manager")
    pass


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--application-dir",
        help="Specify a different working directory than `./``",
        type=Path,
    )
    parser.add_argument(
        "--data-dir",
        help="Specify a different directiory to store data in than `{application-dir}/data/`",
        type=Path,
    )
    parser.add_argument(
        "--model-dir",
        help="Specify a different directiory to store models in than `{application-dir}/models/`",
        type=Path,
    )
    parser.add_argument(
        "--log-to-console", action="store_true", help="Enables console logging."
    )
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(subcommand=handle_training_process)
    train_parser.add_argument(
        "mode",
        choices=["classification", "identification"],
        help="Specify whether to run the job set for classification or identification.",
    )
    (
        train_parser.add_argument(
            "--tui",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable or disable the TUI. (Default: enabled)",
        ),
    )

    manage_parser = subparsers.add_parser("manage")
    manage_parser.add_argument(
        "mode",
        choices=["classification", "identification"],
        help="Specify whether to run the job set for classification or identification.",
    )
    manage_subparsers = manage_parser.add_subparsers()

    manage_data_parser = manage_subparsers.add_parser("data")
    manage_data_parser.set_defaults(subcommand=handle_data_management)

    manage_models_parser = manage_subparsers.add_parser("models")
    manage_models_parser.set_defaults(subcommand=handle_model_management)

    return parser


def main():
    """
    Parse args:
    - manage_datasets
    - manage_models
    - train model

    Manage datasets:
        Load dataset index
        Show list of datasets
        - Create
        - Readdata
        - Update
        - Delete

    Manage models:
        Load model index
        Show list of models & checkpoints
        - Save checkpoints
        - Delete models
        - Test models
        - Define model
            - Datasets & labels
            - Inputs
            - Hyperparemeters
        - Create model training job
            - Stop conditions (n epochs, min accuracy, min accuracy growth rate)

    Train model:
        Fork for each job
        Load relavant information
        Train until stop conditions

    """

    argparser = get_parser()
    args = argparser.parse_args()

    if "subcommand" not in args:
        argparser.print_help()
        return -1

    args.subcommand(args)

    return 0


if __name__ == "__main__":
    main()
