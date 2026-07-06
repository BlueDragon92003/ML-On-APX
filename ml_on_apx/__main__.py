"""The entry point for the application."""

import argparse
import os
from pathlib import Path

import ml_on_apx.cluster_classification.main
import ml_on_apx.dataset_management.app
from ml_on_apx.cluster_classification.cluster_classification_dataset import (
    ClusterClassificationDataset,
)
from ml_on_apx.logging import initialize_file_logging
from ml_on_apx.modes import Mode

LL_DEVELOPMENT = "DEBUG"
LL_PRODUCTION = "INFO"
LL_SILENT = "SILENT"
LOG_LEVELS = [LL_DEVELOPMENT, LL_PRODUCTION, LL_SILENT]

MODE_CLASSIFICATION: str = "classification"
MODE_IDENTIFICATION: str = "identification"
MODES = [MODE_CLASSIFICATION, MODE_IDENTIFICATION]

SUBCOMMAND_TRAIN: str = "train"
SUBCOMMAND_MNG_DATA: str = "manage data"
SUBCOMMAND_MNG_MODEL: str = "manage model"


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser.

    Returns:
        argparse.ArgumentParser: The argument parser.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--application-dir",
        help="Specify a different working directory than `./``",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Specify a different directiory to store data in than \
            `{application-dir}/data/`",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-dir",
        help="Specify a different directiory to store models in than \
            `{application-dir}/models/`",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--console-log-level",
        type=str,
        choices=LOG_LEVELS,
        default="ERROR",
        help="Specify the console logging level.",
    )
    parser.add_argument(
        "--log-to", type=Path, default=None, help="Specify the file to log over."
    )
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a regular file, append to the log file instead of overwriting it. \
                (Default: append)",
    )
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(subcommand=SUBCOMMAND_TRAIN)
    train_parser.add_argument(
        "mode",
        choices=MODES,
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
        choices=MODES,
        help="Specify whether to run the job set for classification or identification.",
    )
    manage_subparsers = manage_parser.add_subparsers()

    manage_data_parser = manage_subparsers.add_parser("data")
    manage_data_parser.set_defaults(subcommand=SUBCOMMAND_MNG_DATA)

    manage_models_parser = manage_subparsers.add_parser("models")
    manage_models_parser.set_defaults(subcommand=SUBCOMMAND_MNG_MODEL)

    return parser


#     Parse args:
#     - manage_datasets
#     - manage_models
#     - train model

#     Manage datasets:
#         Load dataset index
#         Show list of datasets
#         - Create
#         - Readdata
#         - Update
#         - Delete

#     Manage models:
#         Load model index
#         Show list of models & checkpoints
#         - Save checkpoints
#         - Delete models
#         - Test models
#         - Define model
#             - Datasets & labels
#             - Inputs
#             - Hyperparemeters
#         - Create model training job
#             - Stop conditions (n epochs, min accuracy, min accuracy growth rate)


#     Train model:
#         Fork for each job
#         Load relavant information
#         Train until stop conditions
def main() -> int:
    """Start the application."""
    argparser = get_parser()
    args = argparser.parse_args()

    if "subcommand" not in args:
        argparser.print_help()
        return -1

    application_dir: Path = (
        args.application_dir if args.application_dir is not None else Path(os.getcwd())
    )
    os.environ["APP_DIR"] = str(application_dir)
    data_dir: Path = (
        args.data_dir if args.data_dir is not None else application_dir / "data"
    )
    model_dir: Path = (
        args.model_dir if args.model_dir is not None else application_dir / "model"
    )

    log_path: Path = application_dir / "logs"
    if args.log_file is not None:
        if os.path.isfile(args.log_file) or not os.path.exists(args.log_file):
            log_path: Path = args.log_file
        else:
            print(f"{args.log_dir} must not exist or be a regular file.", end="\n\n")
            argparser.print_help()
            return -1
    elif args.log_dir is not None:
        if os.path.isdir(args.log_dir):
            log_path: Path = args.log_dir
        else:
            print(f"{args.log_dir} is not a directory.", end="\n\n")
            argparser.print_help()
            return -1

    initialize_file_logging(log_path, append=args.append)

    mode: Mode
    match args.mode:
        case argsmode if argsmode == MODE_CLASSIFICATION:
            mode = Mode.Classification
        case argsmode if argsmode == MODE_IDENTIFICATION:
            mode = Mode.Identification

    match args.subcommand:
        case subcommand if subcommand == SUBCOMMAND_TRAIN:
            match mode:
                case mode if mode is Mode.Classification:
                    print("`train classification` not yet implemeneted")
                    return 0
                    ml_on_apx.cluster_classification.main.main(data_dir, model_dir)
                case mode if mode is Mode.Identification:
                    print("`train identification` not yet implemeneted")
                    return 0
        case subcommand if subcommand == SUBCOMMAND_MNG_DATA:
            match mode:
                case m if m == Mode.Classification:
                    dataset_class = ClusterClassificationDataset
                case m if m == Mode.Identification:
                    print("`manage identification data` not yet implemented")
                    return 0
            ml_on_apx.dataset_management.app.main(
                data_dir,
                mode,
                dataset_class,
            )
            return 0
        case subcommand if subcommand == SUBCOMMAND_MNG_MODEL:
            print("`manage ___ models` not yet implemeneted")
            return 0
            ml_on_apx.dataset_management.app.main(model_dir, mode)
        case _:
            argparser.print_help()
            return -1

    return 0


if __name__ == "__main__":
    main()
