import argparse
import logging
from pathlib import Path

from larry.common.training.training import Training
from larry.common.utils import set_seed_everywhere

log = logging.getLogger(__name__)


def main() -> None:
    """Main entrypoint for preprocessing data."""
    argparser = argparse.ArgumentParser(
        prog="training",
        description="Super entrypoint for all dataset preprocessing; delegates to implementations of Preprocessor "
                    "per-task.",
    )

    argparser.add_argument(
        "--config_file",
        type=Path,
        required=True,
        help="A path specifying the config_file.yaml to use for this preprocessing run."
    )

    argparser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="A path specifying where to save the run's logs, combined with --run_name."
    )

    argparser.add_argument(
        "--run_name",
        type=Path,
        required=True,
        help="A path specifying where to save the preprocessed dataset."
    )

    argparser.add_argument(
        "--seed",
        required=False,
        default=42,
        help="Random seed for literally every operation I could find to set the seed for."
    )

    argparser.add_argument(
        "--resume_dir",
        type=Path,
        required=False,
        help="A path specifying where to load a run and resume from. "
             "Directory must contain the output of a saved model's singular checkpoint."
    )

    args = argparser.parse_args()

    config_file = Path(args.config_file).expanduser().resolve()
    run_dir = Path(args.logdir).expanduser().resolve() / args.run_name

    set_seed_everywhere(args.seed)

    t = Training(config_file, run_dir, resume_dir=args.resume_dir)
    t.run()
