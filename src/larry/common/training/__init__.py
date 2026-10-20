import argparse
import logging
from pathlib import Path

from larry.common.config.config import ConfigValues
from larry.common.config.training.definition import TrainingDefinition
from larry.common.training.training_run import TrainingRun
from larry.common.utils import reexec_with_hash_seed, set_seed_everywhere
from larry.common.utils.git_utils import WorkingTree
from larry.common.utils.init_helpers import import_submodules

log = logging.getLogger(__name__)


def main() -> None:
    """Main entrypoint for training models."""
    argparser = argparse.ArgumentParser(
        prog="training",
        description="Super entrypoint for all dataset preprocessing; delegates to implementations of Preprocessor "
                    "per-task.",
    )

    ConfigValues.add_arguments(argparser)

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
        type=int,
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
    reexec_with_hash_seed(args.seed)
    WorkingTree.ensure_clean(args.allow_dirty)

    import_submodules("larry")
    definition = ConfigValues.from_args(args).load(TrainingDefinition)
    run_dir = Path(args.logdir).expanduser().resolve() / args.run_name

    set_seed_everywhere(args.seed)

    TrainingRun(definition, run_dir, resume_dir=args.resume_dir).run()
