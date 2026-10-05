import argparse
import logging
import shutil
from pathlib import Path

from larry.common.data.preprocessing.pipeline import Pipeline
from larry.common.utils.init_helpers import import_submodules

log = logging.getLogger(__name__)


def main() -> None:
    import_submodules("larry")

    """Main entrypoint for preprocessing data."""
    argparser = argparse.ArgumentParser(
        prog="preprocessing",
        description="Super entrypoint for all dataset preprocessing; delegates to implementations of Preprocessor "
                    "per-task.",
    )

    argparser.add_argument(
        "--config_file",
        type=Path,
        required=True,
        help="A path specifying the config.yaml to use for this preprocessing run."
    )

    argparser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="A path specifying where to save the preprocessed dataset."
    )

    argparser.add_argument(
        "--clean",
        action='store_true',
        help="Whether to clean the output and intermediate cache prior to preprocessing."
    )

    args = argparser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    config_file = Path(args.config_file).expanduser().resolve()

    if args.clean:
        log.info(f"cleaning up {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    p = Pipeline(config_file, output_dir, clean=args.clean)
    p.run()
