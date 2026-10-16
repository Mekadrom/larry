import argparse
import logging
from pathlib import Path

from larry.common.data.preprocessing.pipeline import Pipeline
from larry.common.utils.init_helpers import import_submodules

log = logging.getLogger(__name__)


def main() -> None:
    """Main entrypoint for all dataset preprocessing pipelines."""
    # forcibly walks the module structure to import all classes so registries for different class types are filled out
    import_submodules("larry")

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

    argparser.add_argument(
        "--provenance_only",
        action='store_true',
        help="Avoids preprocessing again and just updates the provenance.json in the output directory. "
             "Force disables --clean."
    )

    args = argparser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    config_file = Path(args.config_file).expanduser().resolve()

    # override the "clean" flag top False so provenance-only output actually survives
    if args.provenance_only:
        args.clean = False

    Pipeline(config_file, output_dir, clean=args.clean, provenance_only=args.provenance_only).run()
