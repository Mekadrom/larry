import argparse
import glob
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

    argparser.add_argument(
        "--provenance_only",
        action='store_true',
        help="Avoids preprocessing again and just updates the provenance.json in the output directory. "
             "Force disables --clean."
    )

    args = argparser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    config_file = Path(args.config_file).expanduser().resolve()

    if args.provenance_only:
        setattr(args, "clean", False)

    if args.clean and output_dir.exists():
        log.info(f"cleaning up {output_dir}")
        # remove just split dirs containing parquets; leave readmes and anything else
        subdirs = [x for x in output_dir.iterdir() if x.is_dir()]
        for subdir in subdirs:
            shutil.rmtree(subdir, ignore_errors=True)
        # to be re-generated
        (output_dir / "provenance.json").unlink(missing_ok=True)
        # remove markers
        completed_files = glob.glob(str(output_dir / "*.completed"))
        for completed_file in completed_files:
            Path(completed_file).resolve().unlink()

    p = Pipeline(config_file, output_dir, clean=args.clean, provenance_only=args.provenance_only)
    p.run()
