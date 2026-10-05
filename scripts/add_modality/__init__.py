import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _make_module(sub_dir: Path) -> None:
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").touch(exist_ok=True)


class AddModalityTool:
    """Generates a directory or set of directories in the checked-in repository for ease of development."""

    sub_dirs_to_make: list[str] = ["config", "data", "model", "scripts", "scripts/training", "scripts/evaluation"]

    def __init__(self, name: str) -> None:
        """Instantiates a simple ModalityGenerator CLI object with the configured name to generate a modality for."""
        self.name = name

    def generate_modality_dirs(self) -> None:
        """Performs the actual generation of a modality directory or set of directories."""
        root = Path("src") / "larry" / self.name
        _make_module(root)
        [_make_module(root / sub) for sub in self.sub_dirs_to_make]


def main() -> None:
    """Starts the process of creating a new modality directory or directories to the checked in repository."""
    argparser = argparse.ArgumentParser(
        prog="add_modality",
        description="Add modality is a simple buildscript utility meant to be invoked via "
                    "`uv run python -m scripts.add_modality` in order to add a new set of modules for a new "
                    "modality in each concerned module. This includes a config, data, model, scripts/training, and "
                    "scripts/evaluation directories."
    )

    argparser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the modality to create modules for in the required parent modules."
    )

    args = argparser.parse_args()

    generator = AddModalityTool(args.name)
    log.warning(f"Creating modality modules for {args.name}...")
    generator.generate_modality_dirs()
    log.warning(f"Finished creating modality modules for {args.name}.")
