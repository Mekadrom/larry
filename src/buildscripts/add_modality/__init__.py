import argparse
import logging

log = logging.getLogger(__name__)


class AddModalityTool:
    """Generates a directory or set of directories in the checked-in repository for ease of development."""

    def __init__(self, name: str) -> None:
        """Instantiates a simple ModalityGenerator CLI object with the configured name to generate a modality for."""
        self.name = name

    def generate_modality_dirs(self) -> None:
        """Performs the actual generation of a modality directory or set of directories."""
        pass


def main() -> None:
    """Starts the process of creating a new modality directory or directories to the checked in repository."""
    argparser = argparse.ArgumentParser(
        prog="add_modality",
        description="Add modality is a simple buildscript utility meant to be invoked via "
                    "`uv run python -m buildscripts.add_modality` in order to add a new set of modules for a new "
                    "modality in each concerned module. This includes a config, data, model, scripts/train, and "
                    "scripts/eval directories."
    )

    argparser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the modality to create modules for in the required parent modules."
    )

    args = argparser.parse_args()

    generator = AddModalityTool(args.name)
    log.info(f"Creating modality modules for {args.name}...")
    generator.generate_modality_dirs()
    log.info(f"Finished creating modality modules for {args.name}.")
