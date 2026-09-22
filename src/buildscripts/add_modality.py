#!./venv/bin/python

import argparse


class ModalityGenerator:
    def __init__(self, modality_name: str) -> None:
        self.modality_name = modality_name

    def generate_modality_dirs(self) -> None:
        pass


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(
        prog="add-modality",
        description="Add modality is a simple buildscript utility meant to be invoked via "
                    "`uv run python -m buildscripts.add_modality` in order to add a new set of modules for a new modality"
                    "in each concerned module. This includes a config, data, model, scripts/train, and scripts/eval"
                    "directories."
    )

    argparse.add_argument(
        "--module_name",
        type=str,
        required=True,
        help="The entire point of the command, the name of the modality to create modules for in the required parent modules."
    )

    args = argparse.parse_args()

    generator = ModalityGenerator(args.module_name)

    generator.generate_modality_dirs()
