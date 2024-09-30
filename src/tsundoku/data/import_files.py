import logging
import os
import click

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from tsundoku.data.importer import TweetImporter
from tsundoku.utils.config import TsundokuApp

@click.command()
@click.argument("source", type=click.Path(exists=True), nargs=-1)
@click.argument("target", type=str)
def main(source, target):
    app = TsundokuApp('File Importer')

    project = TweetImporter(app.project_path / "config.toml")
    app.logger.info(str(project.config))

    target_path = app.data_path / "raw" / target
    project.import_files(source, target_path)


if __name__ == "__main__":
    main()
