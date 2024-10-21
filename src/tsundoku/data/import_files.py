import click

from tsundoku.data.importer import TweetImporter
from tsundoku.utils.config import TsundokuApp

@click.command("import_files_into_date")
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
