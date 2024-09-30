from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import os
from tsundoku.utils.files import read_toml
import logging
import pandas as pd
import dask
from multiprocessing.pool import ThreadPool


class TsundokuApp(object):
    def __init__(self, name):
        load_dotenv(find_dotenv(usecwd=True), override=True)

        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        self.logger = logging.getLogger(__name__)

        self.tsundoku_path = Path(os.getcwd())

        self.name = name
        self.project_path = Path(os.environ["TSUNDOKU_PROJECT_PATH"])
        self.config = read_toml(self.project_path / "config.toml")["project"]
        self.data_path = Path(self.config["settings"]["data_path"])

        experiment_file = self.project_path / "experiments.toml"

        if not experiment_file.exists():
            raise FileNotFoundError(experiment_file)

        self.experiment_config = read_toml(experiment_file)

        pd.set_option("display.max_colwidth", None)

        dask.config.set(pool=ThreadPool(int(self.config.get("n_jobs", 2))))
        dask.config.set({"dataframe.convert-string": False})

    def read_group_config(self, group):
        if not group in self.experiment_config:
            self.experiment_config[group] = {}

        self.group_config = read_toml(self.project_path / "groups" / f"{group}.toml")

        # print(experiment_config[group])
        self.group_options = set(
            self.experiment_config[group].get("order", self.group_config.keys())
        )
        try:
            self.group_options.remove("undisclosed")
        except KeyError:
            pass
