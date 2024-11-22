import unittest
from pathlib import Path


class TestTsundoku(unittest.TestCase):
    def test_00_data_conversion(self):
        self.assertTrue(Path("test_data/sample_public").exists())
        from tsundoku.data.convert_json_to_parquet import main

        main.callback("20220501", 1, "auroracl_{}.data.gz", "", "")
        self.assertTrue(Path("test_data/parquet_public").exists())

    def test_01_data_import(self):
        from tsundoku.data.import_date import main

        main.callback("20220501", 1, "auroracl_{}.data.parquet", "")
        self.assertTrue(Path("softwarex/data/raw/2022-05-01").exists())

    def test_02_feature_computation(self):
        from tsundoku.features.compute_features import main

        main.callback("20220501", 1, True)
        self.assertTrue(Path("softwarex/data/interim/2022-05-01").exists())
        self.assertTrue(
            Path("softwarex/data/interim/2022-05-01/unique_users.parquet").exists()
        )

    def test_03_experimental_setup(self):
        from tsundoku.features.prepare_experiment import main

        main.callback("workers_day", True)
        self.assertTrue(Path("softwarex/data/processed/workers_day").exists())
        self.assertTrue(
            Path("softwarex/data/processed/workers_day/user.unique.parquet").exists()
        )

    def test_04_user_classifier(self):
        from tsundoku.models.predict import main

        main.callback("workers_day", "relevance", -1)
        self.assertTrue(
            Path(
                "softwarex/data/processed/workers_day/relevance.classification.predictions.parquet"
            ).exists()
        )
        main.callback("workers_day", "stance", -1)
        self.assertTrue(
            Path(
                "softwarex/data/processed/workers_day/stance.classification.predictions.parquet"
            ).exists()
        )

    def test_05_consolidated_analysis(self):
        from tsundoku.analysis.consolidate import main

        main.callback("workers_day", "stance", True)
        self.assertTrue(
            Path("softwarex/data/processed/workers_day/consolidated").exists()
        )
        self.assertTrue(
            Path(
                "softwarex/data/processed/workers_day/consolidated/user.consolidated_groups.parquet"
            ).exists()
        )

    def test_06_report(self):
        from tsundoku.report.generate import main

        main.callback("workers_day", "stance", 500)
        self.assertTrue(Path("softwarex/reports/workers_day/index.html").exists())


if __name__ == "__main__":
    unittest.main()
