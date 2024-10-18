# Tsundoku Workflow

This README provides instructions for setting up and running the Tsundoku workflow.

## Setup

1. Make sure you're in the directory containing the `tsundoku.sh` script.

2. Make the script executable:
   ```bash
   chmod +x tsundoku.sh
   ```

## Usage

Replace `20220501` with your actual date and `workers_day` with your experiment name in the following commands:

1. Parse JSON to Parquet:
   ```bash
   ./tsundoku.sh parse_json_to_parquet 20220501 workers_day
   ```

2. Import Date:

   ```bash
   ./tsundoku.sh import_date 20220501 workers_day
   ```

3. Compute Features:

   ```bash
   ./tsundoku.sh compute_features 20220501 workers_day
   ```

4. Prepare Experiment:

   ```bash
   ./tsundoku.sh prepare_experiment 20220501 workers_day
   ```

5. Predict:

   ```bash
   ./tsundoku.sh predict 20220501 workers_day
   ```

6. Annotate: 

   ```bash
   ./tsundoku.sh annotate 20220501 workers_day
   ```

7. Consolidate data:

   ```bash
   ./tsundoku.sh consolidate 20220501 workers_day
   ```

8. Infer Communities:

   ```bash
   ./tsundoku.sh infer_communities 20220501 workers_day
   ```

9. Detect Anomalies:

   ```bash
   ./tsundoku.sh detect_anomalies 20220501 workers_day
   ```

10. Topic Model:

    ```bash
    ./tsundoku.sh topic_model 20220501 workers_day
    ```

11. Generate Report and View:

    ```bash
    ./tsundoku.sh generate 20220501 workers_day
    ```

    This command will generate the report and start a local server to view it. It will attempt to open your default web browser automatically. If it doesn't, manually navigate to `http://localhost:8000` in your web browser.
