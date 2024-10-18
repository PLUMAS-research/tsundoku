#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: ./tsundoku.sh <command> <date> <experiment>"
    echo "Commands: parse_json_to_parquet, import_date, compute_features, prepare_experiment, predict, annotate, consolidate, infer_communities, detect_anomalies, topic_model, generate"
    echo "Example: ./tsundoku.sh parse_json_to_parquet 20220501 workers_day"
    exit 1
fi

COMMAND=$1
DATE=$2
EXPERIMENT=$3

case $COMMAND in
    parse_json_to_parquet)
        python -m tsundoku.data.parse_json_to_parquet $DATE
        ;;
    import_date)
        python -m tsundoku.data.import_date $DATE
        ;;
    compute_features)
        python -m tsundoku.features.compute_features $DATE
        ;;
    prepare_experiment)
        python -m tsundoku.features.prepare_experiment $EXPERIMENT
        ;;
    predict)
        python -m tsundoku.models.predict $EXPERIMENT relevance
        python -m tsundoku.models.predict $EXPERIMENT stance
        ;;
    annotate)
        python -m tsundoku.models.annotate $EXPERIMENT relevance
        python -m tsundoku.models.annotate $EXPERIMENT stance
        ;;
    consolidate)
        python -m tsundoku.analysis.consolidate $EXPERIMENT stance
        ;;
    infer_communities)
        python -m tsundoku.analysis.infer_communities $EXPERIMENT
        ;;
    detect_anomalies)
        python -m tsundoku.analysis.detect_anomalies $EXPERIMENT stance
        ;;
    topic_model)
        python -m tsundoku.analysis.topic_model $EXPERIMENT
        ;;
    generate)
        python -m tsundoku.report.generate $EXPERIMENT stance
        echo "Report generated. Launching local server to view the report..."
        cd reports/$EXPERIMENT
        python -m http.server 8000 &
        SERVER_PID=$!
        sleep 2
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8000
        elif command -v open &> /dev/null; then
            open http://localhost:8000
        else
            echo "Couldn't open the browser automatically. Please visit http://localhost:8000 in your browser."
        fi
        echo "Press Ctrl+C to stop the server when you're done viewing the report."
        wait $SERVER_PID
        kill $SERVER_PID 2>/dev/null
        ;;
    *)
        echo "Invalid command. Available commands: parse_json_to_parquet, import_date, compute_features, prepare_experiment, predict, annotate, consolidate, infer_communities, detect_anomalies, topic_model, generate"
        exit 1
        ;;
esac