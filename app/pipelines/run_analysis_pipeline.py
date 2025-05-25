import yaml
import logging
import subprocess
import time
import json

# Load config.yaml
with open("app/pipelines/configs/analysis_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Run steps based on config
for action in config["steps"]["actions"]:
    step_type = action["type"]
    is_execute = action.get("is_execute", False)
    file_input = action.get("file_input", "")
    file_output = action.get("file_output", "")

    step_commands = {
        "data_cleaning": f'python -m app.data_cleaning "{file_input}" "{file_output}" "{action.get("split_mode", "")}" {json.dumps(json.dumps(action.get("custom_stopwords", [])))}',
        "eda": f"python -m app.data_eda {file_input}",
        "topic_modeling": f"python -m app.topic_modelings.lda_topic_modeling {file_input} {file_output} {action.get('num_topics', 5)}",
        "topic_labeling": f"python -m app.topic_modelings.topics_labeling {file_input} {file_output} {json.dumps(json.dumps(action.get('candidate_labels', [])))}",
        "sentence_sentiment_analysis": {
            "vader": f"python -m app.analysis.vader_sentiment_analysis {file_input} {file_output}",
            "textblob": f"python -m app.analysis.textblob_sentiment_analysis {file_input} {file_output}",
            "bert": f"python -m app.analysis.bert_sentiment_analysis {file_input} {file_output}",
        },
        # "analyze_sentiment_topic_modeling": f"python -m app.topic_modelings.analyze_sentiment_topic_modeling {file_input} {file_output} {action.get('vectorizer_input', '')} {action.get('model_input', '')}",
    }

    if is_execute:
        if step_type == "sentence_sentiment_analysis":
            sentiment_method = action.get("method", "vader")
            command = step_commands["sentence_sentiment_analysis"].get(
                sentiment_method, ""
            )
        else:
            command = step_commands.get(step_type, "")

        if command:
            start_time = time.time()
            result = subprocess.run(command, shell=True)
            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                logging.error(f"Step {step_type} failed. Exiting pipeline.")
                break
            else:
                logging.info(
                    f"✅ Step {step_type} completed in {elapsed_time:.2f} seconds."
                )
        else:
            logging.warning(f"Invalid step configuration: {step_type}")
    else:
        logging.info(f"Skipping step: {step_type}")

logging.info("Pipeline execution completed.")
