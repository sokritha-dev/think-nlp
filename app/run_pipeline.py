import yaml
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load config.yaml
with open("app/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Run steps based on config
for action in config["steps"]["actions"]:
    step_type = action["type"]
    is_execute = action.get("is_execute", False)
    file_input = action.get("file_input", "")
    file_output = action.get("file_output", "")

    step_commands = {
        "data_cleaning": f"python -m app.data_cleaning -- {file_input} {file_output}",
        "eda": f"python -m app.data_eda -- {file_input}",
        "feature_extraction": {
            "tfidf": f"python -m app.feature_extraction.tfidf_vectorizer -- {file_input} {action.get('feature_output', '')} {action.get('vector_output', '')}",
            "word2vec": f"python -m app.feature_extraction.word2vec_vectorizer -- {file_input} {action.get('feature_output', '')} {action.get('vector_output', '')}",
            "glove": f"python -m app.feature_extraction.glove_vectorizer -- {file_input} {action.get('feature_output', '')} {action.get('vector_output', '')}",
            "bert": f"python -m app.feature_extraction.bert_vectorizer -- {file_input} {action.get('feature_output', '')} {action.get('vector_output', '')}",
        },
        "train_model": {
            "logistic_regression": f"python -m app.models.logistic_regression {file_input} {action.get('feature_input', '')} {action.get('model_output', '')}",
        },
        "predict_sentiment": {
            "logistic_regression": f'python -m app.models.predictions.predict_logistic_regression "{action.get("file_input", "")}" {action.get("model_input", "")} {action.get("vector_input", "")} {action.get("file_output", "")}',
        },
        "topic_modeling": f"python -m app.topic_modelings.lda_topic_modeling {file_input} {file_output} {action.get('num_topics', 5)}",
    }

    if is_execute:
        if step_type == "feature_extraction":
            feature_method = action.get("method", "tfidf")  # Default to TF-IDF
            command = step_commands["feature_extraction"].get(feature_method, "")
        elif step_type == "train_model":
            model_method = action.get(
                "method", "logistic_regression"
            )  # Default to Logistic Regression
            command = step_commands["train_model"].get(model_method, "")
        elif step_type == "predict_sentiment":
            predict_method = action.get(
                "method", "logistic_regression"
            )  # Default to Logistic Regression
            command = step_commands["predict_sentiment"].get(predict_method, "")
        else:
            command = step_commands.get(step_type, "")

        if command:
            logging.info(
                f"Running step: {step_type} ({feature_method if step_type == 'feature_extraction' else predict_method if step_type == 'predict_sentiment' else ''})..."
            )
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
