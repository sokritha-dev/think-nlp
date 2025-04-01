import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Argument parser for file input and output
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Path to cleaned dataset CSV file")
parser.add_argument("feature_input", type=str, help="Path to TF-IDF features file")
parser.add_argument("model_output", type=str, help="Path to save trained model")
args = parser.parse_args()


class LogisticRegressionTrainer:
    def __init__(self, file_input, feature_input):
        """Initialize model training with dataset and features."""
        self.df = pd.read_csv(file_input)

        # Convert rating to sentiment labels
        self.df["sentiment"] = self.df["rating"].apply(self.assign_sentiment)
        self.labels = self.df["sentiment"]

        # Load TF-IDF features
        with open(feature_input, "rb") as f:
            self.X = pickle.load(f)

        self.model = LogisticRegression(max_iter=1000)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.labels, test_size=0.2, random_state=42
        )

    @staticmethod
    def assign_sentiment(rating):
        """Convert numerical rating into categorical sentiment."""
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    def train_model(self):
        """Train Logistic Regression model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the trained model."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        classification_rep = classification_report(self.y_test, y_pred)
        return accuracy, classification_rep

    def save_model(self, filename):
        """Save the trained model."""
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    trainer = LogisticRegressionTrainer(args.file_input, args.feature_input)
    trainer.train_model()

    accuracy, classification_rep = trainer.evaluate_model()
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_rep)

    trainer.save_model(args.model_output)
    print(f"✅ Trained model saved successfully at {args.model_output}!")
