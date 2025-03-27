import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset for training
reviews = ["Amazing hotel!", "Terrible experience.", "The room was clean but the service was slow."]
labels = [1, 0, 1]  # 1 = Positive, 0 = Negative

# Step 1: Convert reviews into numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

# Step 2: Train Logistic Regression Model
clf = LogisticRegression()
clf.fit(X, labels)

# Step 3: Save the trained model and vectorizer
joblib.dump(clf, "app/models/sentiment_model.pkl")
joblib.dump(vectorizer, "app/models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
