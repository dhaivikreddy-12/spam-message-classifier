"""Train Naive Bayes and Logistic Regression for spam detection."""
import argparse
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv("data/messages.csv")
df["message_clean"] = df["message"].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df["message_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='spam'):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label='spam'):.3f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


def classify(text):
    vec = vectorizer.transform([clean(text)])
    model = LogisticRegression(max_iter=1000).fit(
        vectorizer.fit_transform(X_train), y_train
    )
    pred = model.predict(vec)[0]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham")
    parser.add_argument("--text", type=str, required=False)
    args = parser.parse_args()

    if args.text:
        label = classify(args.text)
        print(f"\nThat message looks like: {label}")
