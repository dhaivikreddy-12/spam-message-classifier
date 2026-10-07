"""Train Naive Bayes and Logistic Regression on the real SMS Spam Collection."""
import argparse
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.load_data import load


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df = load()
print(f"Loaded {len(df)} messages ({int((df['label']=='spam').sum())} spam)")
df["message_clean"] = df["message"].apply(clean)

X = df["message_clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "NaiveBayes": MultinomialNB(alpha=0.1),
    "LogisticRegression": LogisticRegression(max_iter=2000, C=10),
}

best_name, best_f1, best_pipe = None, -1, None
for name, model in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("clf", model),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
    if f1_score(y_test, y_pred, pos_label="spam") > best_f1:
        best_name, best_f1, best_pipe = name, f1_score(y_test, y_pred, pos_label="spam"), pipe

print(f"Best by F1: {best_name} ({best_f1:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham")
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    if args.text:
        pred = best_pipe.predict([clean(args.text)])[0]
        prob = best_pipe.predict_proba([clean(args.text)])[0].max()
        print(f"\nThat message looks like: {pred} (confidence {prob:.1%})")
