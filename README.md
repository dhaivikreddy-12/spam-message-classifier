# 📧 Spam Message Classifier

> *"Everyone hates spam. This is the thing that filters it — trained on 5,572 real SMS messages."*

My first real NLP project. Instead of treating text as opaque strings, it converts messages into numbers with TF-IDF, then trains a Naive Bayes model to separate spam from genuine messages. It's the project where I stopped assuming text data is magic and started respecting the preprocessing.

## What this project does

- Loads the real UCI SMS Spam Collection (5,572 messages, 747 spam).
- Cleans text with a small normalisation pipeline (lowercase, strip punctuation, collapse whitespace).
- Converts text to numbers using **TF-IDF** with bigrams.
- Trains Multinomial Naive Bayes and Logistic Regression.
- Reports accuracy, precision, recall, and F1 for spam specifically.
- Includes a CLI to classify any message you paste in.

## The dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 real SMS messages from the UK, 747 of them spam (13.4%).

| Column | Description |
|---|---|
| `label` | `spam` or `ham` — **target** |
| `message` | The raw SMS text |

## How to run it

```bash
pip install -r requirements.txt

python spam.py

# Classify your own message
python -m src.train_model --text "Congratulations! You have won a free iPhone. Claim your prize now at http://spam.example.com"
```

## Project structure

```
spam-message-classifier/
├── data/
│   └── messages.csv
├── src/
│   ├── load_data.py   # download UCI zip + cache
│   └── train_model.py # clean, vectorise, train, evaluate, CLI
├── tests/
├── spam.py
├── requirements.txt
└── README.md
```

## What I learned

- That text preprocessing is most of the work — the model choice matters less.
- Why TF-IDF beats raw counts: rare words in a message carry more signal than common ones.
- That bigrams ("free money") help even on a small dataset.
- Why you should measure recall on the spam class specifically. Overall accuracy hides a model that misses half the spam.

## Results

20% stratified test split (1,115 test messages, ~150 spam):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **NaiveBayes** | **0.9821** | **0.9778** | **0.8859** | **0.9296** |
| LogisticRegression | 0.9776 | 0.9769 | 0.8523 | 0.9104 |

Naive Bayes wins on F1 (0.930). It's also the model a real spam filter would probably ship, because it's fast, tiny, and — as this result shows — genuinely hard to beat on text.

---

*Built with Python, pandas, scikit-learn, matplotlib. Real SMS data, honestly measured.*
