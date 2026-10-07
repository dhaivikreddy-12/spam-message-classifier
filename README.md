# 📧 Spam Message Classifier

> *"Everyone hates spam. This project builds the thing that filters it."*

My first real NLP project. Instead of treating text as opaque, it converts messages into numbers using TF-IDF, then trains a Naive Bayes model to tell spam from ham. It was the moment I realized text data isn't magic — it just needs the right representation.

## What this project does

- Loads a labeled dataset of SMS-style messages.
- Cleans text (lowercasing, removing punctuation) with a small preprocessing pipeline.
- Converts text to numbers with **TF-IDF vectorization**.
- Trains **Multinomial Naive Bayes** and **Logistic Regression**.
- Compares them and evaluates with precision/recall.
- Classifies any new message you paste in.

## The dataset

I used a small, self-contained sample of messages (`data/messages.csv`) — roughly 700 messages, ~50% spam. Each row has:

| Column    | Description                       |
|-----------|-----------------------------------|
| `label`   | `spam` or `ham` (not spam)        |
| `message` | The raw text of the message       |

## How to run it

```bash
pip install -r requirements.txt

# Train + evaluate both models
python spam.py

# Classify your own message
python spam.py --text "Congratulations! You've won a free vacation. Click here to claim."

# Explore word frequencies
python explore.py
```

## What I learned

- How TF-IDF turns words into numbers without losing meaning.
- The difference between multinomial events and a bag of words.
- Why precision/recall matter more than accuracy for spam.
- That simple models (Naive Bayes) can be surprisingly strong on text.

## Results

**Multinomial Naive Bayes hits ~97% precision** — meaning when it flags spam, it's almost always right. Logistic regression is comparable. Both are good enough to be the engine behind a real filter.

---

*Built with Python, scikit-learn, pandas. Made for learning, by a student, for students.*
