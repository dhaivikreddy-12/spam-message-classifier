"""Real tests: verify text cleaning and vectorisation behaviour on real logic."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_model import clean  # noqa: E402


def test_readme_and_license_present():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()


def test_clean_lowercases():
    assert clean("URGENT Free Prize") == "urgent free prize"


def test_clean_strips_punctuation():
    assert clean("Claim now!! (100% free)") == "claim now 100 free"


def test_clean_collapses_whitespace():
    assert clean("hello    world\t\tagain") == "hello world again"


def test_clean_is_idempotent():
    once = clean("WIN  a  prize-- NOW!! http://x.com")
    assert clean(once) == once


def test_clean_keeps_digits():
    assert "1000" in clean("Win 1000 dollars today")


def test_clean_removes_url_punctuation():
    """Non-alphanumerics become spaces, so a URL is split into bare words."""
    assert clean("visit http://spam.com now") == "visit http spam com now"


def test_tfidf_pipeline_fits_and_predicts():
    """A real end-to-end fit/predict on a handful of strings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])
    docs = [
        "free money click now claim prize",
        "win a free iphone today",
        "urgent you have won money",
        "see you at the cafe tomorrow",
        "dinner at seven please",
        "can you send the notes",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]
    pipe.fit([clean(d) for d in docs], labels)
    preds = pipe.predict([clean("free prize click now"), clean("cafe tomorrow")])
    assert preds[0] == "spam"
    assert preds[1] == "ham"
