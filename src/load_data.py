"""Load the real UCI SMS Spam Collection (5,572 SMS, 747 spam).

Source: UCI Machine Learning Repository, dataset id 228. Cached to
data/messages.csv so the repo runs offline after the first run.
"""
import io
import os
import urllib.request
import zipfile
import pandas as pd

CSV_PATH = "data/messages.csv"
ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)

    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    payload = urllib.request.urlopen(req, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(payload))
    with zf.open("SMSSpamCollection") as fh:
        df = pd.read_csv(fh, sep="\t", names=["label", "message"], encoding="utf-8")

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} messages -> {CSV_PATH}")
    print(df["label"].value_counts().to_dict())
