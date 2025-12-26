import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

btc = pd.read_csv("btc.csv", index_col=0)
btc = btc[btc.index != "Ticker"]
btc.index = pd.to_datetime(btc.index, errors="coerce", format="mixed")
btc = btc[btc.index.notna()]

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc = btc[["Open", "High", "Low", "Close", "Volume"]]
btc.columns = [c.lower() for c in btc.columns]
btc = btc.apply(pd.to_numeric, errors="coerce").dropna()

btc_w = btc.resample("W").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
})

wiki = pd.read_csv("wikipedia_edits.csv", index_col=0)
wiki.index = pd.to_datetime(wiki.index, errors="coerce", format="mixed")
wiki = wiki.apply(pd.to_numeric, errors="coerce").dropna()
wiki_w = wiki.resample("W").mean()

data = btc_w.merge(wiki_w, left_index=True, right_index=True, how="inner")

data["return"] = data["close"].pct_change()
data["target"] = (data["return"].shift(-1) > 0).astype(int)
data = data.dropna()

features = ["close", "volume", "edit_count", "sentiment_mean", "neg_sentiment_ratio"]

for h in [2, 4, 12]:
    r = data.rolling(h, closed="left")
    data[f"close_ratio_{h}"] = data["close"] / r["close"].mean()
    data[f"attention_{h}"] = r["edit_count"].mean()
    features += [f"close_ratio_{h}", f"attention_{h}"]

data = data.dropna()

split = int(len(data) * 0.7)
train = data.iloc[:split]
test = data.iloc[split:]

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=6,
    min_samples_split=20,
    random_state=42
)

model.fit(train[features], train["target"])

latest = data.iloc[-1:][features]
prob_up = model.predict_proba(latest)[0][1]

direction = "UP " if prob_up >= 0.5 else "DOWN "
confidence = round(max(prob_up, 1 - prob_up) * 100, 2)

current_price = round(data["close"].iloc[-1], 2)

print("Current BTC Price:", current_price)
print("Prediction:", direction)
print("Confidence Level:", f"{confidence}%")
