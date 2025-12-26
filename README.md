#  Bitcoin Price Direction Predictor using Wikipedia Attention

This project predicts whether **Bitcoin’s price will go UP or DOWN over the next week**, along with a **confidence level**, using **Wikipedia attention metrics** and historical market data.

The goal is not to predict exact prices, but to explore whether **public attention**, captured via Wikipedia activity, contains useful information about future price direction.

---

##  Problem Statement

Cryptocurrency markets are highly volatile and noisy, making short-term price prediction extremely challenging.

This project addresses the question:

> *Can Wikipedia attention help predict the weekly direction of Bitcoin prices?*

---

##  Approach

The system combines:
- Historical Bitcoin price data
- Wikipedia edit volume and sentiment data

Key design choices:
- Weekly aggregation to reduce noise
- Directional prediction (UP / DOWN)
- Probabilistic confidence output
- Lightweight, reproducible implementation

---

## Features Used

### Market Data
- Weekly closing price
- Weekly trading volume

### Wikipedia Attention Metrics
- Edit count
- Average sentiment score
- Negative sentiment ratio

### Engineered Features
- Rolling price ratios (2, 4, 12 weeks)
- Rolling Wikipedia attention trends

---

##  Model

- **Algorithm**: Random Forest Classifier  
- **Prediction Output**:
  - Price direction: **UP / DOWN **
  - Confidence level (%)

The confidence represents the model’s estimated probability for the predicted direction.

---

## Output Example
Current BTC Price: 87292.97
Prediction: DOWN 
Confidence Level: 63.41%

## Evaluation Notes

- The model is trained on historical data using a time-aware split.
- Performance varies across market regimes due to the stochastic nature of crypto markets.
- Directional confidence values above 60% indicate relatively stronger signals.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn

---

## Project Structure

btc-wikipedia-predictor/
├── final_model.py
├── btc.csv
├── wikipedia_edits.csv
├── README.md
├── requirements.txt
├── .gitignore

yaml
Copy code

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
Process
Bitcoin price data and Wikipedia activity data are collected separately.

Both datasets are aligned to a weekly frequency.

Features are engineered using rolling windows.

A Random Forest classifier is trained on historical data.

The model outputs price direction and confidence for the next week.

Disclaimer
This project is for educational and research purposes only.
It does not constitute financial or investment advice.