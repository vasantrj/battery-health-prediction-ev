<div align="center">

<img src="https://gist.githubusercontent.com/vasantrj/8b99ae4c904f870a92c25c9008a2fbb3/raw/banner.svg" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://battery-health-prediction-ev.streamlit.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/vasantrj/battery-health-prediction-ev?style=social)](https://github.com/vasantrj/battery-health-prediction-ev/stargazers)

<br/>

> **Predicting EV battery degradation before it becomes a problem — powered by ML, visualized with Streamlit.**

🌐 **Live Demo:** [battery-health-prediction-ev.streamlit.app](https://battery-health-prediction-ev.streamlit.app/)

</div>

---

## 📌 Overview

Electric vehicle batteries degrade silently over hundreds of charge-discharge cycles. By the time the drop in range is noticeable, significant capacity has already been lost.

This project tackles that problem by building an ML pipeline that estimates the **State of Health (SoH)** of EV batteries at any given cycle — using real NASA battery test data. The result is an interactive dashboard where you can explore predictions, visualize degradation trends, and catch early warning signs before they become expensive failures.

---

## 🧠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Persistence | Pickle |

---

## 📊 Dataset

**Source:** [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) — via Kaggle

Real battery test data collected by NASA across repeated charge-discharge cycles under controlled conditions.

> ⚠️ Not included in this repo (7565 CSV files + parquet). Download from the Kaggle link above before running locally.

---

## 🗂 Project Structure

battery_health_project/
├── src/                  # ML pipeline and utility modules
├── pages/                # Streamlit multi-page dashboard
├── models/               # Trained model (.pkl) + metrics.json
├── Screenshots/          # Dashboard screenshots
├── app.py                # Streamlit entry point
├── main.py               # Reserved for future use
├── requirements.txt      # Full dependency lockfile
├── requirements-min.txt  # Minimal install
└── .gitignore

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/vasantrj/battery-health-prediction-ev.git
cd battery-health-prediction-ev

# 2. Install dependencies
pip install -r requirements-min.txt   # recommended
# pip install -r requirements.txt     # exact environment

# 3. Run the app
streamlit run app.py
```

> Python **3.10 or 3.11** recommended. After cloning, download the NASA dataset from Kaggle and place it as expected by the pipeline.

---

## 📈 Results

The model predicts SoH degradation trends effectively across all cycle ranges in the NASA dataset. It generalizes well and can flag batteries that are degrading faster than the expected curve.

Full performance metrics are saved to `models/metrics.json` after training.

---

## 📸 Dashboard

<p align="center">
  <img src="Screenshots/Dashboard_1.png" width="45%" />
  &nbsp;
  <img src="Screenshots/Dashboard_2.png" width="45%" />
</p>

---

## 🔮 What's Next

- Deeper feature engineering around degradation inflection points
- Real-time data ingestion through IoT sensor integration
- Experiment with LSTM / Transformer models for time-series SoH forecasting
- Battery-level anomaly detection and alerting

---

## 🔗 Links

| | |
|---|---|
| 🌐 Live App | [battery-health-prediction-ev.streamlit.app](https://battery-health-prediction-ev.streamlit.app/) |
| 💻 GitHub | [github.com/vasantrj/battery-health-prediction-ev](https://github.com/vasantrj/battery-health-prediction-ev) |
| 📦 Dataset | [NASA Battery Dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) |

---

## 👨‍💻 Author

**Vasant Joshi** — Final Year CSE Student | Data Science Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-vasantjoshi-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasantjoshi)
[![GitHub](https://img.shields.io/badge/GitHub-vasantrj-181717?logo=github&logoColor=white)](https://github.com/vasantrj)

---

<p align="center">If this project helped you or you found it interesting, consider giving it a ⭐</p>
