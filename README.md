# 2025ML-spamEmail

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spame-mail.streamlit.app/)

垃圾郵件（Spam Email）分類專案：使用 scikit-learn 建立完整文字處理與分類流程，包含前處理、特徵抽取、模型訓練、指標與視覺化，以及 Typer CLI 與 Streamlit 互動式網頁介面。

來源參考：本專案延伸自 Packt《Hands-On Artificial Intelligence for Cybersecurity》書中第 3 章的 Spam Email 問題設計與資料集模式。

## 功能（Features）

- 端到端文字處理：`TfidfVectorizer` + Logistic Regression 或 Naive Bayes
- 擴增前處理：移除 URL/Email、轉小寫、去除符號與數字、空白壓縮
- 指標：Accuracy、Precision、Recall、F1、ROC AUC、混淆矩陣
- 視覺化：混淆矩陣與 ROC 曲線
- CLI：訓練、評估、預測、模型匯出
- Streamlit：資料預覽、訓練、指標與即時預測

## 專案結構（Repo Structure）

```
.
├─ app.py                     # Streamlit app entry
├─ cli.py                     # Typer CLI entry
├─ requirements.txt           # Dependencies for local + Streamlit Cloud
├─ .gitignore
├─ data/
│  └─ sample_spam.csv         # Small sample for quick demos
├─ artifacts/                 # Saved models/plots (created at runtime)
├─ src/
│  └─ spam_email/
│     ├─ __init__.py
│     ├─ data.py
│     ├─ preprocess.py
│     ├─ model.py
│     └─ visualize.py
└─ README.md
```

## 快速開始（Quickstart）

1) Create and activate a Python 3.9+ environment

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) 透過 CLI 訓練與評估

```
python cli.py train --data-path data/sample_spam.csv --model-out artifacts/model.joblib
python cli.py evaluate --data-path data/sample_spam.csv --model-path artifacts/model.joblib
python cli.py predict --model-path artifacts/model.joblib --text "Congratulations! You've won a prize!"
```

3) 本機執行 Streamlit

```
streamlit run app.py
```

Then open the URL shown in the console (usually http://localhost:8501).

## 推到 GitHub（Deploy to GitHub）

目標倉庫（示例）：https://github.com/handsomedudu/hw3

Steps:

```
git init
git add -A
git commit -m "Initial commit: spam email project"
git branch -M main
git remote add origin git@github.com:huanchen1107/2025ML-spamEmail.git
git push -u origin main
```

If using HTTPS instead of SSH, replace the `git remote add origin` line accordingly.

## 部署到 Streamlit Cloud

目標 Demo： https://spame-mail.streamlit.app/

1) 使用 GitHub 帳號登入 https://share.streamlit.io/
2) 建立新 App，選取你的倉庫 `handsomedudu/hw3`
3) 將 `Main file path` 設為 `app.py`
4) 確認 Python 3.9+ 與 `requirements.txt` 存在
5) 部署後會自動安裝依賴並啟動 App

教學影片（參考）：https://www.youtube.com/watch?v=ANjiJQQIBo0

Note: If you want to use a larger dataset, place it in the repo (e.g., `data/spam.csv`) or host it publicly and load it from the app. The app and CLI default to the included `data/sample_spam.csv`.

## 資料集（Dataset）

By default, this repo includes a small sample file at `data/sample_spam.csv` for demo. To improve performance, replace with a full dataset like the UCI SMS Spam Collection. The code expects a CSV with at least:

- `label` column: values like `spam` or `ham`
- `text` column: the message body

## 致謝（Acknowledgements）

- Inspired by Packt's "Hands-On Artificial Intelligence for Cybersecurity" (Chapter 3 spam email problem)
