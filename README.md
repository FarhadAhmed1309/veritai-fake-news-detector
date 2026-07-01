# veritai-fake-news-detector
AI-powered fake news detector using TF-IDF + Logistic Regression
# ⚡ VeritAI — Fake News Detector

VeritAI is an AI-powered web application that analyzes news articles and predicts whether they are **REAL** or **FAKE**, using a Machine Learning model trained with TF-IDF and Logistic Regression. Along with the verdict, it provides a detailed credibility breakdown, sentiment analysis, keyword highlighting, and a downloadable analysis report.

---

## 🚀 Features

- **Instant Verdict** — Paste any news article and get a REAL/FAKE prediction with confidence score
- **Credibility Spectrum** — Visual gauge showing how likely the article is real or fake
- **Credibility Breakdown** — Score across multiple factors: Language, Sentiment, Length, Clickbait, and ML Model confidence
- **Sentiment Analysis** — Polarity and subjectivity scoring of the article
- **Keyword Scan** — Highlights sensational and clickbait words/phrases found in the text
- **Reasoning Panel** — Explains *why* the model reached its verdict
- **File Upload Support** — Upload `.txt` files directly for analysis
- **Sample Articles** — Try built-in REAL and FAKE sample articles for a quick demo
- **Session History** — Tracks all analyses performed in the current session
- **Downloadable Report** — Export a full analysis report as a `.txt` file
- **Copy to Clipboard** — One-click copy of the generated report
- **Modern Dark UI** — Custom-styled interface built with Streamlit + custom CSS

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / App Framework | Streamlit |
| Machine Learning Model | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer |
| Language | Python |
| Styling | Custom CSS (Streamlit markdown) |

---

## 📂 Project Structure

```
fake_news_detector/
│
├── app.py                 # Main Streamlit application
├── classifier.py          # Loads model & handles predictions
├── explainer.py            # Generates explanations, sentiment & credibility scores
├── model/
│   ├── model.pkl           # Trained Logistic Regression model
│   └── tfidf.pkl            # TF-IDF vectorizer
├── requirements.txt        # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/veritai-fake-news-detector.git
cd veritai-fake-news-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (if `model.pkl` / `tfidf.pkl` are not included)
Run your training script (e.g. `train_model.py`) to generate:
```
model/model.pkl
model/tfidf.pkl
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 How It Works

1. User pastes or uploads a news article
2. The text is vectorized using **TF-IDF**
3. A **Logistic Regression** classifier predicts REAL or FAKE with a confidence score
4. Additional analysis (sentiment, clickbait detection, credibility scoring) is layered on top of the ML prediction to provide explainability
5. Results are displayed with visual breakdowns and can be exported as a report

---

## 📌 Future Improvements

- Support for multiple languages
- Deep learning model (BERT/Transformer-based) for higher accuracy
- Source credibility database integration
- Browser extension for real-time news checking

---

## 👤 Author

Developed as an academic / end-semester project.

---

## 📄 License

This project is for educational purposes.
