# ADNI Cognitive Status Classifier

A machine learning tool to classify cognitive status (Cognitively Normal, MCI, Alzheimer's) using clinical and cognitive data.

Built with:
- XGBoost
- SHAP explainability
- Streamlit dashboard
- Python

## 🔧 Features
- Predicts CN/MCI/AD with confidence
- Global and individual SHAP explanations
- Clinician-friendly interface

## 🚀 Try It Locally

```bash
git clone https://github.com/your-username/ADNI-Cognitive-Classifier.git
cd ADNI-Cognitive-Classifier
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
ADNI-Cognitive-Classifier/
├── app.py                  # Streamlit dashboard
├── M1/                     # Moved: interpretability module
│   └── interpretability/
│       ├── __init__.py
│       └── shap_engine.py
├── tests/
├── setup.py
├── requirements.txt
└── README.md

## Note
This tool is for Research Use Only (RUO). Not for clinical diagnosis.