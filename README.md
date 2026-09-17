# Smart Resume Analyzer

🚀 **Live demo:** https://smart-resume-analyzer-1-gpy2.onrender.com

A Flask web application that reads an uploaded PDF resume, uses a trained artificial neural network (ANN) to predict a suitable job role, and reports matching and missing skills.

## Features

- Upload text-based PDF resumes
- Predict the most suitable job role with the trained ANN
- Show model confidence, skills match score, and overall resume score
- List matched and missing skills for the predicted role

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, select **New > Web Service** and connect the repository.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to `gunicorn app:app`.
5. After deployment, copy the generated `https://<service>.onrender.com` address and replace the Live demo placeholder at the top of this README.

The saved model and preprocessing artifacts (`resume_model.keras`, `vectorizer.pkl`, and `encoder.pkl`) are included so the deployed app can predict without retraining.

## Project files

- `app.py` — Flask application
- `train_model.py` — model training script
- `test_model.py` — command-line prediction test
- `dataset.csv` — training data
