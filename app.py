import os
import re
import pickle
import numpy as np

import pdfplumber
import tensorflow as tf
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

# ── App config ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "resume_analyzer_secret"   # needed for flash messages

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Skill dictionary ──────────────────────────────────────────
ROLE_SKILLS = {
    "Data Scientist": [
        "python", "machine learning", "deep learning", "tensorflow", "keras",
        "scikit-learn", "pandas", "numpy", "statistics", "nlp",
        "data analysis", "matplotlib", "seaborn", "jupyter"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "nodejs",
        "responsive design", "bootstrap", "jquery", "rest api",
        "typescript", "git", "vue", "angular"
    ],
    "Backend Developer": [
        "python", "django", "flask", "rest api", "sql",
        "postgresql", "mysql", "mongodb", "docker", "redis",
        "microservices", "java", "spring", "nodejs", "authentication"
    ],
    "AI Engineer": [
        "python", "tensorflow", "pytorch", "deep learning", "nlp",
        "computer vision", "transformers", "bert", "reinforcement learning",
        "model deployment", "mlops", "gpu", "cuda"
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "tableau", "data analysis",
        "reporting", "dashboards", "python", "statistics",
        "visualization", "kpi", "business intelligence"
    ]
}

# ── Load saved ML artifacts once at startup ───────────────────
print("⏳ Loading model artifacts...")
model      = tf.keras.models.load_model("resume_model.keras")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder    = pickle.load(open("encoder.pkl",    "rb"))
print("✅ Model loaded and ready.")


# ── Utility functions ─────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


def clean_text(text):
    """Lowercase and remove non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_skills(resume_text, role):
    """Return matched skills, missing skills, and skill score (0-100)."""
    required = ROLE_SKILLS.get(role, [])
    matched  = [s for s in required if s in resume_text]
    missing  = [s for s in required if s not in resume_text]
    score    = (len(matched) / len(required) * 100) if required else 0
    return matched, missing, round(score, 2)


def calculate_score(skill_score, confidence):
    """Final score = 70% skill match + 30% model confidence."""
    return round(0.7 * skill_score + 0.3 * (confidence * 100), 2)


# ── Routes ────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handle PDF upload and return analysis results."""

    # Validate file presence
    if "resume" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["resume"]
    if file.filename == "":
        flash("Please select a PDF file before uploading.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are supported.")
        return redirect(url_for("index"))

    # Save uploaded file
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        # 1. Extract text
        raw_text = extract_text_from_pdf(save_path)
        if not raw_text.strip():
            flash("Could not extract text from the PDF. Please use a text-based PDF.")
            return redirect(url_for("index"))

        # 2. Clean text
        clean = clean_text(raw_text)

        # 3. TF-IDF → feature vector
        features = vectorizer.transform([clean]).toarray()

        # 4. Predict role
        predictions    = model.predict(features)[0]
        best_idx       = int(np.argmax(predictions))
        predicted_role = encoder.classes_[best_idx]
        confidence     = float(predictions[best_idx])

        # 5. Skill matching
        matched, missing, skill_score = match_skills(clean, predicted_role)

        # 6. Final resume score
        final_score = calculate_score(skill_score, confidence)

        # 7. Render results
        return render_template(
            "index.html",
            predicted_role  = predicted_role,
            confidence      = round(confidence * 100, 2),
            resume_score    = final_score,
            matched_skills  = matched,
            missing_skills  = missing,
            skill_score     = skill_score,
            filename        = filename
        )

    except Exception as e:
        flash(f"An error occurred while processing the resume: {str(e)}")
        return redirect(url_for("index"))

    finally:
        # Remove uploaded file after processing
        if os.path.exists(save_path):
            os.remove(save_path)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
