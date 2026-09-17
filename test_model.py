# =============================================================
# test_model.py
# Smart Resume Analyzer - Direct Run Version
# =============================================================

import pickle
import numpy as np
import pdfplumber
import re
import os
import tensorflow as tf

# (Optional) Hide TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


# ── Extract text from PDF ─────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text


# ── Clean text ────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Skill matching ────────────────────────────────────────────
def match_skills(resume_text, role):
    required = ROLE_SKILLS.get(role, [])
    matched = [s for s in required if s in resume_text]
    missing = [s for s in required if s not in resume_text]
    skill_score = (len(matched) / len(required) * 100) if required else 0
    return matched, missing, skill_score


# ── Score calculation ─────────────────────────────────────────
def calculate_score(skill_score, confidence):
    return round(0.7 * skill_score + 0.3 * (confidence * 100), 2)


# ── Main function ─────────────────────────────────────────────
def analyze_resume(pdf_path):
    # Load model and files
    model = tf.keras.models.load_model("resume_model.keras")
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))

    # Extract and clean text
    raw_text = extract_text_from_pdf(pdf_path)
    clean = clean_text(raw_text)

    # Transform text
    features = vectorizer.transform([clean]).toarray()

    # Predict
    predictions = model.predict(features)[0]
    best_idx = np.argmax(predictions)
    predicted_role = encoder.classes_[best_idx]
    confidence = predictions[best_idx]

    # Skill analysis
    matched, missing, skill_score = match_skills(clean, predicted_role)

    # Final score
    final_score = calculate_score(skill_score, confidence)

    return {
        "predicted_role": predicted_role,
        "confidence": round(float(confidence) * 100, 2),
        "resume_score": final_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_score": round(skill_score, 2)
    }


# ── Run مباشرة (Direct execution) ─────────────────────────────
if __name__ == "__main__":

    # 🔥 Your resume path added here
    pdf_path = r"C:\Users\ROHIT\PycharmProjects\ANN project final\Resume.pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
    else:
        print(f"\n📄 Analyzing: {pdf_path}\n")

        result = analyze_resume(pdf_path)

        print(f"🎯 Predicted Role  : {result['predicted_role']}")
        print(f"📈 Model Confidence: {result['confidence']}%")
        print(f"⭐ Resume Score    : {result['resume_score']} / 100")

        print("\n✅ Matched Skills:")
        print(", ".join(result['matched_skills']) if result['matched_skills'] else "None")

        print("\n❌ Missing Skills:")
        print(", ".join(result['missing_skills']) if result['missing_skills'] else "None")