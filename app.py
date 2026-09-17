import os
import re
import pickle
import numpy as np

import pdfplumber
import tensorflow as tf

from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)
app.secret_key = "resume_analyzer_secret"

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# ============================================================
# SKILL DICTIONARY
# ============================================================

ROLE_SKILLS = {

    "Data Scientist": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "keras",
        "scikit-learn",
        "pandas",
        "numpy",
        "statistics",
        "nlp",
        "data analysis",
        "matplotlib",
        "seaborn",
        "jupyter"
    ],

    "Web Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "nodejs",
        "responsive design",
        "bootstrap",
        "jquery",
        "rest api",
        "typescript",
        "git",
        "vue",
        "angular"
    ],

    "Backend Developer": [
        "python",
        "django",
        "flask",
        "rest api",
        "sql",
        "postgresql",
        "mysql",
        "mongodb",
        "docker",
        "redis",
        "microservices",
        "java",
        "spring",
        "nodejs",
        "authentication"
    ],

    "AI Engineer": [
        "python",
        "tensorflow",
        "pytorch",
        "deep learning",
        "nlp",
        "computer vision",
        "transformers",
        "bert",
        "reinforcement learning",
        "model deployment",
        "mlops",
        "gpu",
        "cuda"
    ],

    "Data Analyst": [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "data analysis",
        "reporting",
        "dashboards",
        "python",
        "statistics",
        "visualization",
        "kpi",
        "business intelligence"
    ]
}


# ============================================================
# LOAD ML ARTIFACTS
# ============================================================

print("⏳ Loading model artifacts...")

MODEL_PATH = os.path.join(BASE_DIR, "resume_model.keras")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")

try:

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    print("✅ Model loaded and ready.")

except Exception as e:

    print("❌ Failed to load model artifacts.")
    print(f"Error: {e}")

    raise


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def allowed_file(filename):
    """
    Check whether uploaded file is a PDF.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using pdfplumber.
    """

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text


def clean_text(text):
    """
    Convert text to lowercase and remove special characters.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def match_skills(resume_text, role):
    """
    Find matched and missing skills for the predicted role.
    """

    required = ROLE_SKILLS.get(role, [])

    matched = [
        skill
        for skill in required
        if skill in resume_text
    ]

    missing = [
        skill
        for skill in required
        if skill not in resume_text
    ]

    if required:

        score = (
            len(matched) /
            len(required)
        ) * 100

    else:

        score = 0

    return (
        matched,
        missing,
        round(score, 2)
    )


def calculate_score(skill_score, confidence):
    """
    Final resume score:

    70% = skill match
    30% = model confidence
    """

    final_score = (
        0.7 * skill_score
        +
        0.3 * (confidence * 100)
    )

    return round(final_score, 2)


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """
    Display the resume upload page.
    """

    return render_template("index.html")


# ============================================================
# ANALYZE ROUTE
# ============================================================

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """
    Handle resume upload and prediction.

    GET:
        Redirects back to homepage.

    POST:
        Processes uploaded PDF and displays results.
    """

    # --------------------------------------------------------
    # If someone directly opens /analyze in browser
    # --------------------------------------------------------

    if request.method == "GET":

        return redirect(url_for("index"))


    # --------------------------------------------------------
    # Check whether file exists
    # --------------------------------------------------------

    if "resume" not in request.files:

        flash("No file part in the request.")

        return redirect(url_for("index"))


    file = request.files["resume"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        flash("Please select a PDF file before uploading.")

        return redirect(url_for("index"))


    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        flash("Only PDF files are supported.")

        return redirect(url_for("index"))


    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(file.filename)

    if not filename:

        flash("Invalid filename.")

        return redirect(url_for("index"))


    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    try:

        file.save(save_path)

        print(f"📄 Resume uploaded: {filename}")


        # ====================================================
        # 1. EXTRACT PDF TEXT
        # ====================================================

        print("🔍 Extracting text from PDF...")

        raw_text = extract_text_from_pdf(save_path)

        if not raw_text.strip():

            flash(
                "Could not extract text from the PDF. "
                "Please use a text-based PDF."
            )

            return redirect(url_for("index"))


        print(
            f"✅ Extracted {len(raw_text)} characters."
        )


        # ====================================================
        # 2. CLEAN TEXT
        # ====================================================

        clean = clean_text(raw_text)


        if not clean:

            flash(
                "No readable text was found in the resume."
            )

            return redirect(url_for("index"))


        # ====================================================
        # 3. TF-IDF FEATURE EXTRACTION
        # ====================================================

        print("🧮 Creating TF-IDF features...")

        features = vectorizer.transform(
            [clean]
        ).toarray()


        print(
            f"✅ Feature shape: {features.shape}"
        )


        # ====================================================
        # 4. MODEL PREDICTION
        # ====================================================

        print("🤖 Predicting resume role...")

        predictions = model.predict(
            features,
            verbose=0
        )[0]


        best_idx = int(
            np.argmax(predictions)
        )


        predicted_role = encoder.classes_[best_idx]


        confidence = float(
            predictions[best_idx]
        )


        print(
            f"✅ Predicted role: {predicted_role}"
        )

        print(
            f"✅ Confidence: {confidence:.4f}"
        )


        # ====================================================
        # 5. MATCH SKILLS
        # ====================================================

        matched, missing, skill_score = match_skills(
            clean,
            predicted_role
        )


        print(
            f"✅ Matched skills: {len(matched)}"
        )

        print(
            f"❌ Missing skills: {len(missing)}"
        )


        # ====================================================
        # 6. CALCULATE FINAL SCORE
        # ====================================================

        final_score = calculate_score(
            skill_score,
            confidence
        )


        print(
            f"📊 Final resume score: {final_score}"
        )


        # ====================================================
        # 7. DISPLAY RESULTS
        # ====================================================

        return render_template(
            "index.html",

            predicted_role=predicted_role,

            confidence=round(
                confidence * 100,
                2
            ),

            resume_score=final_score,

            matched_skills=matched,

            missing_skills=missing,

            skill_score=skill_score,

            filename=filename
        )


    except Exception as e:

        # ----------------------------------------------------
        # Print complete error in Render logs
        # ----------------------------------------------------

        print("❌ ERROR WHILE PROCESSING RESUME")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        flash(
            "An error occurred while processing the resume: "
            + str(e)
        )

        return redirect(url_for("index"))


    finally:

        # ----------------------------------------------------
        # Delete uploaded file after processing
        # ----------------------------------------------------

        if os.path.exists(save_path):

            try:

                os.remove(save_path)

                print(
                    f"🗑️ Removed temporary file: {filename}"
                )

            except Exception as e:

                print(
                    f"⚠️ Could not remove temporary file: {e}"
                )


# ============================================================
# FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "File is too large. Please upload a PDF smaller than 10 MB."
    )

    return redirect(url_for("index"))


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    print("❌ Internal Server Error:")
    print(error)

    flash(
        "An internal server error occurred. "
        "Please try again."
    )

    return redirect(url_for("index"))


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
