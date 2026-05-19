import os
import uuid
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import requests
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)

from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename


# ==============================
# BASE CONFIG
# ==============================
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_finetuned_model.keras"

UPLOAD_DIR = APP_DIR / "static" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

MAX_FILE_SIZE_MB = 16
IMAGE_SIZE = (224, 224)

THRESHOLD = 0.5

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    os.urandom(24)
)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

app.config["MAX_CONTENT_LENGTH"] = (
    MAX_FILE_SIZE_MB * 1024 * 1024
)

CORS(app)


# ==============================
# GPU MEMORY FIX
# ==============================
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# ==============================
# LOAD MODEL
# ==============================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# HELPER FUNCTIONS
# ==============================
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def validate_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:
        return False


def preprocess_image(image_path: Path) -> np.ndarray:

    img = Image.open(image_path).convert("RGB")

    img = img.resize(IMAGE_SIZE)

    arr = np.array(img, dtype=np.float32)

    arr = np.expand_dims(arr, axis=0)

    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return arr


def predict_image(image_path: Path) -> dict:

    img_array = preprocess_image(image_path)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    nsfw_prob = float(prediction[0][0])

    safe_prob = 1.0 - nsfw_prob

    if nsfw_prob >= THRESHOLD:
        label = "NSFW"
        confidence = nsfw_prob

    else:
        label = "SAFE"
        confidence = safe_prob

    return {
        "label": label,

        "confidence": round(
            confidence * 100,
            2
        ),

        "nsfw_probability": round(
            nsfw_prob * 100,
            2
        ),

        "safe_probability": round(
            safe_prob * 100,
            2
        ),
    }


def save_processed_image(
    input_path: Path,
    output_path: Path,
    label: str
):

    img = cv2.imread(str(input_path))

    if img is None:
        raise ValueError(
            "Could not read uploaded image."
        )

    if label == "NSFW":

        processed = cv2.GaussianBlur(
            img,
            (61, 61),
            0
        )

    else:
        processed = img

    cv2.imwrite(
        str(output_path),
        processed
    )


def cleanup_old_uploads(
    max_files: int = 100
):

    files = sorted(
        [
            f for f in UPLOAD_DIR.iterdir()
            if f.is_file()
        ],

        key=lambda x: x.stat().st_mtime,

        reverse=True,
    )

    for old_file in files[max_files:]:

        try:
            old_file.unlink()

        except OSError:
            pass


# ==============================
# MAIN WEBSITE ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "image" not in request.files:

            flash(
                "No image file found."
            )

            return redirect(
                url_for("index")
            )

        file = request.files["image"]

        if file.filename == "":

            flash(
                "Please select an image."
            )

            return redirect(
                url_for("index")
            )

        if not allowed_file(file.filename):

            flash(
                "Invalid file type."
            )

            return redirect(
                url_for("index")
            )

        try:

            original_name = secure_filename(
                file.filename
            )

            extension = original_name.rsplit(
                ".",
                1
            )[1].lower()

            unique_id = uuid.uuid4().hex

            upload_filename = (
                f"{unique_id}.{extension}"
            )

            result_filename = (
                f"{unique_id}_result.{extension}"
            )

            upload_path = (
                UPLOAD_DIR / upload_filename
            )

            result_path = (
                UPLOAD_DIR / result_filename
            )

            file.save(upload_path)

            if not validate_image(upload_path):

                upload_path.unlink(
                    missing_ok=True
                )

                flash(
                    "Uploaded file is not a valid image."
                )

                return redirect(
                    url_for("index")
                )

            prediction = predict_image(
                upload_path
            )

            save_processed_image(
                upload_path,
                result_path,
                prediction["label"]
            )

            cleanup_old_uploads()

            return render_template(
                "result.html",

                original_image=upload_filename,

                result_image=result_filename,

                label=prediction["label"],

                confidence=prediction[
                    "confidence"
                ],

                nsfw_probability=prediction[
                    "nsfw_probability"
                ],

                safe_probability=prediction[
                    "safe_probability"
                ],
            )

        except Exception as e:

            flash(
                f"Error while processing image: {str(e)}"
            )

            return redirect(
                url_for("index")
            )

    return render_template("index.html")


# ==============================
# BROWSER EXTENSION API
# ==============================
@app.route("/api/check", methods=["POST"])
def api_check():

    data = request.get_json()

    image_url = data.get("image_url")

    if not image_url:

        return jsonify({
            "error": "No image URL"
        }), 400

    try:

        response = requests.get(
            image_url,
            timeout=10
        )

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        temp_path = (
            UPLOAD_DIR / "temp.jpg"
        )

        image.save(temp_path)

        prediction = predict_image(
            temp_path
        )

        temp_path.unlink(
            missing_ok=True
        )

        return jsonify(prediction)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ==============================
# HEALTH CHECK
# ==============================
@app.route("/health")
def health():

    return {
        "status": "ok"
    }


# ==============================
# FILE SIZE ERROR
# ==============================
@app.errorhandler(413)
def too_large(_error):

    flash(
        f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB."
    )

    return redirect(
        url_for("index")
    )


# ==============================
# CACHE CONTROL
# ==============================
@app.after_request
def add_header(response):

    response.cache_control.no_store = True

    return response


# ==============================
# START APP
# ==============================
if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )