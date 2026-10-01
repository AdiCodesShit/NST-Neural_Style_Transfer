import os
import sys
import logging
import traceback

import numpy as np
import torch
from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from PIL import Image

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "neural-style-transfer-secret-key"
)

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

Bootstrap(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Using device: {device}")

encoder = None
decoder = None


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def transform_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image)
    tensor = tensor.permute(2, 0, 1)
    return tensor


def tensor_to_pil(image):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def load_models():
    global encoder
    global decoder

    try:
        logger.info("Loading models...")

        vgg_path = os.path.join(
            BASE_DIR,
            "vgg_normalised.pth"
        )

        if not os.path.exists(vgg_path):
            raise FileNotFoundError(
                f"VGG model not found: {vgg_path}"
            )

        encoder = VGGEncoder(
            vgg_path,
            device=device
        ).to(device)

        decoder = Decoder().to(device)

        checkpoint_dir = os.path.join(
            BASE_DIR,
            "experiment",
            "final_exp"
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            "decoder_final.pth"
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Decoder checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif "decoder" in checkpoint:
                state = checkpoint["decoder"]
            else:
                state = checkpoint
        else:
            state = checkpoint

        if isinstance(state, dict):
            state = {
                key.replace("module.", "", 1): value
                for key, value in state.items()
            }

        decoder.load_state_dict(state)

        encoder.eval()
        decoder.eval()

        logger.info("Models loaded successfully.")

    except Exception as e:
        logger.error("Failed to load models.")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        raise


try:
    load_models()
except Exception:
    logger.error("Application startup failed.")
    sys.exit(1)


class UploadForm(FlaskForm):
    content_image = FileField(
        "Content Image",
        validators=[DataRequired()]
    )

    style_image = FileField(
        "Style Image",
        validators=[DataRequired()]
    )

    submit = SubmitField("Apply Style")


def style_transfer(content_image, style_image, alpha=1.0):
    try:
        if encoder is None or decoder is None:
            raise RuntimeError("Models have not been loaded.")

        content_tensor = transform_image(
            content_image
        ).unsqueeze(0).to(device)

        style_tensor = transform_image(
            style_image
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            content_feats = encoder(
                content_tensor,
                is_test=True
            )

            style_feats = encoder(
                style_tensor,
                is_test=True
            )

            stylized_feats = adaptive_instance_normalization(
                content_feats,
                style_feats
            )

            stylized_feats = (
                alpha * stylized_feats
                + (1 - alpha) * content_feats
            )

            stylized_image = decoder(
                stylized_feats
            )

        return stylized_image

    except Exception as e:
        logger.error(
            f"Style transfer failed: {str(e)}"
        )
        logger.error(
            traceback.format_exc()
        )
        raise


def save_image(image, path):
    try:
        pil_image = tensor_to_pil(image)

        pil_image.save(
            path,
            quality=85
        )

        logger.info(
            f"Saved image to: {path}"
        )

    except Exception as e:
        logger.error(
            f"Failed to save image: {str(e)}"
        )
        raise


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    if form.validate_on_submit():
        content_file = form.content_image.data
        style_file = form.style_image.data

        if not content_file or not style_file:
            flash(
                "Please upload both images.",
                "danger"
            )
            return redirect(url_for("index"))

        if not allowed_file(content_file.filename):
            flash(
                "Invalid content image format.",
                "danger"
            )
            return redirect(url_for("index"))

        if not allowed_file(style_file.filename):
            flash(
                "Invalid style image format.",
                "danger"
            )
            return redirect(url_for("index"))

        try:
            content_filename = secure_filename(
                content_file.filename
            )

            style_filename = secure_filename(
                style_file.filename
            )

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "content_" + content_filename
            )

            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "style_" + style_filename
            )

            content_file.save(content_path)
            style_file.save(style_path)

            content_image = Image.open(
                content_path
            ).convert("RGB")

            style_image = Image.open(
                style_path
            ).convert("RGB")

            result = style_transfer(
                content_image,
                style_image,
                alpha=1.0
            )

            output_filename = (
                "result_"
                + content_filename.rsplit(".", 1)[0]
                + "_"
                + style_filename.rsplit(".", 1)[0]
                + ".jpg"
            )

            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                output_filename
            )

            save_image(
                result,
                output_path
            )

            return render_template(
                "index.html",
                form=form,
                result_image=url_for(
                    "static",
                    filename="uploads/" + output_filename
                ),
                content_image=url_for(
                    "static",
                    filename="uploads/content_" + content_filename
                ),
                style_image=url_for(
                    "static",
                    filename="uploads/style_" + style_filename
                )
            )

        except Exception as e:
            logger.error(
                f"Error processing images: {str(e)}"
            )

            logger.error(
                traceback.format_exc()
            )

            flash(
                "An error occurred while processing the images.",
                "danger"
            )

            return redirect(url_for("index"))

    return render_template(
        "index.html",
        form=form
    )


@app.route("/examples/<path:filename>")
def examples(filename):
    return send_from_directory(
        os.path.join(BASE_DIR, "examples"),
        filename
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": (
            encoder is not None
            and decoder is not None
        )
    }


@app.errorhandler(413)
def request_entity_too_large(error):
    flash(
        "File is too large. Maximum size is 16 MB.",
        "danger"
    )

    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_server_error(error):
    logger.error(
        f"Internal server error: {error}"
    )

    return render_template(
        "index.html",
        form=UploadForm(),
        error="Internal server error."
    ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get("PORT", 5000)
        ),
        debug=False
    )
