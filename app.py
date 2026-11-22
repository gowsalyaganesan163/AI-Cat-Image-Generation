
import os
import uuid

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

from models import (
    build_autoencoder,
    build_generator
)


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = "image-generation-secret-key"

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
WEIGHTS_FOLDER = "weights"

IMG_SIZE = 128
LATENT_DIM = 100
AUTOENCODER_EPOCHS = 30

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)


# --------------------------------------------------
# Image processing functions
# --------------------------------------------------

def load_image(image_path):
    """Load and preprocess an image for the Autoencoder."""

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image, dtype=np.float32) / 255.0

    return np.expand_dims(image_array, axis=0)


def save_autoencoder_output(image_path, output_path):
    """Reconstruct an uploaded image using the Autoencoder."""

    image_array = load_image(image_path)

    autoencoder = build_autoencoder()

    autoencoder.fit(
        image_array,
        image_array,
        epochs=AUTOENCODER_EPOCHS,
        batch_size=1,
        verbose=0
    )

    reconstructed = autoencoder.predict(
        image_array,
        verbose=0
    )

    reconstructed = np.clip(
        reconstructed[0] * 255,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(reconstructed).save(output_path)


# --------------------------------------------------
# DCGAN image generation
# --------------------------------------------------

def save_gan_output(output_path):
    """Generate a new image using the trained DCGAN."""

    generator_weights = os.path.join(
        WEIGHTS_FOLDER,
        "generator.weights.h5"
    )

    if not os.path.exists(generator_weights):
        raise FileNotFoundError(
            "Trained GAN weights were not found. "
            "Please train the GAN first using train_gan.py."
        )

    generator = build_generator(LATENT_DIM)

    generator.load_weights(generator_weights)

    noise = tf.random.normal(
        [1, LATENT_DIM]
    )

    generated_image = generator(
        noise,
        training=False
    ).numpy()[0]

    # Convert generator output from [-1, 1] to [0, 255]

    generated_image = (
        (generated_image + 1.0) * 127.5
    )

    generated_image = np.clip(
        generated_image,
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(generated_image).save(output_path)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Image generation route
# --------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate():

    selected_model = request.form.get("model")

    # Create unique filenames

    unique_id = uuid.uuid4().hex

    try:

        # ------------------------------------------
        # AUTOENCODER
        # ------------------------------------------

        if selected_model == "autoencoder":

            uploaded_file = request.files.get("image")

            if not uploaded_file or not uploaded_file.filename:
                flash("Please upload a cat image.")
                return redirect(url_for("home"))

            filename = secure_filename(uploaded_file.filename)

            original_name = f"{unique_id}_original.png"
            output_name = f"{unique_id}_autoencoder.png"

            original_path = os.path.join(
                UPLOAD_FOLDER,
                original_name
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            # Save original image as PNG

            image = Image.open(uploaded_file).convert("RGB")
            image.save(original_path)

            save_autoencoder_output(
                original_path,
                output_path
            )

            return render_template(
                "result.html",
                model="Autoencoder",
                original=url_for(
                    "static",
                    filename=f"uploads/{original_name}"
                ),
                generated=url_for(
                    "static",
                    filename=f"outputs/{output_name}"
                )
            )

        # ------------------------------------------
        # DCGAN
        # ------------------------------------------

        elif selected_model == "gan":

            output_name = f"{unique_id}_gan.png"

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            save_gan_output(output_path)

            return render_template(
                "result.html",
                model="DCGAN",
                generated=url_for(
                    "static",
                    filename=f"outputs/{output_name}"
                )
            )

        # ------------------------------------------
        # BOTH MODELS - COMPARISON
        # ------------------------------------------

        elif selected_model == "compare":

            uploaded_file = request.files.get("image")

            if not uploaded_file or not uploaded_file.filename:
                flash("Please upload a cat image for comparison.")
                return redirect(url_for("home"))

            original_name = f"{unique_id}_original.png"
            autoencoder_name = f"{unique_id}_autoencoder.png"
            gan_name = f"{unique_id}_gan.png"

            original_path = os.path.join(
                UPLOAD_FOLDER,
                original_name
            )

            autoencoder_path = os.path.join(
                OUTPUT_FOLDER,
                autoencoder_name
            )

            gan_path = os.path.join(
                OUTPUT_FOLDER,
                gan_name
            )

            # Save original image

            image = Image.open(uploaded_file).convert("RGB")
            image.save(original_path)

            # Autoencoder reconstruction

            save_autoencoder_output(
                original_path,
                autoencoder_path
            )

            # DCGAN generation

            save_gan_output(gan_path)

            return render_template(
                "result.html",
                model="Comparison",
                original=url_for(
                    "static",
                    filename=f"uploads/{original_name}"
                ),
                autoencoder_output=url_for(
                    "static",
                    filename=f"outputs/{autoencoder_name}"
                ),
                gan_output=url_for(
                    "static",
                    filename=f"outputs/{gan_name}"
                )
            )

        else:

            flash("Please select a valid model.")
            return redirect(url_for("home"))

    except FileNotFoundError as error:

        flash(str(error))
        return redirect(url_for("home"))

    except Exception as error:

        app.logger.exception("Image generation failed")
        flash(f"An error occurred: {error}")
        return redirect(url_for("home"))


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
