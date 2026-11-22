
# AI-Based Cat Image Reconstruction and Generation

## Overview
A deep learning web application that reconstructs existing cat images using an Autoencoder and generates new cat images using DCGAN.

## Models Used
- **Autoencoder:** Reconstructs uploaded cat images.
- **DCGAN:** Generates new cat images from random noise.
- **Model Comparison:** Allows users to view Autoencoder reconstruction and DCGAN generation results.

## Technologies
- Python
- TensorFlow / Keras
- Flask
- CNN
- Autoencoder
- DCGAN
- HTML
- CSS
- JavaScript

## Features
- Select between Autoencoder and DCGAN.
- Upload an image for reconstruction.
- Generate new cat images using DCGAN.
- Compare outputs from both models.
- Web-based interface using Flask.

## Dataset
Oxford-IIIT Pet Dataset.

## Project Structure
image_generation/
- app.py
- models.py
- train_gan.py
- templates/
- static/
- weights/

## How to Run

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Train the DCGAN.
5. Run the Flask application.

## Installation

```bash
pip install -r requirements.txt
```

## Run GAN Training

```bash
python train_gan.py
```

## Run the Web Application

```bash
python app.py
```

Open in your browser:

http://127.0.0.1:5000

## Author
GowsalyaGanesan

