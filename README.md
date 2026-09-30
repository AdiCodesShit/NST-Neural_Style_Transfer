# 🎨 StyleForge AI

### AI-Powered Neural Style Transfer

StyleForge AI is a deep-learning-based web application that performs
**Neural Style Transfer (NST)** by combining the content of one image
with the artistic style of another.

The project uses **custom-trained models** trained on the **Microsoft
COCO Dataset** and the **Painter by Numbers Dataset** obtained from
Kaggle.

Users can upload a content image, select a style reference, adjust the
style strength, and generate a stylized image through a Flask web
interface.

------------------------------------------------------------------------

## ✨ Features

-   🖼️ Upload custom content images
-   🎨 Upload custom style reference images
-   🧠 Custom-trained Neural Style Transfer model
-   🎚️ Adjustable style strength
-   ⚡ AI-powered image stylization
-   👀 Preview generated results
-   📥 Download stylized images
-   🌐 Flask-based web application
-   🎨 Futuristic AI-themed frontend
-   🖼️ Includes sample content and style images
-   🧪 Includes model training and experiment files

------------------------------------------------------------------------

## 🧠 What is Neural Style Transfer?

Neural Style Transfer is a deep learning technique that combines the
**content** of one image with the **artistic style** of another.

The content image provides the structure, objects, and composition,
while the style image provides characteristics such as colors, textures,
patterns, and brush strokes.

``` text
Content Image + Style Reference
              ↓
         StyleForge AI
              ↓
       Stylized Image
```

------------------------------------------------------------------------

## 🧠 Model & Training

StyleForge AI uses a **custom-trained Neural Style Transfer pipeline**
rather than relying on an external image-generation API.

### Microsoft COCO Dataset

The **COCO (Common Objects in Context)** dataset was used as a source of
diverse real-world imagery and content.

### Painter by Numbers Dataset

The **Painter by Numbers** dataset was obtained from **Kaggle** and used
as a source of artistic imagery and styles.

### Training Pipeline

``` text
              TRAINING DATA
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     COCO Dataset      Painter by Numbers
       (Content)             (Style)
          │                   │
          └─────────┬─────────┘
                    ▼
             Model Training
                    │
                    ▼
            Custom NST Model
                    │
                    ▼
              StyleForge AI
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

### Backend

-   Python
-   Flask

### Machine Learning

-   PyTorch
-   Neural Style Transfer
-   Deep Learning
-   Computer Vision
-   VGG-based feature extraction

### Frontend

-   HTML
-   CSS
-   JavaScript

### Datasets

-   Microsoft COCO Dataset
-   Painter by Numbers Dataset
-   Painter by Numbers dataset obtained through Kaggle

------------------------------------------------------------------------

## 📂 Project Structure

``` text
NST_Code/
├── content_data/
├── examples/
├── experiment/
│   └── final_exp/
│       ├── decoder_final.pth
│       ├── options.txt
│       └── sample_iter_*.png
├── static/
│   └── uploads/
├── style_data/
├── templates/
│   ├── index.html
│   └── index.html.backup
├── utils/
│   ├── models.py
│   └── utils.py
├── app.py
├── train.py
└── vgg_normalised.pth
```

> `__pycache__` and other generated/cache files generally should not be
> committed to GitHub. Consider adding them to `.gitignore`.

------------------------------------------------------------------------

## 📁 Important Files & Directories

### `app.py`

The main **Flask application** responsible for running the web interface
and connecting the frontend with the Neural Style Transfer pipeline.

### `train.py`

Contains the training pipeline used to train the Neural Style Transfer
model.

### `utils/models.py`

Contains the model architecture and neural network components used by
the project.

### `utils/utils.py`

Contains utility functions used for image processing and other
operations required by the model.

### `vgg_normalised.pth`

Contains VGG weights used for feature extraction/normalization within
the style-transfer pipeline.

### `experiment/final_exp/`

Contains artifacts from the final training experiment, including the
trained decoder, training configuration, and intermediate generated
samples.

### `content_data/`

Contains example content images.

### `style_data/`

Contains example artistic style images.

### `examples/`

Contains example content images and generated stylized results.

### `static/uploads/`

Contains images used by the web application, including uploaded images
and generated results.

------------------------------------------------------------------------

## 🚀 Getting Started

### 1. Clone the Repository

``` bash
git clone https://github.com/YOUR_USERNAME/StyleForge-AI.git
cd StyleForge-AI
```

### 2. Create a Virtual Environment

#### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

``` bash
python app.py
```

Open the local URL displayed by Flask, typically:

``` text
http://127.0.0.1:5000
```

------------------------------------------------------------------------

## 🎨 Using StyleForge AI

### Step 1 --- Select a Content Image

Upload an image containing the content you want to preserve.

### Step 2 --- Select a Style Image

Upload an artwork or image whose artistic characteristics you want to
apply.

### Step 3 --- Adjust Style Strength

Use the **Style Strength** slider to control how strongly the artistic
style is applied.

### Step 4 --- Transfer the Style

Click **TRANSFER STYLE** to generate the stylized image.

------------------------------------------------------------------------

## 🔬 High-Level Architecture

``` text
                         USER
                          │
                          ▼
                  ┌───────────────┐
                  │  Flask Web UI │
                  └───────┬───────┘
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
          Content Image       Style Image
                 │                 │
                 └────────┬────────┘
                          ▼
                  ┌───────────────┐
                  │  NST Model    │
                  │    PyTorch    │
                  └───────┬───────┘
                          │
                          ▼
                   Stylized Image
                          │
                          ▼
                    Web Interface
```

------------------------------------------------------------------------

## 📊 Training

The training script can be run with:

``` bash
python train.py
```

Training artifacts are stored under:

``` text
experiment/final_exp/
```

The final trained decoder is:

``` text
decoder_final.pth
```

Training samples are stored as:

``` text
sample_iter_20.png
sample_iter_40.png
sample_iter_60.png
sample_iter_80.png
sample_iter_100.png
sample_iter_120.png
sample_iter_140.png
sample_iter_160.png
sample_iter_180.png
sample_iter_200.png
```

These samples provide a visual representation of how the model's output
evolved during training.

> Training a deep learning model can require significant computational
> resources. Running inference with an already-trained model is
> generally much less demanding than training it from scratch.

------------------------------------------------------------------------

## 🖼️ Example Results

The repository contains several example transformations in the
`examples/` directory.

Examples include:

-   Brad Pitt
-   Mona Lisa
-   Starry Night
-   Personal portrait images
-   Picasso-style transformations

Example file pairs include:

``` text
brad_pitt.jpg
stylized_brad_pitt.jpg

mona_lisa.png
stylized_mona_lisa.png

starry_night.jpg
stylized_starry_night.jpg
```

------------------------------------------------------------------------

## 📈 Training Data

  Dataset              Purpose
  -------------------- -----------------------------
  Microsoft COCO       Content / real-world images
  Painter by Numbers   Artistic style images

The **Painter by Numbers** dataset used for this project was obtained
through **Kaggle**.

The exact number of images used for training and the final preprocessing
configuration may depend on the specific training experiment.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   [ ] Improve inference speed
-   [ ] GPU acceleration
-   [ ] Support multiple NST models
-   [ ] Add style presets
-   [ ] Add image history
-   [ ] Add before/after comparison
-   [ ] Improve output resolution
-   [ ] Add batch style transfer
-   [ ] Improve mobile responsiveness
-   [ ] Add more artistic datasets
-   [ ] Improve model quality through longer training
-   [ ] Add additional image-processing controls

------------------------------------------------------------------------

## ⚠️ Notes

-   Neural Style Transfer results depend heavily on the selected content
    and style images.
-   Processing time depends on available hardware.
-   GPU acceleration can significantly improve training and inference
    performance.
-   The included trained model can be used for inference without
    retraining the entire network.
-   Large datasets and model files may not be suitable for standard
    GitHub repositories due to repository size limitations.
-   Make sure to comply with the licenses and usage terms of the
    datasets and artwork used with the project.

------------------------------------------------------------------------

## 🎓 Project Information

**Project Name:** StyleForge AI

**Project Type:** Academic / NST Project

**Domain:** Artificial Intelligence, Deep Learning & Computer Vision

**Technique:** Neural Style Transfer

**Backend:** Flask

**Deep Learning Framework:** PyTorch

**Programming Language:** Python

### Datasets

-   Microsoft COCO Dataset
-   Painter by Numbers Dataset --- obtained through Kaggle

------------------------------------------------------------------------

## 👨‍💻 Author

### Aditya Bhagat

This project was developed as an academic exploration of Artificial
Intelligence, Deep Learning, Computer Vision, Neural Style Transfer,
PyTorch, model training, and web-based AI applications.

------------------------------------------------------------------------

## ⭐ Support

If you found this project interesting, consider giving the repository a
⭐ on GitHub!

------------------------------------------------------------------------

### Built with

**Python • PyTorch • Flask • Deep Learning • Computer Vision**
