import os
import sys
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
import logging
import traceback

# Configure logging for Render
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting NST Flask App Initialization")
logger.info("=" * 80)

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
Bootstrap(app)

# Create upload folder
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Upload folder ready: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Failed to create upload folder: {e}")

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Base directory: {BASE_DIR}")

# Global model variables
encoder = None
decoder = None

def load_models():
    global encoder, decoder
    try:
        logger.info("Loading models...")
        
        # Load VGG encoder
        vgg_path = os.path.join(BASE_DIR, 'vgg_normalised.pth')
        logger.info(f"Looking for VGG at: {vgg_path}")
        
        if not os.path.exists(vgg_path):
            logger.error(f"VGG not found at {vgg_path}")
            logger.info(f"Contents of {BASE_DIR}: {os.listdir(BASE_DIR)}")
            sys.exit(1)
        
        logger.info(f"VGG found, loading... (size: {os.path.getsize(vgg_path) / 1e9:.2f} GB)")
        encoder = VGGEncoder(vgg_path, device=device).to(device)
        logger.info("VGG Encoder loaded successfully")
        
        decoder = Decoder().to(device)
        logger.info("Decoder initialized")
        
        # Load decoder checkpoint
        checkpoint_dir = os.path.join(BASE_DIR, 'experiment', 'final_exp')
        checkpoint_path = os.path.join(checkpoint_dir, 'decoder_final.pth')
        
        logger.info(f"Looking for checkpoint at: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            # Try alternatives
            alt_paths = [
                os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_1.pth'),
                os.path.join(BASE_DIR, 'experiment', 'decoder_final.pth'),
            ]
            checkpoint_path = None
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    logger.info(f"Found checkpoint at: {checkpoint_path}")
                    break
            
            if not checkpoint_path:
                logger.error("No decoder checkpoint found!")
                logger.info(f"Contents of experiment dir: {os.listdir(os.path.join(BASE_DIR, 'experiment')) if os.path.exists(os.path.join(BASE_DIR, 'experiment')) else 'N/A'}")
                return False
        
        logger.info(f"Loading checkpoint... (size: {os.path.getsize(checkpoint_path) / 1e6:.2f} MB)")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model' in checkpoint):
            state = checkpoint.get('state_dict', checkpoint.get('model'))
        else:
            state = checkpoint
        
        decoder.load_state_dict(state)
        encoder.eval()
        decoder.eval()
        
        logger.info("✓ Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())
        return False

# Load models on startup
if not load_models():
    logger.error("Failed to load models. App cannot start.")
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    try:
        logger.info("Starting style transfer...")

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        logger.info("Converting content image to tensor...")
        content_tensor = transform(content_image).unsqueeze(0).to(device)

        logger.info("Converting style image to tensor...")
        style_tensor = transform(style_image).unsqueeze(0).to(device)

        logger.info(f"Content shape: {content_tensor.shape}")
        logger.info(f"Style shape: {style_tensor.shape}")

        with torch.inference_mode():
            logger.info("Encoding content...")
            content_feats = encoder(content_tensor, is_test=True)
            logger.info("Content encoded")

            logger.info("Encoding style...")
            style_feats = encoder(style_tensor, is_test=True)
            logger.info("Style encoded")

            logger.info("Applying AdaIN...")
            stylized_feats = adaptive_instance_normalization(
                content_feats,
                style_feats
            )

            stylized_feats = (
                alpha * stylized_feats +
                (1 - alpha) * content_feats
            )

            logger.info("Decoding...")
            stylized_image = decoder(stylized_feats)

        logger.info("Style transfer completed")

        return stylized_image

    except Exception as e:
        logger.error(f"Style transfer error: {e}")
        logger.error(traceback.format_exc())
        raise
def save_image(image, path):
    """Save tensor to image file"""
    try:
        image = image.cpu().clone().squeeze(0).clamp(0, 1)
        pil_image = transforms.ToPILImage()(image)
        pil_image.save(path, quality=85)
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    logger.info(f"Route / accessed, method: {request.method}")

    if form.validate_on_submit():
        try:
            logger.info("Form submitted")
            
            # Handle content image
            if form.content.data and form.content.data.filename:
                logger.info(f"Content file received: {form.content.data.filename}")
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(form.content.data.filename)
                    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                    form.content.data.save(content_path)
                    logger.info(f"Content saved: {content_path}")
                    form.content_path.data = content_filename
                else:
                    error = 'Invalid content image format'
            else:
                content_filename = form.content_path.data
                logger.info(f"Using existing content: {content_filename}")

            # Handle style image
            if form.style.data and form.style.data.filename:
                logger.info(f"Style file received: {form.style.data.filename}")
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                    form.style.data.save(style_path)
                    logger.info(f"Style saved: {style_path}")
                    form.style_path.data = style_filename
                else:
                    error = 'Invalid style image format'
            else:
                style_filename = form.style_path.data
                logger.info(f"Using existing style: {style_filename}")

            if not error and content_filename and style_filename:
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                
                if not os.path.exists(content_path):
                    error = 'Content image not found'
                elif not os.path.exists(style_path):
                    error = 'Style image not found'
                else:
                    try:
                        logger.info("Opening images...")
                        content_image = Image.open(content_path).convert('RGB')
                        style_image = Image.open(style_path).convert('RGB')
                        
                        alpha = float(form.alpha.data) if form.alpha.data else 1.0
                        if not (0 <= alpha <= 1):
                            alpha = 1.0
                        
                        logger.info(f"Processing with alpha={alpha}")
                        stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)
                        
                        result_filename = 'stylized_' + content_filename
                        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                        save_image(stylized_image, result_path)
                        result_image = result_filename
                        logger.info(f"✓ Success: {result_filename}")
                        
                    except Exception as e:
                        error = f'Processing error: {str(e)}'
                        logger.error(error)
            elif not error:
                if not content_filename:
                    error = 'Please upload content image'
                elif not style_filename:
                    error = 'Please upload style image'
                    
        except Exception as e:
            error = f'Unexpected error: {str(e)}'
            logger.error(error)
            logger.error(traceback.format_exc())
    else:
        if request.method == 'POST':
            logger.warning("Form validation failed")
            error = 'Form validation failed'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'device': str(device)}), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File too large. Maximum file size is 16MB.', 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal server error: {error}')
    logger.error(traceback.format_exc())
    return 'An internal error occurred. Please try again.', 500

if __name__ == '__main__':
    logger.info("Running Flask app locally (not for production)")
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
