import os
import sys
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
import logging

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
Bootstrap(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Load VGG encoder with correct path
    vgg_path = os.path.join(BASE_DIR, 'vgg_normalised.pth')
    if not os.path.exists(vgg_path):
        logger.error(f"VGG model not found at {vgg_path}")
        sys.exit(1)
    
    encoder = VGGEncoder(vgg_path, device=device).to(device)
    decoder = Decoder().to(device)
    
    # Load decoder checkpoint
    checkpoint_dir = os.path.join(BASE_DIR, 'experiment', 'final_exp')
    checkpoint_path = os.path.join(checkpoint_dir, 'decoder_final.pth')
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}, trying alternative paths...")
        # Try to find any decoder checkpoint
        alt_paths = [
            os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_1.pth'),
            os.path.join(BASE_DIR, 'experiment', 'decoder_final.pth'),
        ]
        checkpoint_path = None
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                logger.info(f"Found checkpoint at {checkpoint_path}")
                break
        
        if not checkpoint_path:
            logger.error("No decoder checkpoint found!")
            sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model' in checkpoint):
        state = checkpoint.get('state_dict', checkpoint.get('model'))
    else:
        state = checkpoint
    
    decoder.load_state_dict(state)
    encoder.eval()
    decoder.eval()
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Fixed size to reduce memory
            transforms.ToTensor()
        ])
        
        content_image = transform(content_image).unsqueeze(0).to(device)
        style_image = transform(style_image).unsqueeze(0).to(device)
        
        # Clear GPU cache before processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            content_feats = encoder(content_image, is_test=True)
            style_feats = encoder(style_image, is_test=True)
            stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
            stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
            stylized_image = decoder(stylized_feats)
        
        # Clear GPU cache after processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return stylized_image
    except RuntimeError as e:
        logger.error(f"CUDA/Memory error during style transfer: {str(e)}")
        raise Exception(f"Memory error: {str(e)}")
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        raise

def save_image(image, path):
    try:
        image = image.cpu().clone().squeeze(0).clamp(0, 1)
        transforms.ToPILImage()(image).save(path, quality=85)
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(form.content.data.filename)
                    form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                    form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                    form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            if content_filename and style_filename:
                content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
                
                if not os.path.exists(content_path) or not os.path.exists(style_path):
                    error = 'Image files not found. Please upload again.'
                else:
                    try:
                        content_image = Image.open(content_path).convert('RGB')
                        style_image = Image.open(style_path).convert('RGB')
                        
                        # Validate alpha
                        alpha = float(form.alpha.data)
                        if alpha < 0 or alpha > 1:
                            alpha = 1.0
                        
                        logger.info(f"Processing images with alpha={alpha}")
                        stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)
                        result_filename = 'stylized_' + content_filename
                        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                        save_image(stylized_image, result_path)
                        result_image = result_filename
                        logger.info(f"Style transfer completed: {result_filename}")
                    except Exception as e:
                        error = f'Error processing images: {str(e)}'
                        logger.error(error)
            else:
                if not content_filename:
                    error = 'Please upload content image'
                if not style_filename:
                    error = 'Please upload style image'
        except Exception as e:
            error = f'An error occurred: {str(e)}'
            logger.error(error)
    else:
        if request.method == 'POST':
            error = 'Form validation failed. Please check your inputs.'

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

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File too large. Maximum file size is 16MB.', 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal server error: {error}')
    return 'An internal error occurred. Please try again.', 500

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
