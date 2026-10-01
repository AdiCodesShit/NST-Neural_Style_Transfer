#!/usr/bin/env python
"""
Production startup script for NST Flask app on Render.
This script handles model loading and server startup with proper error handling.
"""
import os
import sys
import logging

# Set up logging FIRST before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("NST Production Startup Script")
logger.info("=" * 80)

# Change to NST_Code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Working directory: {os.getcwd()}")

# Import Flask app
try:
    from app import app, encoder, decoder, device
    logger.info("✓ Flask app imported successfully")
    logger.info(f"✓ Models loaded (device: {device})")
except Exception as e:
    logger.error(f"✗ Failed to import app: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

if __name__ == '__main__':
    # For Gunicorn, this script is imported, not executed directly
    # But if run directly, start development server
    logger.info("Running app...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
