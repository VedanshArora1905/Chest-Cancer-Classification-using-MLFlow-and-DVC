from flask import Flask, request, jsonify, render_template
import os
import logging
from flask_cors import CORS, cross_origin
from ChestCancerClassifier.utils.common import decodeImage
from ChestCancerClassifier.pipeline.prediction import PredictionPipeline
from werkzeug.middleware.proxy_fix import ProxyFix
from config import get_config
from logging_config import setup_logging

# Get configuration
config = get_config()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)

# Configure CORS with specific origins
CORS(app, resources={r"/*": {"origins": config.CORS_ORIGINS}})

# Add ProxyFix middleware for proper handling behind reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Setup logging
setup_logging(app)
logger = logging.getLogger(__name__)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

    def predict(self, image_data):
        try:
            decodeImage(image_data, self.filename)
            return self.classifier.predict()
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize the application
clApp = ClientApp()

@app.route("/health", methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train_route():
    try:
        # Use subprocess instead of os.system for better security
        import subprocess
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return jsonify({"error": "Training failed"}), 500
        return jsonify({"message": "Training completed successfully"}), 200
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        image = request.json.get('image')
        if not image:
            return jsonify({"error": "No image data provided"}), 400

        result = clApp.predict(image)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )