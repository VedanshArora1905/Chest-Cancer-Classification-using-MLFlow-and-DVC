import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import os
from ChestCancerClassifier import logger


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Class mapping based on ImageDataGenerator alphabetical ordering
        # {'adenocarcinoma': 0, 'normal': 1}
        self.class_mapping = {
            0: 'Adenocarcinoma Cancer',
            1: 'NORMAL'
        }

    def predict(self):
        """
        Predicts whether the image shows cancer or is normal.
        
        Returns:
            list: [{"image": prediction_label}]
        """
        try:
            # Load model
            model_path = os.path.join("artifacts", "training", "model.h5")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")

            # Load and preprocess image
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"Image not found at {self.filename}")
            
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            
            # Normalize image to match training preprocessing (rescale=1./255)
            test_image = test_image / 255.0
            
            # Add batch dimension
            test_image = np.expand_dims(test_image, axis=0)
            
            # Get prediction probabilities
            predictions = model.predict(test_image, verbose=0)
            
            # Get predicted class index
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get probabilities for both classes
            cancer_prob = float(predictions[0][0])
            normal_prob = float(predictions[0][1])
            
            # Map class index to label
            prediction = self.class_mapping[predicted_class_idx]
            
            logger.info(f"Prediction: {prediction} (class {predicted_class_idx}, confidence: {confidence:.4f})")
            logger.info(f"Probabilities - Cancer: {cancer_prob:.4f}, Normal: {normal_prob:.4f}")
            
            return [{
                "image": prediction,
                "confidence": round(confidence, 4),
                "class_index": int(predicted_class_idx),
                "probabilities": {
                    "adenocarcinoma": round(cancer_prob, 4),
                    "normal": round(normal_prob, 4)
                }
            }]
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise