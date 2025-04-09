from transformers import AutoModelForSequenceClassification
from config import Config

class EmotionModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME, 
            num_labels=Config.NUM_LABELS
        )
    
    def save(self, path: str):
        self.model.save_pretrained(path)
    
    def get_model(self):
        return self.model