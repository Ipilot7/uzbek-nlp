import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class MetricsCalculator:
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted")
        }
    
    @staticmethod
    def generate_classification_report(labels, preds):
        emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        return classification_report(labels, preds, target_names=emotion_labels)
    
    @staticmethod
    def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png'):
        emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        #emotion_labels = ["печаль", "радость", "любовь", "гнев", "страх", "удивление"]
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_path)
        plt.close()