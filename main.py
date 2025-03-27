import logging
import numpy as np
from data.dataset import DataProcessor
from model.model import EmotionModel
from trainer.trainer import TrainerSetup
from utils.metrics import MetricsCalculator
from config import Config


def setup_logging():
    logging.basicConfig(
        filename='memory_usage.log', 
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    
    # Data preparation
    print("Preparing data...")
    processor = DataProcessor(Config.MODEL_NAME)
    train_df = processor.load_data(Config.TRAIN_FILE)
    val_df = processor.load_data(Config.VAL_FILE)
    
    train_dataset = processor.create_dataset(train_df)
    val_dataset = processor.create_dataset(val_df)
    
    # Model setup
    print("Initializing model...")
    emotion_model = EmotionModel()
    
    # Training
    print("Starting training...")
    trainer_setup = TrainerSetup()
    trainer = trainer_setup.create_trainer(
        model=emotion_model.get_model(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        compute_metrics=MetricsCalculator.compute_metrics
    )
    
    trainer.train()
    
    # Evaluation
    print("\nEvaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save model
    print("\nSaving the model...")
    emotion_model.save(Config.MODEL_SAVE_PATH)
    processor.tokenizer.save_pretrained(Config.TOKENIZER_SAVE_PATH)
    
    # Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = val_df['label'].tolist()
    
    print(MetricsCalculator.generate_classification_report(labels, preds))
    MetricsCalculator.plot_confusion_matrix(labels, preds)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()