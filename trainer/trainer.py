from transformers import TrainingArguments, Trainer
from utils.memory import MemoryTracker
from config import Config

class CustomTrainer(Trainer, MemoryTracker):
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        MemoryTracker.__init__(self)
        self.step_count = 0
        self.print_frequency = 100
    
    def training_step(self, model, inputs, num_items_in_batch):
        self.step_count += 1
        
        if self.step_count % self.print_frequency == 0:
            self.log_memory_usage(self.step_count, "Training ")
        
        return super().training_step(model, inputs,num_items_in_batch)

class TrainerSetup:
    @staticmethod
    def get_training_args():
        return TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            num_train_epochs=Config.EPOCHS,
            per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
            warmup_steps=Config.WARMUP_STEPS,
            weight_decay=Config.WEIGHT_DECAY,
            logging_dir=Config.LOGGING_DIR,
            logging_steps=Config.LOGGING_STEPS,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",
        )
    
    def create_trainer(self, model, train_dataset, val_dataset, compute_metrics):
        args = self.get_training_args()
        return CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )