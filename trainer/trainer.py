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
        cfg = Config()
        return TrainingArguments(
            output_dir=cfg.OUTPUT_DIR,
            num_train_epochs=cfg.EPOCHS,
            per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
            warmup_steps=cfg.WARMUP_STEPS,
            weight_decay=cfg.WEIGHT_DECAY,
            learning_rate=cfg.LEARNING_RATE,
            logging_dir=cfg.LOGGING_DIR,
            logging_steps=cfg.LOGGING_STEPS,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            disable_tqdm=False
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