class Config:
    MODEL_NAME = 'tahrirchi/tahrirchi-bert-small'
    NUM_LABELS = 6
    MAX_LENGTH = 128
    TRAIN_FILE = 'data/raw/train.csv'
    VAL_FILE = 'data/raw/validation.csv'
    OUTPUT_DIR = "./results"
    LOGGING_DIR = "./logs"
    MODEL_SAVE_PATH = "model/emotion_classifier_model"
    TOKENIZER_SAVE_PATH = "model/emotion_classifier_tokenizer"

    # Training arguments
    EPOCHS = 4
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 64
    WARMUP_STEPS = 300
    WEIGHT_DECAY = 0.005
    LOGGING_STEPS = 20
    LEARNING_RATE = 2e-5
    GRADIENT_ACCUMULATION_STEPS = 2
    LR_SCHEDULER_TYPE = "cosine"
    EARLY_STOPPING_PATIENCE = 2