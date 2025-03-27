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
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 64
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LOGGING_STEPS = 10