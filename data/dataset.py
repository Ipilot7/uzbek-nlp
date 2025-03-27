import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict, List
from config import Config

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)

class DataProcessor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(self, texts: List[str]) -> Dict:
        return self.tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=Config.MAX_LENGTH
        )
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    def create_dataset(self, df: pd.DataFrame) -> EmotionDataset:
        encodings = self.tokenize(df['uzbek_text'].tolist())
        return EmotionDataset(encodings, df['label'].tolist())