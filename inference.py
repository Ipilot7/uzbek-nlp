from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Union
import numpy as np
from config import Config
class EmotionClassifier:
    """
    Классификатор эмоций для узбекского текста на основе fine-tuned модели
    
    Args:
        model_path (str): Путь к сохраненной модели
        tokenizer_path (str): Путь к сохраненному токенизатору
        device (str): Устройство для вычислений ('cuda' или 'cpu')
    """
    
    def __init__(self, 
                 model_path: str = Config.MODEL_SAVE_PATH, 
                 tokenizer_path: str = Config.TOKENIZER_SAVE_PATH,
                 device: str = None):
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Метки классов (должны соответствовать порядку при обучении)
        self.class_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        # self.class_labels = ["печаль", "радость", "любовь", "гнев", "страх", "удивление"]
        
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Загрузка модели"""
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Переводим модель в режим inference
    
    def _load_tokenizer(self):
        """Загрузка токенизатора"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
    
    def preprocess(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Токенизация входного текста
        
        Args:
            text: Входной текст или список текстов
            
        Returns:
            Словарь с токенизированными входами
        """
        return self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
    
    def predict(self, text: Union[str, List[str]], return_probs: bool = False) -> Union[str, List[str], Dict]:
        """
        Предсказание эмоции для текста
        
        Args:
            text: Входной текст или список текстов
            return_probs: Если True, возвращает вероятности для всех классов
            
        Returns:
            Предсказанная эмоция или словарь с подробными результатами
        """
        # Токенизация
        inputs = self.preprocess(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Обработка выходов
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        
        # Преобразование в метки классов
        results = [self.class_labels[p] for p in preds]
        
        if isinstance(text, str):
            results = results[0]
            probs = probs[0].cpu().numpy() if return_probs else None
        else:
            probs = probs.cpu().numpy() if return_probs else None
        
        if return_probs:
            return {
                "predictions": results,
                "probabilities": probs,
                "class_labels": self.class_labels
            }
        return results
    
    def predict_with_confidence(self, text: str) -> Dict:
        """
        Предсказание с возвратом уверенности модели
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с предсказанием и информацией об уверенности
        """
        result = self.predict(text, return_probs=True)
        
        if isinstance(text, str):
            top_prob = np.max(result["probabilities"])
            top_class = result["predictions"]
            
            return {
                "emotion": top_class,
                "confidence": float(top_prob),
                "probabilities": dict(zip(result["class_labels"], result["probabilities"].tolist()))
            }
        
        return result


if __name__ == "__main__":
    # Пример использования
    classifier = EmotionClassifier()
    
    # Тестовые примеры
    test_texts = [
        "Men bugun juda xursandman!",  # Радость
        "Nima qilayotganimni bilmayman, juda qo'rqyapman",  # Страх
        "Bu mening eng yomon kunim",  # Грусть
        "Aqlsiz odamlar bilan gaplashyotganimda asabiylashyapman!"  # Гнев
    ]
    
    # Предсказание для одного текста
    single_text = test_texts[0]
    print(f"Text: '{single_text}'")
    print(f"Predicted emotion: {classifier.predict(single_text)}")
    print(f"With confidence: {classifier.predict_with_confidence(single_text)}")
    
    # Предсказание для нескольких текстов
    print("\nBatch prediction:")
    batch_results = classifier.predict(test_texts, return_probs=True)
    for text, pred, probs in zip(test_texts, batch_results["predictions"], batch_results["probabilities"]):
        print(f"\nText: '{text}'")
        print(f"Predicted emotion: {pred}")
        print("Probabilities:")
        for label, prob in zip(batch_results["class_labels"], probs):
            print(f"  {label}: {prob:.4f}")