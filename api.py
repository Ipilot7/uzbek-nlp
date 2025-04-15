from fastapi import FastAPI
from pydantic import BaseModel
from classifier import EmotionClassifier  # Импорт твоего класса
from typing import Dict,List

app = FastAPI()

# === Инициализация модели при старте ===
classifier = EmotionClassifier()

# === Pydantic модель для запроса ===
class TextRequest(BaseModel):
    text: str

# ===== Схема для списка текстов =====
class TextListRequest(BaseModel):
    texts: List[str]

# === Эндпоинт предсказания эмоции ===
@app.post("/predict/")
def predict_emotion(request: TextRequest) -> Dict:
    result = classifier.predict_with_confidence(request.text)
    return result

@app.post("/predict_batch/")
def predict_batch_emotions(request: TextListRequest) -> List[Dict]:
    batch_result = classifier.predict(request.texts, return_probs=True)
    
    results = []
    for text, emotion, probs in zip(request.texts, batch_result["predictions"], batch_result["probabilities"]):
        result = {
            "text": text,
            "emotion": emotion,
            "confidence": float(max(probs)),
            "probabilities": dict(zip(batch_result["class_labels"], map(float, probs)))
        }
        results.append(result)
    
    return results

# uvicorn main:app --reload
