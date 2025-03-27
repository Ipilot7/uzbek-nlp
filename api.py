from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict_emotion(request: TextRequest):
    model.eval()
    tokens = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).item()

    return {"emotion": prediction}

# Запустить сервер: uvicorn main:app --reload
