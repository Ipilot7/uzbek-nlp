from classifier import EmotionClassifier

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
    # emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    #emotion_labels = ["печаль", "радость", "любовь", "гнев", "страх", "удивление"]
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