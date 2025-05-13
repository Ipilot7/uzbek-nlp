import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import FastText
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Загрузка данных
train_df = pd.read_csv("data/raw/train.csv")
val_df = pd.read_csv("data/raw/validation.csv")

# Общая функция предобработки текста
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\sʼʻ’]", " ", text)
    text = re.sub(r"[ʼʻ’]", "'", text)
    return text

train_df["text_processed"] = train_df["uzbek_text"].apply(preprocess)
val_df["text_processed"] = val_df["uzbek_text"].apply(preprocess)

# Разделение данных
X_train, y_train = train_df["text_processed"], train_df["label"]
X_val, y_val = val_df["text_processed"], val_df["label"]

# ==============================================
# Модель 2: FastText + Логистическая регрессия
# ==============================================

print("\n🔧 Training FastText + Logistic Regression...")

# Обучение FastText на наших текстах
sentences = [text.split() for text in X_train]
ft_model = FastText(vector_size=100, window=5, min_count=2, workers=4)
ft_model.build_vocab(sentences)
ft_model.train(sentences, total_examples=len(sentences), epochs=10)

# Векторизация текстов
def text_to_vector(text):
    words = text.split()
    word_vecs = [ft_model.wv[word] for word in words if word in ft_model.wv]
    if len(word_vecs) == 0:
        return np.zeros(100)
    return np.mean(word_vecs, axis=0)

X_train_ft = np.array([text_to_vector(text) for text in X_train])
X_val_ft = np.array([text_to_vector(text) for text in X_val])

# Обучение логистической регрессии
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_ft, y_train)

# Оценка
y_pred = lr_model.predict(X_val_ft)
print("\n📊 FastText + Logistic Regression Report:")
print(classification_report(y_val, y_pred))

# ==============================================
# Модель 3: BiLSTM + GloVe
# ==============================================

print("\n🔧 Training BiLSTM + GloVe...")

# Токенизация текстов
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Паддинг последовательностей
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)

# Загрузка предобученных GloVe векторов (пример для русского, нужен узбекский аналог)
# В реальном проекте нужно загрузить узбекские эмбеддинги
embedding_dim = 100
embedding_matrix = np.random.rand(10000, embedding_dim)  # Заглушка

# Создание модели BiLSTM
model = Sequential()
model.add(Embedding(input_dim=10000, 
                   output_dim=embedding_dim, 
                   input_length=max_len,
                   weights=[embedding_matrix],
                   trainable=False))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение
model.fit(X_train_pad, y_train,
          validation_data=(X_val_pad, y_val),
          epochs=5,
          batch_size=32)

# Оценка
y_pred = (model.predict(X_val_pad) > 0.5).astype("int32")
print("\n📊 BiLSTM + GloVe Report:")
print(classification_report(y_val, y_pred))

# ==============================================
# Модель 4: mBERT (без дообучения)
# ==============================================

print("\n🔧 Evaluating mBERT...")

# Загрузка предобученной модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')

# Подготовка данных для BERT
def convert_examples_to_tfdataset(texts, labels):
    inputs = tokenizer(texts.tolist(), 
                      padding=True, 
                      truncation=True, 
                      max_length=128, 
                      return_tensors="tf")
    return tf.data.Dataset.from_tensor_slices((
        dict(inputs),
        labels
    )).batch(16)

train_data = convert_examples_to_tfdataset(X_train, y_train)
val_data = convert_examples_to_tfdataset(X_val, y_val)

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Оценка (без дообучения)
loss, accuracy = model.evaluate(val_data)
print(f"\n📊 mBERT Accuracy: {accuracy:.4f}")

# Для реального использования нужно дообучение:
# model.fit(train_data, epochs=2, validation_data=val_data)