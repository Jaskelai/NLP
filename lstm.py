import pandas as pd
import numpy as np
import keras

from keras. preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM

# убирает мусор из консоли
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_from_pickle(name):
    return pd.read_pickle(name)


def create_embedding_matrix(df, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for index in range(len(df["word"])):
        if df["word"][index] in word_index:
            idx = word_index[df["word"][index]]
            embedding_matrix[idx] = np.array(
                df["value"][index], dtype=np.float32)[:embedding_dim]

    return embedding_matrix


filename = "meh.pkl"
file_name_df_normalized_training = "df_normalized_training.pkl"
file_name_df_normalized_test = "df_normalized_test.pkl"
df_train = read_from_pickle(filename)

# получаем словарь с закодированными словами
tokenizer = Tokenizer(num_words=189193)
tokenizer.fit_on_texts(df_train["word"])

# читаем дф с отзывами
Reviews_train = read_from_pickle(file_name_df_normalized_training)
Reviews_test = read_from_pickle(file_name_df_normalized_test)

# кодируем отзывы
Reviews_train["text"] = tokenizer.texts_to_sequences(Reviews_train["text"])
Reviews_test["text"] = tokenizer.texts_to_sequences(Reviews_test["text"])

# подготавливаем отзывы (дополняем до длины в 300 и до 3 в случае с классами)
reviews_train_prepared = pad_sequences(Reviews_train["text"].to_numpy(), maxlen=300, padding='post')
reviews_test_prepared = pad_sequences(Reviews_test["text"].to_numpy(), maxlen=300, padding='post')
labels_train_prepared = keras.utils.to_categorical(Reviews_train["label"], 3)
labels_test_prepared = keras.utils.to_categorical(Reviews_test["label"], 3)

# создаем эмбеддинг
embedding_matrix = create_embedding_matrix(df_train, tokenizer.word_index, 300)

batch_size = 32
epochs = 3

# создаем нейронку, добавляем слои, компилируем
model = Sequential()
model.add(Embedding(142734, 300, weights=[embedding_matrix], input_length=300))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))

model.compile(metrics=["accuracy"], optimizer='adam', loss='binary_crossentropy')
result = model.fit(reviews_train_prepared, labels_train_prepared, epochs=10, batch_size=batch_size, validation_data=(reviews_test_prepared, labels_test_prepared))

score = model.evaluate(reviews_test_prepared, labels_test_prepared, batch_size=batch_size, verbose=1)

print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))
