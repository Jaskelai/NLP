import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

tfidf = TfidfVectorizer(max_features=500)


def read_from_excel():
    return pd.read_excel("movies.xlsx")


def save_to_pickle(df, name):
    pd.to_pickle(df, name)


def read_from_pickle(name):
    return pd.read_pickle(name)


file_name_df_normalized_training = "df_normalized_training.pkl"
file_name_df_normalized_test = "df_normalized_test.pkl"

file_name_array_meshok_training = "meshok_training.pkl"
file_name_array_meshok_test = "meshok_test.pkl"

Reviews = read_from_excel()

# получаем данные из всех отзывов, кроме своих (тренировочные данные)
Reviews_training = Reviews.drop(Reviews.index[452:520])

# получаем данные из своих отзывов (тестовые данные)
Reviews_test = Reviews[452:520]

Reviews_training = read_from_pickle(file_name_df_normalized_training)
Reviews_test = read_from_pickle(file_name_df_normalized_test)
Reviews_training["text"] = [" ".join(entry) for entry in Reviews_training['text']]
Reviews_test["text"] = [" ".join(entry) for entry in Reviews_test['text']]

# tfidf

tfidf.fit_transform(Reviews["text"])
Train_X_Tfidf = tfidf.fit_transform(Reviews_training["text"])
Test_X_Tfidf = tfidf.fit_transform(Reviews_test["text"])

# keras
Train_Y_keras = keras.utils.to_categorical(Reviews_training["label"], 3)
Test_Y_keras = keras.utils.to_categorical(Reviews_test["label"], 3)
model = Sequential()
model.add(Dense(512, input_shape=(500,)))
model.add(Dropout(0.5))
model.add(Dense(3))
model.compile(metrics=["accuracy"], optimizer='adam', loss='categorical_crossentropy')

model.fit(Train_X_Tfidf, Train_Y_keras, epochs=10, batch_size=32)

result = model.predict(Test_X_Tfidf)

print(classification_report(Test_Y_keras.argmax(axis=1), result.argmax(axis=1)))
