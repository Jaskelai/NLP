import pandas as pd
import pymorphy2
import nltk
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

analyzer = pymorphy2.MorphAnalyzer()
model = LogisticRegression(max_iter=100000)


def read_from_excel():
    return pd.read_excel("movies.xlsx")


def normalize_words(rows):
    result = []
    for index, row in rows.iterrows():
        words = nltk.word_tokenize(row["text"])
        for word in words:
            word = analyzer.parse(word)[0]
            result.append(word.normal_form)
    return result


def to_meshok(array_of_unique_words, array_of_reviews):
    result = []
    for review in array_of_reviews:
        result_from_review = []
        for word in array_of_unique_words:
            occurrences = review.count(word)
            result_from_review.append(occurrences)
        result.append(result_from_review)

    return result


def save_array_to_file(array, text):
    with open(text, 'wb') as f:
        pickle.dump(array, f)


def read_array_from_file(text):
    with open(text, 'rb') as f:
        return pickle.load(f)


df_all_reviews = read_from_excel()
df_all_my_reviews = read_from_excel()[452:520]
array_reviews = df_all_reviews["text"].to_numpy()
array_ratings = df_all_reviews["label"].to_numpy()
array_my_reviews = df_all_my_reviews["text"].to_numpy()

array_normalized_words = normalize_words(df_all_reviews)
array_normalized_words_unique = list(set(array_normalized_words))
# meshok = to_meshok(array_normalized_words_unique, array_reviews)
# save_array_to_file(meshok, "meshok.txt")

meshok_my_reviews = to_meshok(array_normalized_words_unique, array_my_reviews)
save_array_to_file(meshok_my_reviews, "meshok-my-reviews.txt")

meshok = read_array_from_file("meshok.txt")
meshok_my_reviews = read_array_from_file("meshok-my-reviews.txt")

model.fit(meshok, array_ratings)
predicted = model.predict(meshok_my_reviews)
print(model.coef_)
print(predicted)

true_positives = 0
i = 0
for prediction in predicted:
    if (prediction)
    if prediction == my_reviews.values[i][2]:
        true_positives += 1
    i += 1

print("Accuracy: {}".format(true_positives / len(predicted)))
precision_recall_fscore = precision_recall_fscore_support(my_reviews['label'].values, predicted)
print(f"Precision(-1, 0, 1) = {precision_recall_fscore[0]}")
print(f"Recall(-1, 0, 1) = {precision_recall_fscore[1]}")
print(f"Fscore(-1, 0, 1) = {precision_recall_fscore[2]}")

dict_of_negative = dict(zip(vocab, regression.coef_[0]))
dict_of_neutral = dict(zip(vocab, regression.coef_[1]))
dict_of_positive = dict(zip(vocab, regression.coef_[2]))
