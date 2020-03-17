import pandas as pd
import pymorphy2
import nltk
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

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


# all reviews
df_all_all_reviews = read_from_excel()
df_all_reviews = df_all_all_reviews.drop(df_all_all_reviews.index[452:520])
array_reviews = df_all_reviews["text"].to_numpy()
array_ratings = df_all_reviews["label"].to_numpy()

# my reviews
df_all_my_reviews = read_from_excel()[452:520]
array_reviews_my_reviews = df_all_my_reviews["text"].to_numpy()

# normalize words from all reviews
array_normalized_words = normalize_words(df_all_reviews)
array_normalized_words_unique = list(set(array_normalized_words))

# get and save meshok all reviews
meshok = to_meshok(array_normalized_words_unique, array_reviews)
save_array_to_file(meshok, "meshok.txt")

# get and save meshok my reviews
meshok_my_reviews = to_meshok(array_normalized_words_unique, array_reviews_my_reviews)
save_array_to_file(meshok_my_reviews, "meshok-my-reviews.txt")

# read meshok all reviews
meshok = read_array_from_file("meshok.txt")
# read meshok my reviews
meshok_my_reviews = read_array_from_file("meshok-my-reviews.txt")

# predict
model.fit(meshok, array_ratings)
predicted = model.predict(meshok_my_reviews)

# count true positives
true_positives = 0
i = 0
j = 1
for prediction in predicted:
    if prediction == df_all_my_reviews[i:j]["label"].values[0]:
        true_positives += 1
    i += 1
    j += 1

# calculate accuracy
accuracy = true_positives / len(predicted)
print("Accuracy:")
print(accuracy)

precision, recall, fscore, support = precision_recall_fscore_support(df_all_my_reviews['label'].values, predicted)

# calculate precision
print("Precision:")
print(precision)

# calculate recall
print("Recall:")
print(recall)

# calculate Fscore
print("Fscore:")
print(fscore)

# extract weights
positive_ratings_dict = dict(zip(array_normalized_words_unique, model.coef_[2]))
neutral_ratings_dict = dict(zip(array_normalized_words_unique, model.coef_[1]))
negative_ratings_dict = dict(zip(array_normalized_words_unique, model.coef_[0]))

# map positive weights to words
sorted_positive_dictionary = {k: v for k, v in sorted(positive_ratings_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_positive_list = list(sorted_positive_dictionary)
first_10_positive = sorted_positive_list[0:10]
print("Positive top 10:")
print(first_10_positive)
reversed_sorted_positive_list = sorted_positive_list[::-1]
last_10_positive = reversed_sorted_positive_list[0:10]
print("Positive last 10:")
print(last_10_positive)

# map neutral weights to words
sorted_neutral_dictionary = {k: v for k, v in sorted(neutral_ratings_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_neutral_list = list(sorted_neutral_dictionary)
first_10_neutral = sorted_neutral_list[0:10]
print("Neutral top 10:")
print(first_10_neutral)
reversed_sorted_neutral_list = sorted_neutral_list[::-1]
last_10_neutral = reversed_sorted_neutral_list[0:10]
print("Neutral last 10:")
print(last_10_neutral)

# map negative weights to words
sorted_negative_dictionary = {k: v for k, v in sorted(negative_ratings_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_negative_list = list(sorted_negative_dictionary)
first_10_negative = sorted_negative_list[0:10]
print("Negative top 10:")
print(first_10_negative)
reversed_sorted_negative_list = sorted_negative_list[::-1]
last_10_negative = reversed_sorted_negative_list[0:10]
print("Negative last 10:")
print(last_10_negative)
