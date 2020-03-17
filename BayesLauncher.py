import pandas as pd
import pymorphy2
import nltk
import numpy

analyzer = pymorphy2.MorphAnalyzer()


def read_from_excel():
    return pd.read_excel("movies.xlsx")


def get_reviews_by_label(label):
    df = read_from_excel()
    return df[df.label == label]["text"]


def normalize_words(rows):
    result = list()
    for row in rows:
        words = nltk.word_tokenize(row)
        for word in words:
            word1 = analyzer.parse(word)[0]
            result.append(word1.normal_form)
        return result


def normalize_words_for_row(row):
    result = list()
    words = nltk.word_tokenize(row)
    for word in words:
        word1 = analyzer.parse(word)[0]
        result.append(word1.normal_form)
    return result


ds_all_review_only_text = read_from_excel()["text"]
ds_normalized_all_words = normalize_words(ds_all_review_only_text)
length_unique_words = len(numpy.unique(ds_normalized_all_words))

ds_all_positive_words = normalize_words(get_reviews_by_label(1))
length_all_positive_words = len(ds_all_positive_words)

ds_all_neutral_words = normalize_words(get_reviews_by_label(0))
length_all_neutral_words = len(ds_all_neutral_words)

ds_all_negative_words = normalize_words(get_reviews_by_label(-1))
length_all_negative_words = len(ds_all_negative_words)

words_chance = {}
for word in numpy.unique(ds_normalized_all_words):
    words_chance[word] = [
        (ds_all_positive_words.count(word) + 1) / (length_all_positive_words + length_unique_words),
        (ds_all_neutral_words.count(word) + 1) / (length_all_neutral_words + length_unique_words),
        (ds_all_negative_words.count(word) + 1) / (length_all_negative_words + length_unique_words)
    ]

words_type = {}
for word in words_chance:
    max_value = max(word)
    if word.index(max_value):
        words_type[word] = 1
    elif word.index(max_value):
        words_type[word] = 0
    else:
        words_type[word] = -1


def get_reviews_by_title(title):
    df = read_from_excel()
    return df[df.title == title]["text"]


ds_reviews_django = get_reviews_by_title("Джанго освобожденный")

for review in ds_reviews_django:
    count_positive = 0
    count_neutral = 0
    count_negative = 0
    normalize_review_django = normalize_words_for_row(review)
   
    for word in normalize_review_django:
        if words_type[word] == 1:
            count_positive = count_positive + 1
        elif words_type[word] == 0:
            count_neutral = count_neutral + 1
        else:
            count_negative = count_negative + 1
    print(count_positive)
    print(count_neutral)
    print(count_negative)
