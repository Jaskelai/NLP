import pandas as pd
import gensim
import nltk
import pymorphy2

from nltk.corpus import stopwords
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

analyzer = pymorphy2.MorphAnalyzer()


def read_from_excel():
    return pd.read_excel("movies.xlsx")


def save_to_pickle(df, name):
    pd.to_pickle(df, name)


def read_from_pickle(name):
    return pd.read_pickle(name)


def get_column_with_normalized_words(reviews):
    # приводим слова к нижнему кейсу
    reviews['text'] = [entry.lower() for entry in reviews['text']]
    # токенизируем (разделяем) слова
    reviews['text'] = [nltk.word_tokenize(entry) for entry in reviews['text']]

    result = []
    for index, entry in enumerate(reviews['text']):
        review_words = []
        for word in entry:
            word = analyzer.parse(word)[0].normal_form
            # добавляем слово, если оно не в стоп-листе
            if word not in stopwords.words('russian'):
                review_words.append(word)
        result.append(review_words)
    return result


def remove_punctuation(column):
    result = []
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~–«»"""
    for row_review in column:
        temp = [''.join(c for c in s if c not in punctuation) for s in row_review]
        temp = [s for s in temp if s]
        result.append(temp)
    return result


file_name_df_normalized_reviews = "df_normalized.pkl"

# # читаем дф с отзывами
# Reviews = read_from_excel()
#
# # нормализуем тренировочные данные
# Reviews["text"] = get_column_with_normalized_words(Reviews)
# Reviews["text"] = remove_punctuation(Reviews["text"])
# save_to_pickle(Reviews, file_name_df_normalized_reviews)

Reviews = read_from_pickle(file_name_df_normalized_reviews)
dictionary = corpora.Dictionary(Reviews["text"])
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in Reviews["text"]]
print(corpus)

NUM_TOPICS = 10
model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)

NUM_WORDS = 15
topics = model.print_topics(num_words=NUM_WORDS)
print(topics)

cm = CoherenceModel(model=model, texts=Reviews["text"], dictionary=dictionary)
coherence = cm.get_coherence()
print(coherence)
