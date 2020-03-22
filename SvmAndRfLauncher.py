import pandas as pd
import pymorphy2
import nltk
import string
from nltk.corpus import stopwords

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support

pd.options.mode.chained_assignment = None
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
            # добавляем слово если оно не в стоп-листе
            if word not in stopwords.words('russian'):
                review_words.append(word)
        result.append(review_words)
    return result


def get_array_with_unique_words(reviews):
    result = []
    for index, entry in enumerate(reviews['text']):
        for word in entry:
            result.append(word)
    return list(set(result))


def to_meshok(array_of_unique_words, reviews):
    result = []
    for review in reviews["text"]:
        result_from_review = []
        for word in array_of_unique_words:
            occurrences = review.count(word)
            result_from_review.append(occurrences)
        result.append(result_from_review)

    return result


def get_part_of_speech_count(reviews):
    result = []
    for index, entry in enumerate(reviews['text']):
        nouns_counter = 0
        verbs_counter = 0
        adjectives_counter = 0
        adverbs_counter = 0
        for word in entry:
            word = analyzer.parse(word)[0].tag.POS
            if word == 'NOUN':
                nouns_counter += 1
            if word == 'ADJF' or word == 'ADJS' or word == 'COMP':
                adjectives_counter += 1
            if word == 'VERB' or word == 'INFN':
                verbs_counter += 1
            if word == 'ADVB':
                adverbs_counter += 1
        result.append([nouns_counter, verbs_counter, adjectives_counter, adverbs_counter])
    return result


def get_punctuation_count(reviews):
    result = []
    for index, entry in enumerate(reviews['text']):
        result_row = []
        for punctuation in string.punctuation:
            single_punctuation_counter = 0
            for word in entry:
                if word == punctuation:
                    single_punctuation_counter += 1
            result_row.append(single_punctuation_counter)
        result.append(result_row)
    return result


file_name_df_normalized_training = "df_normalized_training.pkl"
file_name_df_normalized_test = "df_normalized_test.pkl"

file_name_array_meshok_training = "meshok_training.pkl"
file_name_array_meshok_test = "meshok_test.pkl"

Reviews = read_from_excel()

# получаем данные из всех отзывов, кроме своих (тренировочные данные)
Reviews_training = Reviews.drop(Reviews.index[452:520])

# получаем данные из своих отзывов (тестовые данные)
Reviews_test = Reviews[452:520]

# # нормализуем тренировочные данные
# Reviews_training["text"] = get_column_with_normalized_words(Reviews_training)
# save_to_pickle(Reviews_training, file_name_df_normalized_training)
#
# # нормализуем тестовые данные
# Reviews_test["text"] = get_column_with_normalized_words(Reviews_test)
# save_to_pickle(Reviews_test, file_name_df_normalized_test)

Reviews_training = read_from_pickle(file_name_df_normalized_training)
Reviews_test = read_from_pickle(file_name_df_normalized_test)

#
# unique_words = get_array_with_unique_words(Reviews_training)
# # Reviews_test["unique_text"] = get_column_with_unique_words(Reviews_test)
#
# # мешок тренировочных данных
# meshok_training = to_meshok(unique_words, Reviews_training)
# save_to_pickle(meshok_training, file_name_array_meshok_training)
#
# # мешок тестовых данных
# meshok_test = to_meshok(unique_words, Reviews_test)
# save_to_pickle(meshok_test, file_name_array_meshok_test)

# прочитать мешки из файлов
meshok_training = read_from_pickle(file_name_array_meshok_training)
meshok_test = read_from_pickle(file_name_array_meshok_test)

# тестируем с SVM
SVM = svm.LinearSVC(max_iter=100000)
SVM.fit(meshok_training, Reviews_training["label"])
SVM_prediction = SVM.predict(meshok_test)

# тестируем с Random Forest
clf = RandomForestClassifier(max_depth=20)
clf.fit(meshok_training, Reviews_training["label"])
clf_prediction = clf.predict(meshok_test)

# # считаем метрики для SVM
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("!!!! Просто мешок !!!!")
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
print()

# считаем метрики для Random Forest
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], clf_prediction)
print("Precision for Random Forest:")
print(precision)
print("Recall for Random Forest:")
print(recall)
print("Fscore for Random Forest:")
print(fscore)
print()

# считаем и добавляем в мешок части речи
part_of_speech_training = get_part_of_speech_count(Reviews_training)
part_of_speech_test = get_part_of_speech_count(Reviews_test)

index = 0
for row in meshok_training:
    for row_speech in part_of_speech_training[index]:
        row.append(row_speech)
    index += 1

index = 0
for row in meshok_test:
    for row_speech in part_of_speech_test[index]:
        row.append(row_speech)
    index += 1

# считаем и добавляем в мешок пунктуацию
punctuation_training = get_punctuation_count(Reviews_training)
punctuation_test = get_punctuation_count(Reviews_test)

index = 0
for row in meshok_training:
    for row_punctuation in punctuation_training[index]:
        row.append(row_punctuation)
    index += 1

index = 0
for row in meshok_test:
    for row_punctuation in punctuation_test[index]:
        row.append(row_punctuation)
    index += 1

# тестируем с SVM
SVM = svm.LinearSVC(max_iter=100000)
SVM.fit(meshok_training, Reviews_training["label"])
SVM_prediction = SVM.predict(meshok_test)

#  считаем метрики для SVM - мешок, части речи и пунктуация
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("!!!! Мешок + части речи + пунктуация !!!!")
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
print()


# тестируем с SVM
SVM = svm.LinearSVC(max_iter=100000)
meshok_training_except_bag = meshok_training
for row in meshok_training_except_bag:
    row[0:20159] = []
meshok_test_except_bag = meshok_test
for row in meshok_test_except_bag:
    row[0:20159] = []
SVM.fit(meshok_training_except_bag, Reviews_training["label"])
SVM_prediction = SVM.predict(meshok_test_except_bag)

#  считаем метрики для SVM - мешок, части речи и пунктуация
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("!!!! части речи + пунктуация !!!!")
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
print()

# тестируем с SVM
SVM = svm.LinearSVC(max_iter=100000)
meshok_training_except_part_of_speech = meshok_training
for row in meshok_training_except_part_of_speech:
    row[20160:20163] = []
meshok_test_except_part_of_speech = meshok_test
for row in meshok_test_except_part_of_speech:
    row[20160:20163] = []
SVM.fit(meshok_training_except_part_of_speech, Reviews_training["label"])
SVM_prediction = SVM.predict(meshok_test_except_part_of_speech)

#  считаем метрики для SVM - мешок, части речи и пунктуация
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("!!!! мешок + пунктуация !!!!")
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
print()

# тестируем с SVM
SVM = svm.LinearSVC(max_iter=100000)
meshok_training_except_punkt = meshok_training
for row in meshok_training_except_punkt:
    row[20163:20195] = []
meshok_test_except_punkt = meshok_test
for row in meshok_test_except_punkt:
    row[20163:20195] = []
SVM.fit(meshok_training_except_punkt, Reviews_training["label"])
SVM_prediction = SVM.predict(meshok_test_except_punkt)

#  считаем метрики для SVM - мешок, части речи и пунктуация
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("!!!! мешок + части речи !!!!")
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
print()
