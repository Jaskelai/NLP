import pandas as pd
import string

from sklearn import svm
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support


def read_from_excel():
    return pd.read_excel("movies.xlsx")


def read_from_pickle(name):
    return pd.read_pickle(name)


def remove_punctuation(column):
    result = []
    for row_review in column:
        temp = [''.join(c for c in s if c not in string.punctuation) for s in row_review]
        temp = [s for s in temp if s]
        result.append(temp)
    return result


def get_df_train_with_vectors(column, size):
    columns = [0] * size
    result = pd.DataFrame(columns=columns)
    i = 0
    for review in column:
        i += 1
        result_row = []
        for word in review:
            vector = model.wv[word]
            result_row.append(vector)
        result.loc[i] = get_array_with_average(result_row, size)
    return result


def get_df_test_with_vectors(column, size):
    columns = [0] * size
    result = pd.DataFrame(columns=columns)
    index = 0
    for review in column:
        index += 1
        array_of_arrays = []
        for word in review:
            if word in list(model.wv.vocab.keys()):
                vector = model.wv[word]
                array_of_arrays.append(vector)
            else:
                vector = []
                for i in range(SIZE):
                    vector.append(0)
                array_of_arrays.append(vector)
        result.loc[index] = get_array_with_average(array_of_arrays, SIZE)
    return result


def get_array_with_average(array_of_arrays, size):
    result = [0] * size
    for array in array_of_arrays:
        for j in range(len(array)):
            result[j] += array[j]
    i = 0
    for value in result:
        result[i] = value / len(array_of_arrays)
    return result


file_name_df_normalized_training = "df_normalized_training.pkl"
file_name_df_normalized_test = "df_normalized_test.pkl"

# получаем нормализированные и токенизированные отзывы
Reviews_training = read_from_pickle(file_name_df_normalized_training)
Reviews_test = read_from_pickle(file_name_df_normalized_test)

# убираем пунктуацию
Reviews_training["text"] = remove_punctuation(Reviews_training["text"])
Reviews_test["text"] = remove_punctuation(Reviews_test["text"])
Reviews = Reviews_training.append(Reviews_test)

# обучаем модель
model = Word2Vec(sentences=Reviews["text"], min_count=0)

# синонимы у 5 слов
print(model.wv.most_similar("сюжет"))
print(model.wv.most_similar("картинка"))
print(model.wv.most_similar("музыка"))
print(model.wv.most_similar("тарантино"))
print(model.wv.most_similar("актёр"))

# получаем усредненное векторное представление слов
SIZE = 100
Train_vect = get_df_train_with_vectors(column=Reviews_training["text"], size=SIZE)
Test_vect = get_df_test_with_vectors(column=Reviews_test["text"], size=SIZE)

# используем с SVM
SVM = svm.LinearSVC(max_iter=1000000)
SVM.fit(Train_vect, Reviews_training["label"])
SVM_prediction = SVM.predict(Test_vect)
precision, recall, fscore, support = precision_recall_fscore_support(Reviews_test["label"], SVM_prediction)
print("Precision for SVM:")
print(precision)
print("Recall for SVM:")
print(recall)
print("Fscore for SVM:")
print(fscore)
