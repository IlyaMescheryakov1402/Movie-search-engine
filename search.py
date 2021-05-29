from math import nan
import re
import copy
import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
import math

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
sw_eng = set(stopwords.words('english'))
stemmer = SnowballStemmer(language='english')

# Определяем часть речи слова
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Общая функция очистки текста    
def clean_text(text):

    # Делаем буквы строчными
    text = text.lower()

    # Разделяем слова на токены по знакам пунктуации
    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # Убираем цифры из токенов
    text = [word for word in text if not any(c.isdigit() for c in word)]

    # Убираем стоп-слова из токенов
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]

    # Убираем случайные пустые токены
    text = [t for t in text if len(t) > 0]

    # определяем часть речи
    pos_tags = pos_tag(text)

    # Проводим лемматизацию в соответствии с частью речи
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # убираем слова длины в 1 символ
    text = [t for t in text if len(t) > 1]

    # Выполняем стемминг
    text = " ".join(text)
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return(text)

# Класс, в котором отображаются документы по запросу
class Document:
    def __init__(self, title, text):
        self.title = title
        self.text = text
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text]

index = {}
raw_data = []

# Считывает данные из датасета
data = pd.read_csv('tmdb_5000_movies.csv')
data = data[['original_title', 'overview', ]]

# Cчитывает сырые данные и строит инвертированный индекс
def build_index():

    # Считываем сырые данные
    for i in range(len(data)):
        raw_data.append(Document(str(data.loc[i, 'original_title']), str(data.loc[i, 'overview'])))

    # Для каждого документа из raw_data:
    for idx, doc in enumerate(raw_data):

        # Для каждого слова из очищенного набора слов описания:
        for word in (' '.join([word for word in doc.text.split() if not word in sw_eng])).split():

            # Если слова нет в index - создаем под него пару ключ-значение, где ключ - слово, значение - индекс документа из raw_data
            if word.lower() not in index:
                index[word.lower()] = []

            # Заносим номер документа в словарь индексов по соответствующему ключу (слову)
            index[word.lower()].append(idx)

        # То же самое, но для названий фильма
        for word in (' '.join([word for word in doc.title.split() if not word in sw_eng])).split():
            if word.lower() not in index:
                index[word.lower()] = []
            index[word.lower()].append(idx)
    
    # Отображаем длину составленного инвертированного индекса
    print('Количество ключей в инвертированном индексе = ', len(index.keys()))
        


# Возвращает скор для пары запрос-документ. Больше -> Релевантнее
def score(query, document):

    # Если очередь пустая - возвращаем случайный скор
    if query == '':
        return random.random()

    # Если очередь не пустая:
    else:

        # Делаем очистку текста
        text = clean_text(document.text)

        # Делаем TF-IDF для очереди и для документа
        vec = TfidfVectorizer()
        document_tfidf = vec.fit_transform([text]).todense()
        query_tfidf = vec.transform(query.split()).todense()

        # Используем косинусную схожесть
        rec_idx = 1. - cdist(query_tfidf, document_tfidf, 'cosine')[0, 0]

        # Достаем название документа и убираем все лишнее из названия
        title = re.sub(r'[^a-zA-z\s]', '', document.title)

        # Увеличиваем вес, если слово из очереди встречается в названии
        for word in query.split():
            if word in title.lower():
                rec_idx = rec_idx + 0.5
            
        # Отлавливаем НаНы
        if math.isnan(rec_idx):
            rec_idx = 0

        return rec_idx

# Возвращает начальный список релевантных документов
def retrieve(query):

    # В candidates записываем конечный набор документов, который будет выводиться на экран
    candidates = []

    # sets будет служить двумерным списком, в котором будут сравниваться по значениям разные ключи (слова) словаря index
    sets = []

    # через mid_raw_data вывод candidates по индексам raw_data
    mid_raw_data = []

    # Если пустой запрос - пусть возвращает первые 50 документов сырых данных
    if query == '':
        return raw_data[:50]
    
    # Токенизируем запрос и достаем его длину
    query = query.split()
    n = len(query)

    # Если длина запроса - 1 слово, то в релевантные документы заносим все из списка по инвертированному индексу
    if n == 1:
        candidates.extend(index[query[0].lower()])

    # Если нет
    else:
        # Создаем массив нулей размером с длину очереди
        sets = [0] * n

        # Для каждого слова очереди в массив sets заносим копию списка документов (описаний), в которых это слово встречается
        for word_idx, word in enumerate(query):
            sets[word_idx] = copy.deepcopy(index[word.lower()])

            # Печатаем длину каждого списка в sets (для отладки)
            print('len(sets[{}]]) = '.format(word_idx), len(sets[word_idx]))

        # Сравниваются списки индексированных статей для каждых ДВУХ слов запроса, пересечения записываются в candidates
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                candidates.extend(list(set(sets[i]) & set(sets[j])))

        # Если пересечений так и не нашлось, возвращаем списки релевантных документов для каждого слова по отдельности
        if len(candidates) <= 5:

            # Печатаем, что пересечений не нашлось (для отладки)
            print('They have few or no shared candidates!!!')
            for i in range(n):
                candidates.extend(index[query[i].lower()])
    
    # Печатаем длину конечного набора документов
    print('len(candidates) = ', len(candidates))

    # Превращаем массив индексов в массив документов
    for i in set(candidates):
        mid_raw_data.append(raw_data[i])
    
    # Выводим 100 документов
    return mid_raw_data[:100]