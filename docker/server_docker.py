'''server docker part для определения тональности отзывов на телефоны'''

import numpy as np
import os
import pickle as pkl
import socket
import time

import nltk
from nltk.corpus import stopwords
import onnxruntime as rt

from prepare_text_class import TextPrepareClass


HOST = '172.17.0.2'
PORT = 9126

MODEL_PATH = os.path.join('.')
PHONES = r'\b(samsung|galaxy|xiaomi|iphon|redmi|note|honor|huawei|apple|' +\
          'nokia|meizu|google|самсунг|айфон|mi|lenovo|lg|redme|asus|vivo|' +\
          'zte|helio|mediatek|oppo|htc|pixel|xperia|fly|realme|zenfone|' +\
          'alcatel|blade|philips|touch|lumia|oneplus|motorola|inoi|red|neo|' +\
          'moto|panasonic|band|honnor|bbk|vertex|lafleur|xiomi|редми|хонор|' +\
          'ноки|хуаве|мейзу|асус|галакси|иной|гэлакси|хонор)(|[a-zа-я]+)'  # \b

nltk.data.path.append('/usr/src/app/nltk')


def recv_data(inp_socket):
    '''
    Получение данных/команды от клиентской части
    args
        inp_socket - сокет, связанный с клиентской частью
    return
        str - полученные данные/команда
    '''
    inp_socket.settimeout(0)
    package_len = int(inp_socket.recv(5))
    if package_len > 1024:
        return 0

    inp_socket.settimeout(10)
    ret_data = inp_socket.recv(package_len)

    return ret_data


def send_data(inp_socket, inp_data: str):
    '''
    Отпрака данных клиентской части
    args
        inp_socket - сокет, связанный с клиентской частью
        inp_data - данные (текстовые) для отправки
    '''
    package_len = str(len(inp_data)).encode()
    inp_socket.send(package_len)

    time.sleep(1)
    inp_socket.send(inp_data.encode())

    return 0


def work(connection, determine_sentiment) -> int:
    '''
    Основная рабочая функция для взаимодействия с клиентской частью
    args
        connection - сокет, связанный с клиентской частью
        determine_sentiment - класс определения тональности и работы с отзывом
    return
        int - код обрыва связи с клиентской частью или на завершения работы
    '''
    command = connection.recv(1024)
    if not command:
        return -1

    if command.decode('utf-8') == 'exit':
        return -13

    print('recived command: ', command.decode('utf-8'))

    # команда рудимент
    if command.decode('utf-8') == 'get_available_models':
        # print('get_available_models')
        available_models = 'all'
        connection.send(available_models.encode('utf-8'))
        return 0

    if command.decode('utf-8') == 'load_model':
        # print('load_model')
        model_type = connection.recv(1024).decode('utf-8')
        print(model_type)
        if model_type == 'logreg':
            vect_path = os.path.join(MODEL_PATH, 'logreg_vektorizer.pkl')
            model_path = os.path.join(MODEL_PATH, 'logreg_model.onnx')
        elif model_type == 'catboost':
            vect_path = os.path.join(MODEL_PATH, 'catboost_vektorizer.pkl')
            model_path = os.path.join(MODEL_PATH, 'catboost_model.onnx')

        with open(vect_path, 'rb') as fd:
            determine_sentiment.set_vectorizer(pkl.load(fd))

        determine_sentiment.set_sess(model_path, model_type)

        return 0

    if command.decode('utf-8') == 'get_sentiment':
        # часто 2 отправки подряд объединяются в одну,
        # что бы разорвать эту последовательность, после получения данной 
        # команды, отправляю в ответ пакет бесполезных данных
        connection.send('unusefull data'.encode('utf-8'))

        # print('get sentiment')
        review_size = connection.recv(1024).decode('utf-8')
        connection.send('unusefull data'.encode('utf-8'))

        review = connection.recv(int(review_size)).decode('utf-8')
        print(review_size, review)
        # получаю тональность отзыва
        review = determine_sentiment.textprepare.clean_all(review)
        probb = determine_sentiment.make_prediction(review)

        if probb[0][0][0] > probb[0][0][1]:
            sentiment = 'negative'
        else:
            sentiment = 'positive'
        print(f'sentiment: {sentiment}')
        connection.send(sentiment.encode('utf-8'))

        return 0

    if command.decode('utf-8') == 'get_colored_sentiment':
        # часто 2 отправки подряд объединяются в одну,
        # что бы разорвать эту последовательность, после получения данной 
        # команды, отправляю в ответ пакет бесполезных данных
        connection.send('unusefull data'.encode('utf-8'))

        # print('get sentiment')
        review_size = connection.recv(1024).decode('utf-8')
        connection.send('unusefull data'.encode('utf-8'))

        review = connection.recv(int(review_size)).decode('utf-8')
        print(review_size, review)
        # получаю тональность отзыва
        review = determine_sentiment.textprepare.clean_all(review)
        probb = determine_sentiment.make_prediction(review)

        if probb[0][0][0] > probb[0][0][1]:
            sentiment = 'negative'
        else:
            sentiment = 'positive'
        print(f'sentiment: {sentiment}')
        connection.send(sentiment.encode('utf-8'))

        _ = connection.recv(1024)

        colored_review = 'her_tam_colored'
        connection.send(colored_review.encode('utf-8'))

        return 0


class DetermineSentimentClass:
    '''
    Класс для инференса работы с отзывами
    '''
    def __init__(self, inp_textprepare) -> None:
        self.__vectorizer = None
        self.__sess = None
        self.__input_name = None
        self.__label_name = None
        self.textprepare = inp_textprepare

    def set_vectorizer(self, inp_vectorizer) -> None:
        '''
        Задать векторайзер приватным аттрибутом класса
        args
            inp_vectorizer - входной векторайзер
        '''
        self.__vectorizer = inp_vectorizer

    def set_sess(self, inp_model_path: str, inp_type: str) -> None:
        '''
        Задать модель приватным аттрибутом класса, используемого для инеренса
        args
            inp_model_path: str - путь к модели
            inp_type: str - тип входной модели
        return
        '''
        sess = rt.InferenceSession(inp_model_path,
                                   providers=['CPUExecutionProvider']
                                   )
        self.__sess = sess

        if inp_type == 'logreg':
            self.__input_name = self.__sess.get_inputs()[0].name
            self.__label_name = self.__sess.get_outputs()[1].name
        else:  # catboost
            self.__input_name = 'features'
            self.__label_name = 'probabilities'

    def make_prediction(self, inp_text: str) -> float:
        '''
        Определить тональность
        args
            inp_text: str - текст отзыва
        return
            list[dict] - величины отнесения входного текста к классам 0 и 1
        '''
        # inp_text = self.textprepare.clean_all(inp_text)
        inp_text = self.__vectorizer.transform([inp_text]).toarray()
        probabilities = self.__sess.run(
                    [self.__label_name],
                    {self.__input_name: inp_text.astype(np.float32)}
        )
        return probabilities

    def __get_sentence(self, inp_text: str, inp_idxs: list[int]) -> float:
        '''
        args
            inp_text: str - входной текст
            inp_idxs: list[int] - список индексов слов, которые необходимо 
                    исключить из входного текста перед получением тональности
                    !!! важно: индексы в обратном порядке (n+1, n, n-1)
                    что бы не учитывать появившейся сдвиг в больших
                    индексах после удаления меньших.
        return
            str - выходной текст с подсвеченными словами
        '''
        for idx in inp_idxs:
            del inp_text[idx]

        inp_text = [el for el in inp_text if el != '_']
        inp_text = self.__vectorizer.transform([' '.join(inp_text)])

        return model.predict_proba(inp_text)[0][1]

    def get_colored_words(self, inp_text: str) -> np.ndarray:  # str:
        '''
        Получить исходный текст с подсвеченными словами, оказавшими наибольшее
        влияние на полученное предсказание. n-gram = (1, 1) by word
        args
            inp_text: str - входной текст
        return
            str - выходной текст с подсвеченными словами
        '''
        # предсказание на всем тексте отзыва, которое будет браться за основу
        full_text = self.text_prepare.clean_all(inp_text) 
        full_text = self.__vectorizer.transform([full_text])
        basis = model.predict_proba(full_text)[0][1]
        if basis > .5:
            print('positive')
            color = 'green'
            reverse_influence = False
        else:
            print('negative')
            color = 'red'
            reverse_influence = True

        # подгготовка массива 'влияния' слова
        text_split = self.text_prepare.clean_all(inp_text, for_coloring=True)\
                                      .split()
        # word_infl = np.ndarray((len(text_split), 4), dtype=np.float16)
        word_infl = np.full((len(text_split), 1), np.nan)
        words_infl = np.full((len(text_split), 3), np.nan)

        # убираю по слову и получаю результат на всем отзыве без этого слова
        print(text_split)
        for idx, el in enumerate(text_split):
            #print(el, idx)
            if el == '_':
                word_infl[idx, 0] == np.nan
                words_infl[idx, 0] == np.nan
                words_infl[idx, 1] == np.nan
                words_infl[idx, 2] == np.nan
                continue

            # влияние слова на тональность
            word_infl[idx, 0] = basis - self.__get_sentence(text_split.copy(),
                                                            [idx])

            # влияние слова и предыдущего слова на тональность
            if idx > 0 and text_split[idx-1] != '_':
                pred = self.__get_sentence(text_split.copy(), [idx, idx-1])
                words_infl[idx, 0] = basis - pred
            else:
                words_infl[idx, 0] = np.nan

            # влияние слова и следующего слова на тональность
            if idx < (len(text_split) - 2) and text_split[idx+1] != '_':
                pred = self.__get_sentence(text_split.copy(), [idx+1, idx])
                words_infl[idx, 1] = basis - pred
            else:
                words_infl[idx, 1] = np.nan

            # влияние слова, предыдущего и следующего слов на тональность
            if idx >= 1 and idx < (len(text_split) - 2)\
               and text_split[idx - 1] != '_'\
               and text_split[idx + 1] != '_':
                pred = self.__get_sentence(text_split.copy(),
                                           [idx+1, idx, idx-1])
                words_infl[idx, 2] = basis - pred
            else:
                words_infl[idx, 2] = np.nan

        words_infl = np.nansum(words_infl, axis=1)
        for idx in range(len(text_split)):
            if words_infl[idx] != 0:
                word_infl[idx] = word_infl[idx]*0.7 + words_infl[idx]*0.3

        #return word_infl, words_infl
        # добавляю в изначальный текст отзыва html тэги с цветом
        # на основные слова, повлиявшие больше всего на полученный результат
        word_w_influence = [el for el in zip(text_split, word_infl,
                                             range(len(word_infl)))
                            ]
        #print(word_w_influence)
        word_w_influence = sorted(word_w_influence,
                                  key=lambda x: x[1][0],
                                  reverse=reverse_influence,
                                  )
        important_indexes = set([el[2] for el in word_w_influence[:10]])
        vals = set([el[1][0] for el in word_w_influence[:10]])

        #print(word_w_influence)
        print(important_indexes)
        #print(vals)

        colored_text = ['']*len(inp_text.split())
        for idx, el in enumerate(inp_text.split()):
            if idx in important_indexes:
                colored_text[idx] = f'<span style="color: {color}">{el}</span>'
            else:
                colored_text[idx] = el

        return ' '.join(colored_text)


if __name__ == "__main__":

    print('starting')

    textprepare = TextPrepareClass(PHONES, stopwords.words('russian'))
    determine_sentiment = DetermineSentimentClass(textprepare)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f'Server started on {HOST}:{PORT}')

    exit = False
    while not exit:
        conn, addr = server_socket.accept()
        print(f'Connect by {addr}')

        while not exit:
            # клиент закрыл/оборвал соединение. возвращаюсь
            # к ожиданию нового подклюяения
            ret = work(conn, determine_sentiment)
            if ret == -1:
                print('Client close connection')
                print('Returning to waiting for a new connection')
                break

            # получена команда на завершению работы сервера
            if ret == -13:
                print('exiting')
                exit = True

            if ret == 0:
                pass

    server_socket.close()
    print('Docker endings')
