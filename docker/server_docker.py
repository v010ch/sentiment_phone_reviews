'''server docker part for phone review sentiment'''

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
          'ноки|хуаве|мейзу|асус|галакси|иной|гэлакси|хонор)(|[a-zа-я]+)\b'

nltk.data.path.append('/usr/src/app/nltk')


def recv_data(inp_socket):
    '''
    '''
    inp_socket.settimeout(0)
    package_len = int(inp_socket.recv(5))
    if package_len > 1024:
        return 0

    inp_socket.settimeout(10)
    ret_data = inp_socket.recv(package_len)

    return ret_data


def send_data(inp_socket, inp_data):
    '''
    '''
    package_len = str(len(inp_data)).encode()
    inp_socket.send(package_len)

    time.sleep(1)
    inp_socket.send(inp_data.encode())

    return 0


def work(connection, determine_sentiment) -> int:
    '''
    '''
    command = connection.recv(1024)
    if not command:
        return -1

    if command.decode('utf-8') == 'exit':
        return -13

    print('recived command: ', command.decode('utf-8'))

    # rudiment
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
        # often two send in a row work like one send/string (from client)
        # should devide them with send
        # it this case unusefull data send
        connection.send('unusefull data'.encode('utf-8'))

        # print('get sentiment')
        review = connection.recv(1024).decode('utf-8')
        print(review)
        # get model prediction
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
        # print('get get_colored_sentiment')
        review = connection.recv(1024).decode('utf-8')
        print(review)
        # get model prediction
        sentiment = 'neutral'
        connection.send(sentiment.encode('utf-8'))
        return 0


class DetermineSentimentClass:
    '''
    '''
    def __init__(self, inp_textprepare) -> None:
        self.__vectorizer = None
        self.__sess = None
        self.__input_name = None
        self.__label_name = None
        self.textprepare = inp_textprepare

    def set_vectorizer(self, inp_vectorizer) -> None:
        '''
        Задать внутренний векторайзер
        '''
        self.__vectorizer = inp_vectorizer

    def set_sess(self, inp_model_path: str, inp_type: str) -> None:
        '''
        Задать внутреннюю модель
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
        '''
        # inp_text = self.textprepare.clean_all(inp_text)
        inp_text = self.__vectorizer.transform([inp_text]).toarray()
        probabilities = self.__sess.run(
                    [self.__label_name],
                    {self.__input_name: inp_text.astype(np.float32)}
        )
        return probabilities


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
            # client closed. awaiting new connection
            ret = work(conn, determine_sentiment)
            if ret == -1:
                print('Client close connection')
                print('Returning to waiting for a new connection')
                break

            # command to end execution and turn off docker
            if ret == -13:
                print('exiting')
                exit = True

            if ret == 0:
                pass

    server_socket.close()
    print('Docker endings')
