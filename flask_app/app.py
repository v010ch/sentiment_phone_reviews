'''flask main app для анализа тональности отзывов на телефоны'''
import socket
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for
# from markupsafe import Markup


HOST = '172.17.0.2'
PORT = 9126


app = Flask(__name__)

socket_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_cli.connect((HOST, PORT))


def send_data(inp_msg: str):
    '''
    рудимент. не используется.
    args:
         inp_msg - текст для отправки на сервер
    '''
    socket_cli.send(inp_msg.encode('utf-8'))


def get_data(inp_size: Optional[int] = 1024) -> str:
    '''
    Получения данных с сервера
    args:
        inp_size: int - размер требуемых данных. опционально (1024)
    returns:
        str - полученные данные
    '''
    data = socket_cli.recv(inp_size)
    if not data:
        raise 'Out of data from server'

    return data.decode('utf-8')


def get_sentiment(inp_review: str) -> str:
    '''
    Получение тональности отзыва
    args:
        inp_review: str - отзыв, тональность которого необходимо получить
    return:
        str: тональность
    '''
    command = 'get_sentiment'
    socket_cli.send(command.encode('utf-8'))

    # часто 2 отправки подряд объединяются в одну,
    # что бы разорвать эту последовательность, после отправки данной
    # команды, получаю в ответ пакет бесполезных данных
    _ = get_data()

    review_size = str(len(inp_review.encode('utf-8')))
    socket_cli.send(review_size.encode('utf-8'))
    _ = get_data()

    socket_cli.send(inp_review.encode('utf-8'))

    sentiment = get_data()

    return sentiment


def get_colored_sentiment(inp_review: str) -> tuple[str, str]:
    '''
    Получение тональности отзыва с подсвеченными словами, оказавшими
    наибольшее влияние на полученную тональность
    args:
        inp_review: str - отзыв, тональность которого необходимо получить
    return:
        str - тональность
        str - текст отзыва с подсвеченными словами...
    '''
    command = 'get_colored_sentiment'
    socket_cli.send(command.encode('utf-8'))

    # часто 2 отправки подряд объединяются в одну,
    # что бы разорвать эту последовательность, после отправки данной
    # команды, получаю в ответ пакет бесполезных данных
    _ = get_data()

    review_size = str(len(inp_review.encode('utf-8')))
    socket_cli.send(review_size.encode('utf-8'))
    _ = get_data()

    socket_cli.send(inp_review.encode('utf-8'))

    sentiment = get_data()
    socket_cli.send('unusefull_data'.encode('utf-8'))
    # !!!!!  Размер раскрашенного обзыва
    colored_review = get_data()

    print(f'get colored review >{colored_review}<')

    return sentiment, colored_review


@app.route('/', methods=['POST', 'GET'])
def index_page():
    '''
    Метод работы с главной страницей
    returns:
        сгенерированная страница, либо пренаправление на страницу review_page
    '''
    # рудимент
    print(f'index_page: {request.method}')
    if request.method == 'GET':
        command = 'get_available_models'
        socket_cli.send(command.encode('utf-8'))

        available_models = get_data()
        print(available_models)

    # указать серверу работать с выбранной моделью
    if request.method == 'POST':
        print(request.form['model_type'])
        command = 'load_model'
        socket_cli.send(command.encode('utf-8'))

        model = request.form['model_type']
        socket_cli.send(model.encode('utf-8'))

        return redirect(url_for('review_page'))

    # рудимент
    if available_models == 'all':
        return render_template('index.html',)
    else:
        return render_template('index_noroberta.html',)


@app.route('/review', methods=['POST', 'GET'])
def review_page(text: Optional[str] = '',
                prediction_message: Optional[str] = '',
                colored_text: Optional[str] = '',
                ):
    '''
    Страница для ввода и отображения отзывов и танальности,
    взаимодействиями с пользователями и сервером.
    args:
        text: str - текст входного отзыва, опционально
        prediction_message: str - предсказанная тональность, опционально
        colored_text: str - текст входного отзыва с подсвеченными важными для
                      предсказанной тональности словами
    returns:
        страница взаимодействия с отзывами, пользователями и сервером
    '''
    print(f'review_page: {request.method}')
    if request.method == 'GET':
        if 'model type' in request.form.keys():
            print(request.form['model_type'])

        return render_template('input_review.html',)

    if request.method == 'POST':
        if 'text' in request.form.keys():
            review = request.form['text']

            sentiment = get_sentiment(review)
            review_colored = ''

            # sentiment, review_colored = get_colored_sentiment(review)
        else:
            print(request.form)

        return render_template('input_review.html', text=review,
                               prediction_message=sentiment,
                               colored_text=review_colored,
                               )


if __name__ == '__main__':
    app.run()
