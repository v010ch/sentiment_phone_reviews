''' flask main app for phone review sentiment analysis'''
import socket
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for


HOST = '172.17.0.2'
PORT = 9126


app = Flask(__name__)

socket_cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_cli.connect((HOST, PORT))


def send_data(inp_msg: str):
    '''
    '''
    socket_cli.send(inp_msg.encode('utf-8'))


def get_data() -> str:
    '''
    '''
    data = socket_cli.recv(1024)
    if not data:
        raise 'Out of data from server'

    return data.decode('utf-8')


@app.route("/", methods=["POST", "GET"])
def index_page():
    '''
    '''
    # rudiment
    # chech which model can we use
    print(f'index_page: {request.method}')
    if request.method == 'GET':
        command = 'get_available_models'
        socket_cli.send(command.encode('utf-8'))

        available_models = get_data()
        print(available_models)

    # tell to server to load the selected model
    if request.method == "POST":
        print(request.form['model_type'])
        command = 'load_model'
        socket_cli.send(command.encode('utf-8'))

        model = request.form['model_type']
        socket_cli.send(model.encode('utf-8'))

        return redirect(url_for('review_page'))

    # no model choosed. check can we use medium and large models
    if available_models == 'all':
        return render_template('index.html',)
    else:
        return render_template('index_noroberta.html',)


@app.route("/review", methods=["POST", "GET"])
def review_page(text: Optional[str] = '', 
                prediction_message: Optional[str] = ''
                ):
    '''
    '''
    print(f'review_page: {request.method}')
    if request.method == "GET":
        if 'model type' in request.form.keys():
            print(request.form['model_type'])

        return render_template('input_review.html',  # text=text,
                               # model_type=model_type,
                               # prediction_message=prediction_message,
                               )

    if request.method == "POST":
        if 'text' in request.form.keys():
            review = request.form['text']
            print(review)

            command = 'get_sentiment'
            socket_cli.send(command.encode('utf-8'))

            # often two send in a row work like one send
            # should devide them with recv
            # getting unusefull data
            _ = get_data()

            model = review
            socket_cli.send(model.encode('utf-8'))

            sentiment = get_data()
        else:
            print(request.form)

        return render_template('input_review.html', text=review,
                               # model_type=model_type,
                               prediction_message=sentiment,
                               )


if __name__ == "__main__":
    # app.run(port=int(os.getenv('PORT', 31031)))
    app.run()
