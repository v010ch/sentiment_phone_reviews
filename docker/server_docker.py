''' tmp '''

import socket
import time

HOST = '172.17.0.2'
PORT = 9126


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


def work(connection) -> int:
    '''
    '''
    command = connection.recv(1024)
    if not command:
        return -1

    if command.decode('utf-8') == 'exit':
        return -13

    print('getted command: ', command.decode('utf-8'))

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
        # load model
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
        sentiment = 'neutral'
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


if __name__ == "__main__":

    print('starting')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(type(server_socket))
    print(f'Server started on {HOST}:{PORT}')

    exit = False
    while not exit:
        conn, addr = server_socket.accept()
        print(f'Connect by {addr}')

        while not exit:
            # client closed. awaiting new connection
            ret = work(conn)
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
