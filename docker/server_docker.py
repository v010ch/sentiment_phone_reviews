''' tmp '''

import socket
import time


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


if __name__ == "__main__":

    print('starting')
    HOST = 'localhost'
    PORT = 30200

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    print('awaiting input request')
    s.listen()
    server_socket, addr = s.accept()
    print('connected')

    data = recv_data(server_socket)
    print(data)

    server_socket.close()
    print('thats all kids')
