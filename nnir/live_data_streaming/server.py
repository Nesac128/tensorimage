import time
import json
from tornado import websocket, web, ioloop
from datetime import timedelta
from threading import Thread


_data = None
_time = None


def update_data(**data):
    global _data
    _data = data


def update_time(time):
    global _time
    _time = time


class LiveDataStreamingServer(Thread):
    def __init__(self, port):
        Thread.__init__(self)
        self.port = port

        web._port = self.port

    def start(self):
        app = web.Application([(r'/', WebSocketHandler)])
        app.listen(self.port)
        ioloop.IOLoop.instance().start()

    def main(self):
        threaded_start = Thread(target=self.start)
        threaded_start.start()


class WebSocketHandler(websocket.WebSocketHandler, LiveDataStreamingServer):
    def open(self):
        print('Connection has been established!')
        ioloop.IOLoop.instance().add_timeout(
            timedelta(seconds=0),
            self.send_data)

    def send_data(self):
        self.write_message(bytes(str(_data), "utf-8"))
        ioloop.IOLoop.instance().add_timeout(timedelta(seconds=_time), self.send_data)

