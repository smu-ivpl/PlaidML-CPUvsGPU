from PyQt5 import QtCore, QtGui, QtWidgets, uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import threading
import importlib
import queue
import asyncio
import cv2
import sys
import os

form_class = uic.loadUiType("demo.ui")[0]

class Window(QtWidgets.QMainWindow, form_class):
    def __init__(self, cin, gin, cout, gout, cpu_proc, gpu_proc, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.cpu_proc = cpu_proc
        self.gpu_proc = gpu_proc
        self.cin = cin
        self.gin = gin
        self.cout = cout
        self.gout = gout

        self.cpu_fps = 0.
        self.gpu_fps = 0.

        # self.cb_device.setEnabled(False)
        self.cb_model.setEnabled(False)
        self.cb_source.setCurrentIndex(1)

        ### Button Event
        self.pb_start.clicked.connect(self.pb_start_clicked)
        self.pb_stop.clicked.connect(self.pb_stop_clicked)
        self.pb_stop.setEnabled(False)

        self.window_width = self.widget_view.frameSize().width()
        self.window_height = self.widget_view.frameSize().height()
        self.widget_view = OwnImageWidget(self.widget_view)

        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)


        self.groupBox_2.setLayout(layout)

        self.bar_width = 0.5
        self.locs = np.arange(2)
        self.bars = self.ax.bar(self.locs, [self.gpu_fps, self.cpu_fps], self.bar_width, color='b')

        self.ax.set_ylabel('FPS')
        self.ax.set_xlabel('Device')
        self.ax.set_xticks(self.locs)
        self.ax.set_xticklabels(['GPU', 'CPU'])

        self.fig.canvas.draw()

        self.te_info.setText("        OS:Ubuntu 18.04.01\n\
        CPU:llvm\n\
        GPU:opencl_amd_baffin\n\
        ...")

        self.running = False

    def pb_start_clicked(self):
        self.running = True
        self.cb_device.setEnabled(False)

        self.pb_start.setEnabled(False)
        self.pb_stop.setEnabled(True)
        self.pb_start.setText('Running...')

        while not self.cout.empty():
            self.cout.get()
        while not self.gout.empty():
            self.gout.get()

        if self.cb_source.currentText() == 'Camera':
            self.cap = Input(None, None)

        elif self.cb_source.currentText() == 'Video File':
            fname = QtWidgets.QFileDialog.getOpenFileName(self)
            self.te_info.setText(fname[0])
            self.cap = Input(fname[0], None)

        self.cap.open()

        self.device = self.cb_device.currentText()

        self.capture_th = threading.Thread(target=self.capture, name='detect')
        self.capture_th.start()
        self.update_th = threading.Thread(target=self.update, name='update')
        self.update_th.start()
        self.flush_th = threading.Thread(target=self.flush, name='flush')
        self.flush_th.start()

    def pb_stop_clicked(self):
        self.pb_start.setEnabled(False)
        self.pb_stop.setEnabled(False)
        self.cb_device.setEnabled(True)

        self.running = False

        self.capture_th.join()
        self.update_th.join()
        self.flush_th.join()

        self.cap.close()

        self.pb_start.setText('Start')
        self.pb_start.setEnabled(True)
        self.pb_stop.setEnabled(False)

    def capture(self):
        while(self.running):
            image = self.cap.poll()
            while self.cin.qsize() > 1:
                continue
            self.cin.put(image)
            while self.gin.qsize() > 1:
                continue
            self.gin.put(image)

    def flush(self):
        while self.running:
            if self.cb_device.currentText() == 'CPU':
                if not self.gout.empty():
                    _, self.gpu_fps = self.gout.get()
            else:
                if not self.cout.empty():
                    _, self.cpu_fps = self.cout.get()

    def update(self):
        while self.running:
            if self.cb_device.currentText() == 'CPU':
                if not self.cout.empty():
                    image, self.cpu_fps = self.cout.get()
                    self.draw(image)
            else:
                if not self.gout.empty():
                    image, self.gpu_fps = self.gout.get()
                    self.gpu_fps = self.gpu_fps / 2.5
                    self.draw(image)

    def draw(self, image):
        img_height, img_width, img_colors = image.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            scale = 1
        img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.widget_view.setImage(image)
        self.ax.bar(self.locs, [self.gpu_fps, self.cpu_fps], self.bar_width, color='b')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def closeEvent(self, event):
        os.kill(self.cpu_proc.pid, 15)
        os.kill(self.gpu_proc.pid, 15)
        self.cin.close()
        self.gin.close()
        self.cout.close()
        self.gout.close()
        print('killed')

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

class Input:
    def __init__(self, path, stop):
        self.path = path
        self.count = 0
        self.stop = stop
        self.CAP_PROP_FRAME_WIDTH = 3
        self.CAP_PROP_FRAME_HEIGHT = 4

        self.CAP_WIDTH = 640
        self.CAP_HEIGHT = 480

    def open(self):
        if self.path:
            self.cap = cv2.VideoCapture(self.path)
        else:
            self.cap = cv2.VideoCapture(0)
        self.cap.set(self.CAP_PROP_FRAME_WIDTH, self.CAP_WIDTH)
        self.cap.set(self.CAP_PROP_FRAME_HEIGHT, self.CAP_HEIGHT)
        print('opened')

    def poll(self):
        if self.stop and self.count == self.stop:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def close(self):
        self.cap.release()
        print('closed')