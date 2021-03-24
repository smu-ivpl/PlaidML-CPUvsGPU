from PyQt5 import QtWidgets
from window import Window
from multiprocessing import Process, Queue
import multiprocessing as mp
import tensorflow as tf
import sys
import os

def inference(is_gpu, input_q, output_q):
    pid = os.getpid()
    with tf.Graph().as_default():
        with tf.Session().as_default():
            if is_gpu:
                print('GPU(PlaidML)-Process ID: %d' % (pid))
                import ssd
                model = ssd.Model(True)
            else :
                print('CPU(Tensorflow)-Process ID: %d' % (pid))
                import ssd
                model = ssd.Model(False)

            while True:
                if not input_q.empty():
                    image = input_q.get()
                    result, time = model.inference(image)

                    if output_q.qsize() <= 10:
                        output_q.put((result, time))
                    else:
                        continue

def main():
    try:
        ### Process
        cin = Queue()
        cout = Queue()
        gin = Queue()
        gout = Queue()
        
        gpu_proc = Process(target=inference, args=(True, gin, gout))
        cpu_proc = Process(target=inference, args=(False, cin, cout))

        gpu_proc.start()
        cpu_proc.start()

        app = QtWidgets.QApplication(sys.argv)
        DemoGUI = Window(cin, gin, cout, gout, cpu_proc, gpu_proc, None)
        DemoGUI.setWindowTitle('Demo')
        DemoGUI.show()
        app.exec_()
    
    finally:
        os.kill(gpu_proc.pid, 15)
        os.kill(cpu_proc.pid, 15)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()