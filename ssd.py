from multiprocessing import Process
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

class Model(Process):
    def __init__(self, is_gpu=True):
        Process.__init__(self)

        self.is_gpu = is_gpu

        ### PlaidML 사용
        if self.is_gpu:
            print("Using PlaidML backend.")
            import plaidml.keras
            plaidml.keras.install_backend()

        plt.rcParams['figure.figsize'] = (8, 8)
        plt.rcParams['image.interpolation'] = 'nearest'

        np.set_printoptions(suppress=True)

        self.voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                    'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                    'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                    'Sheep', 'Sofa', 'Train', 'Tvmonitor']

        self.coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        NUM_CLASSES = len(self.voc_classes) + 1

        # from model.ssd300VGG16_V1 import SSD
        from model.ssd300VGG16_V2 import SSD
        from model.ssd_utils import BBoxUtility

        input_shape = (300, 300, 3)
        
        self.Model = SSD(input_shape, n_class=NUM_CLASSES)
        # self.Model.load_weights('weights/VGG16SSD300weights_voc_2007_class20.hdf5', by_name=True) # This is weights for model version 1
        self.Model.load_weights('weights/VGG_VOC0712_SSD_300x300_iter_120000.h5', by_name=True)
        self.bbox_util = BBoxUtility(NUM_CLASSES)

    def inference(self, frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        data = np.reshape(np.array(img), [1, img.shape[0], img.shape[1], img.shape[2]])

        start_time = time.time()
        preds = self.Model.predict(data, batch_size=1, verbose=1)
        end_time = time.time()

        result = self.bbox_util.detection_out(preds)

        # Parse the outputs.
        det_label = result[0][:, 0]
        det_conf = result[0][:, 1]
        det_xmin = result[0][:, 2]
        det_ymin = result[0][:, 3]
        det_xmax = result[0][:, 4]
        det_ymax = result[0][:, 5]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * frame.shape[1]))
            ymin = int(round(top_ymin[i] * frame.shape[0]))
            xmax = int(round(top_xmax[i] * frame.shape[1]))
            ymax = int(round(top_ymax[i] * frame.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = self.voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.putText(frame, display_txt, (xmin, ymin-5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

        return frame, 1.0 / (end_time - start_time)
