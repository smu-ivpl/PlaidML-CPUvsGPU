"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model

from model.ssd_layers import Normalize
from model.ssd_layers import PriorBox


def SSD(input_shape, n_class=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    # Input & Variables
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    input0 = input_tensor
    n_boxes = [4, 6, 6, 6, 4, 4]

    # Block 1
    conv1_1 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_1')(input0)
    conv1_2 = Conv2D(64, (3, 3),activation='relu',padding='same',name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),activation='relu',padding='same',name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3),activation='relu',padding='same',name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),activation='relu',padding='same',name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),activation='relu',padding='same',name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding='same',name='pool5')(conv5_3)

    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6),activation='relu', padding='same',name='fc6')(pool5)

    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu',padding='same', name='fc7')(fc6)

    # Block 6
    conv6_1 = Conv2D(256, (1, 1), activation='relu',padding='same',name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2),activation='relu', padding='valid', name='conv6_2')(conv6_1)

    # Block 7
    conv7_1 = Conv2D(128, (1, 1), activation='relu',padding='same',name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2),activation='relu', padding='valid',name='conv7_2')(conv7_1)

    # Block 8
    conv8_1 = Conv2D(128, (1, 1), activation='relu',padding='same',name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1),activation='relu', padding='valid',name='conv8_2')(conv8_1)

    # Block 9
    conv9_1 = Conv2D(128, (1, 1), activation='relu',padding='same',name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1),activation='relu', padding='valid',name='conv9_2')(conv9_1)

    # Normalize
    conv4_3_norm = Normalize(n_class-1, name='conv4_3_norm')(conv4_3)

    # Prediction from conv4_3
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same',name="conv4_3_norm_mbox_loc")(conv4_3_norm)
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_class, (3, 3), padding='same',name="conv4_3_norm_mbox_conf")(conv4_3_norm)
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_class), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    conv4_3_norm_mbox_priorbox = PriorBox(img_size, min_size=30.0, max_size=60.0, aspect_ratios=[1.0, 2.0, 0.5], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    
    # Prediction from fc7
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3),padding='same',name="fc7_mbox_loc")(fc7)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_class, (3, 3),padding='same',name="fc7_mbox_conf")(fc7)
    fc7_mbox_conf_reshape = Reshape((-1, n_class), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    fc7_mbox_priorbox = PriorBox(img_size, min_size=60.0, max_size=114.0, aspect_ratios=[1.0, 2.0, 0.5, 3.0, 1.0/3.0], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')(fc7)

    # Prediction from conv6_2
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same',name="conv6_2_mbox_loc")(conv6_2)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_class, (3, 3), padding='same',name="conv6_2_mbox_conf")(conv6_2)
    conv6_2_mbox_conf_reshape =  Reshape((-1, n_class), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv6_2_mbox_priorbox = PriorBox(img_size, min_size=114.0, max_size=168.0, aspect_ratios=[1.0, 2.0, 0.5, 3.0, 1.0/3.0], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')(conv6_2)

    # Prediction from conv7_2
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',name="conv7_2_mbox_loc")(conv7_2)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_class, (3, 3), padding='same',name="conv7_2_mbox_conf")(conv7_2)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_class), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv7_2_mbox_priorbox = PriorBox(img_size, min_size=168.0, max_size=222.0, aspect_ratios=[1.0, 2.0, 0.5, 3.0, 1.0/3.0], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')(conv7_2)
    
    # Prediction from conv8_2
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same',name="conv8_2_mbox_loc")(conv8_2)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_class, (3, 3), padding='same',name="conv8_2_mbox_conf")(conv8_2)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_class), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv8_2_mbox_priorbox = PriorBox(img_size, min_size=222.0, max_size=276.0, aspect_ratios=[1.0, 2.0, 0.5], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')(conv8_2)

    # Prediction from conv9_2
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same',name="conv9_2_mbox_loc")(conv9_2)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_class, (3, 3), padding='same',name="conv9_2_mbox_conf")(conv9_2)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_class), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    conv9_2_mbox_priorbox = PriorBox(img_size, min_size=276.0, max_size=330.0, aspect_ratios=[1.0, 2.0, 0.5], variances=[0.1, 0.1, 0.2, 0.2], name='conv9_2_mbox_priorbox')(conv9_2)

    # Reshaping of priorbox
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)
    
    # Gather all predictions
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_reshape,
                            fc7_mbox_loc_reshape,
                            conv6_2_mbox_loc_reshape,
                            conv7_2_mbox_loc_reshape,
                            conv8_2_mbox_loc_reshape,
                            conv9_2_mbox_loc_reshape],
                           axis=1,
                           name='mbox_loc')
    
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_reshape,
                             fc7_mbox_conf_reshape,
                             conv6_2_mbox_conf_reshape,
                             conv7_2_mbox_conf_reshape,
                             conv8_2_mbox_conf_reshape,
                             conv9_2_mbox_conf_reshape],
                            axis=1,
                            name='mbox_conf')

    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox_reshape,
                                 fc7_mbox_priorbox_reshape,
                                 conv6_2_mbox_priorbox_reshape,
                                 conv7_2_mbox_priorbox_reshape,
                                 conv8_2_mbox_priorbox_reshape,
                                 conv9_2_mbox_priorbox_reshape],
                                axis=1,
                                name='mbox_priorbox')
    
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    predictions = concatenate([mbox_loc, mbox_conf_softmax, mbox_priorbox],axis=2,name='predictions')
    model = Model(input0, predictions)
    return model
