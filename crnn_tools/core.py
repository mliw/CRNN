import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tensorflow.keras import backend as K
from crnn_tools import config
from crnn_tools import libs
import warnings
warnings.filterwarnings("ignore")
CLASS_NUM = 5990


def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def cut_top_2(y_pred):
    return  y_pred[:, 2:, :]


def my_custom_loss(y_true, y_pred):
    return K.mean(y_pred)


def translate(pre):
    pre[pre==-1] = 0
    tem_list = libs.total_list.copy()
    tem_list[0] = ""
    result = [''.join(tem_list[items]) for items in pre]
    return result

    
class CRNN:

    def __init__(self,):
        self._build_crnn()
        
    def _build_crnn(self,):
        labels = tf.keras.Input(name='the_labels',shape=[config.LABEL_LENGTH], dtype='float32')
        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int64')
        
        vgg_input = tf.keras.Input(shape=(32,None,1),name='vgg_input')
        l0 = layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu",name="l0")(vgg_input)
        l1 = layers.MaxPool2D(pool_size=(2, 2),strides=2)(l0)
        l2 = layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu")(l1)
        l3 = layers.MaxPool2D(pool_size=(2, 2),strides=2)(l2)
        l4 = layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu")(l3)
        l5 = layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu")(l4)
        l6 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(l5)
        l7 = layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu")(l6)
        l8 = layers.BatchNormalization()(l7)
        l9 = layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding="same",activation="relu")(l8)
        l10 = layers.BatchNormalization()(l9)       
        l11 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(l10)
        l12 = layers.Conv2D(filters=512,kernel_size=(2,2),strides=(1,1),padding="valid",activation="relu")(l11)           
        l13 = layers.Permute((2, 1, 3), name='permute')(l12)
        l14 = layers.TimeDistributed(layers.Flatten(), name='timedistrib')(l13)
        l15 = layers.Bidirectional(layers.GRU(256, return_sequences=True), name='blstm1')(l14)
        l16 = layers.Dense(256, name='blstm1_out', activation='linear')(l15)
        l17 = layers.Bidirectional(layers.GRU(256, return_sequences=True), name='blstm2')(l16)
        l18 = layers.Dense(CLASS_NUM+1, name='blstm2_out', activation='softmax')(l17)
        l18 = layers.Lambda(cut_top_2)(l18)
        self.base_core = Model(vgg_input,l18)
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, l18, input_length, label_length])
        self.core = Model(inputs=[vgg_input, labels, input_length, label_length], outputs=[loss_out])
    
    def _compile_net(self,opt):
        self.core.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=opt)
        
    def predict(self,test_data):
        print("Please note! The first 2 elements of RNN are deleted!")
        pre = self.base_core.predict(test_data)
        seq_length = pre.shape[1]
        batch_size = pre.shape[0]
        sequence_length = np.ones(batch_size)*seq_length
        sequence_length = sequence_length.astype(np.int32)
        result = K.ctc_decode(pre, sequence_length, greedy=True)        
        return result[0][0].numpy(),pre
    
    def predict_and_translate(self,test_data):
        pre = self.base_core.predict(test_data)
        seq_length = pre.shape[1]
        batch_size = pre.shape[0]
        sequence_length = np.ones(batch_size)*seq_length
        sequence_length = sequence_length.astype(np.int32)
        result = K.ctc_decode(pre, sequence_length, greedy=True)        
        return translate(result[0][0].numpy())   

    def _load_weights(self,path):
        self.core.load_weights(path)
        


