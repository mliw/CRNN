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
    y_pred = y_pred[:, 2:, :]
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
        # Input
        labels = tf.keras.Input(name='the_labels',shape=[config.LABEL_LENGTH], dtype='float32')
        input_length = tf.keras.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.Input(name='label_length', shape=[1], dtype='int64')
        vgg_input = tf.keras.Input(shape=(32,None,1),name='vgg_input')
        
        # None stands for batch_size
        # Vgg filters=64 input_shape (None,32,280,1) output_shape (None,16,140,64)
        l0 = layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",name="conv1")(vgg_input)
        l0 = layers.BatchNormalization()(l0)
        l0 = layers.Activation('relu')(l0)
        l0 = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(l0)  

        # Vgg filters=128 input_shape (None,16,140,64) output_shape (None,8,70,128)
        l1 = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(l0) 
        l1 = layers.BatchNormalization()(l1)
        l1 = layers.Activation('relu')(l1)
        l1 = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(l1)    
        
        # Vgg filters=256 input_shape (None,8,70,128) output_shape (None,4,70,256)
        l2 = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(l1)  
        l2 = layers.BatchNormalization()(l2)
        l2 = layers.Activation('relu')(l2)
        l2 = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(l2)  
        l2 = layers.BatchNormalization()(l2)
        l2 = layers.Activation('relu')(l2)
        l2 = layers.MaxPooling2D(pool_size=(2,1), name='max3')(l2) 
        
        # Vgg filters=512 input_shape (None,4,70,256) output_shape (None,2,70,512)
        l3 = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(l2)
        l3 = layers.BatchNormalization()(l3)
        l3 = layers.Activation('relu')(l3)
        l3 = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(l3) 
        l3 = layers.BatchNormalization()(l3)
        l3 = layers.Activation('relu')(l3)
        l3 = layers.MaxPooling2D(pool_size=(2, 1), name='max4')(l3) 

        # Vgg filters=512 input_shape (None,2,70,512) output_shape (None,2,70,512)
        l4 = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(l3)  
        l4 = layers.BatchNormalization()(l4)
        l4 = layers.Activation('relu')(l4)
        
        # Transform from CNN to RNN input_shape (None,2,70,512) output_shape (None,70,64) 
        l5 = layers.Permute((2, 1, 3), name='permute')(l4)
        l5 = layers.TimeDistributed(layers.Flatten(), name='timedistrib')(l5)
        l5 = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(l5)    

        # RNN layer input_shape (None,70,64) output_shape (None,70,512)
        l6 = layers.Bidirectional(layers.GRU(256, return_sequences=True), name='blstm1')(l5)
        l6 = layers.BatchNormalization()(l6)
        l6 = layers.Bidirectional(layers.GRU(256, return_sequences=True), name='blstm2')(l6)
        l6 = layers.BatchNormalization()(l6) 
        
        # Get prediction input_shape (None,70,512) output_shape (None,70,5991)
        l7 = layers.Dense(CLASS_NUM+1, name='blstm2_out', activation='softmax')(l6)
        self.predict_core = Model(vgg_input,l7)  

        # Get loss and train_core
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, l7, input_length, label_length]) #(None, 1)
        self.train_core = Model([vgg_input, labels, input_length, label_length], loss_out)

    
    def _compile_net(self,opt):
        self.train_core.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=opt)
        
        
    def predict(self,test_data):
        print("Please note! The first 2 elements of RNN are deleted!")
        pre = self.predict_core.predict(test_data)
        seq_length = pre.shape[1]
        batch_size = pre.shape[0]
        sequence_length = np.ones(batch_size)*seq_length
        sequence_length = sequence_length.astype(np.int32)
        result = K.ctc_decode(pre, sequence_length, greedy=True)        
        return result[0][0].numpy(),pre
    
    
    def predict_and_translate(self,test_data):
        pre = self.predict_core.predict(test_data)
        seq_length = pre.shape[1]
        batch_size = pre.shape[0]
        sequence_length = np.ones(batch_size)*seq_length
        sequence_length = sequence_length.astype(np.int32)
        result = K.ctc_decode(pre, sequence_length, greedy=True)        
        return translate(result[0][0].numpy())   


    def _load_weights(self,path):
        self.train_core.load_weights(path)
        


