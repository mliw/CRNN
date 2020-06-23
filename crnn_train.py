import tensorflow as tf
import os
import numpy as np
from crnn_tools.data_loader import DataGenerator
from crnn_tools.data_loader import libs
from crnn_tools import data_loader
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")

    
if __name__ == '__main__':
    
    # 0 Load pre-train weight
    from crnn_tools.core import CRNN
    train_model = CRNN()   
    train_model._load_weights("weights/rd_5.h5")
    
    
    # 2 Begin training
    for i in range(10):
        dll = DataGenerator(data_loader.TRAIN_LIST,5000) 
        it_dll = iter(dll)
        j = i+6
        lr = (1e-3)*(0.995)**j
        sgd = SGD(lr=lr)
        train_model._compile_net(sgd)
        
        saved_count = 0
        for epoch_iter in range(dll.__len__()):
            print(epoch_iter/dll.__len__())
            tem_data = next(it_dll)
            his = train_model.train_core.fit(x=tem_data[0],y=tem_data[1],epochs=1,callbacks = [tf.keras.callbacks.History()],batch_size=16)
            del(tem_data)
            saved_count+=1
            if saved_count==50:
                 train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
                 saved_count = 0
    
        train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
            
        
        

