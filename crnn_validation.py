import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")

    
if __name__ == '__main__':

    # 0 Prepare test_data for validation 
    test_names = os.listdir(data_loader.TEST_LIST[0])
    test_iter = iter(DataGenerator(data_loader.TEST_LIST,5000))
    test_data = next(test_iter)
    
    
    # 1 Load pre-train weight
    from crnn_tools.core import CRNN
    
    
    train_model = CRNN()   
    train_model._load_weights("weights/rd_5.h5")
    
    
    path = "asset/english.jpg"
    img_gray = load_picture(path)
    train_model.predict_and_translate(img_gray)
    
    
    
    img_gray = load_picture(path)
     
    train_model.predict_core.predict(img_gray)
    
    
    # 2 Begin training
    for i in range(10):
        dll = DataGenerator(data_loader.TRAIN_LIST,5000) 
        it_dll = iter(dll)
        j = i+3
        lr = (1e-3)*(0.995)**j
        sgd = SGD(lr=lr)
        train_model._compile_net(sgd)
        
        for epoch_iter in range(dll.__len__()):
            print(epoch_iter/dll.__len__())
            tem_data = next(it_dll)
            his = train_model.train_core.fit(x=tem_data[0],y=tem_data[1],epochs=1,callbacks = [tf.keras.callbacks.History()],batch_size=16)
            del(tem_data)
            
        train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
            
        
        

