import tensorflow as tf
import numpy as np
from crnn_tools.data_loader import DataGenerator
from crnn_tools import data_loader
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings("ignore")
gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == '__main__':
    
    # 0 Load pre-train weight
    from crnn_tools.core import CRNN
    train_model = CRNN() 
    #train_model._load_weights("weights/rd_2.h5")
    
   
    # 1 Begin training    
    for i in range(10):
        target_list = data_loader.TRAIN_LIST.copy()
        np.random.shuffle(target_list)
        targets = [target_list[index*2:(index+1)*2] for index in range(50)]
        
        j = i+2
        lr = (1e-4)*(0.995)**j
        sgd = SGD(lr=lr)
        train_model._compile_net(sgd)
        dlls = [DataGenerator(target,64) for target in targets]
        
        current_num = 0
        for dll in dlls:
            print(current_num)
            current_num+=1
            length = dll.__len__()
            it_dll = iter(dll)
            his = train_model.train_core.fit(it_dll,epochs=1,callbacks = [tf.keras.callbacks.History()],steps_per_epoch=length)                
            train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
                
        train_model.train_core.save_weights("weights/rd_"+str(j)+".h5") 
    
    
# Train without generator    
"""    
    # 1 Begin training
    for i in range(10):
        j = i+2
        lr = (1e-3)*(0.995)**j
        sgd = SGD(lr=lr)
        train_model._compile_net(sgd)
        dll = DataGenerator(data_loader.TRAIN_LIST,6400)
        it_dll = iter(dll)
        saved = 0
        deleted = 0
        for epoch_iter in range(dll.__len__()):
            print(epoch_iter/dll.__len__())
            tem_data = next(it_dll)
            his = train_model.train_core.fit(x=tem_data[0],y=tem_data[1],epochs=1,callbacks = [tf.keras.callbacks.History()],batch_size=16)
            del(tem_data)
                
            saved+=1
            if saved==10:
                train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
                saved = 0
                
        train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
"""


# Train with generator
"""
    # 1 Begin training    
    for i in range(10):
        target_list = data_loader.TRAIN_LIST.copy()
        np.random.shuffle(target_list)
        targets = [target_list[index*2:(index+1)*2] for index in range(50)]
        
        j = i+2
        lr = (1e-3)*(0.995)**j
        sgd = SGD(lr=lr)
        train_model._compile_net(sgd)
        dlls = [DataGenerator(target,16) for target in targets]
        
        current_num = 0
        for dll in dlls:
            print(current_num)
            current_num+=1
            length = dll.__len__()
            it_dll = iter(dll)
            his = train_model.train_core.fit(it_dll,epochs=1,callbacks = [tf.keras.callbacks.History()],steps_per_epoch=length)                
            train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")
                
        train_model.train_core.save_weights("weights/rd_"+str(j)+".h5")  
"""        
