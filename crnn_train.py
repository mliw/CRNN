import tensorflow as tf
import os
import numpy as np
from crnn_tools.data_loader import DataGenerator
from crnn_tools.data_loader import libs
from crnn_tools import data_loader
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

    
if __name__ == '__main__':

    # 0 Prepare test_data for validation 
    test_names = os.listdir(data_loader.TEST_LIST[0])
    test_iter = iter(DataGenerator(data_loader.TEST_LIST,50))
    test_data = next(test_iter)
    
    # test_iter = iter(DataGenerator(data_loader.TEST_LIST,len(test_names)))
    # test_data = next(test_iter)

    
    # 1 Define train_model  
    from crnn_tools.core import CRNN
    train_model = CRNN()   
    adam = Adam(lr=1e-3)
    train_model._compile_net(adam)

    # 2 Start training
    for i in range(50):
        dll = DataGenerator(data_loader.TRAIN_LIST) 
        it_dll = iter(dll)
        lr = (1e-3)*(0.995)**i
        adam = Adam(lr=lr)
        train_model._compile_net(adam)
        his = train_model.core.fit
        
        his = train_model.core.fit(it_dll,steps_per_epoch=dll.__len__(),epochs=1,callbacks = [tf.keras.callbacks.History()])
        
        tem_pre = train_model.predict_and_translate(test_data[0][0])
        #msteps_per_epoch=6000
        saved_str = "weights/"+str(i)+"_"+parser(his.history)+".h5"
        train_model.train_model.save_weights(saved_str)
    
    
    cr = train_model.model(data[0])
    y_true = data[1]
    _ctc_loss(y_true,cr)
    
    
    
    tf.compat.v1.nn.ctc_loss(labels=cr, inputs=test_data[1][:200],time_major=False)
    
    
    
"""    
train_model.train_model.save_weights("weights/0.1044_0.1138.h5")


train_model.train_model.compile(optimizer=SGD(0.0001),
                       loss={'rpn_regress_reshape': _rpn_loss_regr, 'rpn_class_reshape': _rpn_loss_cls},
                       loss_weights={'rpn_regress_reshape': 1.0, 'rpn_class_reshape': 1.0})

"""