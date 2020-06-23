import os
import numpy as np
from crnn_tools.data_loader import libs
from crnn_tools import data_loader
from crnn_tools.core import load_picture
import warnings
import difflib
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

 
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

    
if __name__ == '__main__':

    # 0 Prepare the name of test images
    test_names = os.listdir(data_loader.TEST_LIST[0])

    
    # 1 Load pre-train weight
    from crnn_tools.core import CRNN
    valid_model = CRNN()   
    valid_model._load_weights("weights/rd_5.h5")
    
    
    # 2 Begin evaluating
    result = []
    try:
        with tqdm(test_names) as t:
            for name in t:
                path_name = os.path.join(data_loader.TEST_LIST[0],name)
                img_gray = load_picture(path_name)
                pre_str = valid_model.predict_and_translate(img_gray)
                true_str = ''.join(libs.total_list[libs.total_dic[name]])
                result.append(string_similar(pre_str[0],true_str))
    except:
        t.close()
    print(np.mean(result))

