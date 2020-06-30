import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    
    # 0 Load CRNN weight
    from crnn_tools.infer_core import CRNN
    infer_model = CRNN() 
    infer_model._load_weights("weights/CRNN.h5")
    
   
    # 1 Load paths
    files = os.listdir("asset")
    paths = [os.path.join("asset", file) for file in files]
    
    
    # 2 Begin inferring
    for path in paths:
        result = infer_model.predict_path(path)
    
