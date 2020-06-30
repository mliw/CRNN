import os

while True:
    try:
        os.system("python crnn_train.py")
    except:
        print("error!")
        continue