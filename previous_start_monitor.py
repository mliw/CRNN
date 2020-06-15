import os
import shutil
import time
from tqdm import tqdm
MONITOR_PATH = "data/image/images"
START_NUM = 1


def renew(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)    
        

def force_renew(path):
    while True:
        length = len(os.listdir(path))
        if length==0:
            return
        else:
            try:
                renew(path)
            except:
                print("something wrong!")
                

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)    
        
        
def pool_re(entity):
    os.rename(entity[0],entity[1])
    
            
def scan(monitor_path,num,threshold):
      
    monitor_list = os.listdir(monitor_path)
    length = len(monitor_list)
    copy_path = "data/img_batches/batch_"+str(num)
    safe_mkdir(copy_path)
    copy_length = len(os.listdir(copy_path))
    num_logi = True if copy_length==36000 else False
    
    if num_logi:
        return None,num_logi
        
    print("="*60)
    if length>threshold:
        print("Current length is {}, start copying and deleting...".format(length))
        print("="*60)
        copy_list = monitor_list[:threshold]
        copy_path = "data/img_batches/batch_"+str(num)
        poop_list = [(monitor_path+"/"+items,copy_path+"/"+items) for items in copy_list]

        try:
          with tqdm(poop_list) as t:
            for entity in t:
                pool_re(entity)

        except KeyboardInterrupt:
              t.close()
    else:
        print("Current length is {}, start scanning next time...".format(length))
        print("="*60)   
    return monitor_list,num_logi


if __name__=="__main__":
    number = START_NUM
    monitor_path = MONITOR_PATH
    threshold = 36000
    while True:
        current_monitor,logi = scan(monitor_path,number,threshold)
        if logi:
            print("Update the number, the current number is {}".format(number))
            number+=1
        
        