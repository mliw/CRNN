import os
from multiprocessing import Pool

def execute_line(my_str):
    os.system(my_str)
    return


if __name__=="__main__":

    # 0 Set multiprocessing function
    executor = Pool(2)


    # 1 Load order lines
    with open("data/unpacking.txt","r") as f:
        order_lines = f.readlines()


    # 2 Start unpacking
    executor.map(execute_line,order_lines) 
    