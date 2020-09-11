import numpy as np
import argparse

    
if __name__ == '__main__':
    #arg_parser = argparse.ArgumentParser()
    #arg_parser.add_argument('--loop_num', type=int, default=1)
    #args = arg_parser.parse_args()
    cost_list=[]
    for i in range(1,9):
        data = np.loadtxt('./result_apply/cost_'+str(i)+'.csv',  delimiter=',')
        print(i,data.sum())
        cost_list.append(data.sum())
    
