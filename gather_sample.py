import numpy as np
import argparse

def gather_real_world_sample(load_path="./result_apply/",
                             save_path="./result_apply/",
                             loop_num=1):

    print("gather sample loop_num =",loop_num)
    data_si = np.loadtxt(load_path+'real_world_samples_input_1.csv', delimiter=',')
    data_so = np.loadtxt(load_path+'real_world_samples_output_1.csv',delimiter=',')
    #data_c  = np.loadtxt(load_path+'cost_1.csv',                     delimiter=',')

    if 1<loop_num:
        for i in range(1,loop_num):
            data_si = np.r_[data_si, np.loadtxt(load_path+'real_world_samples_input_'+str(i+1)+'.csv' , delimiter=',')]
            data_so = np.r_[data_so, np.loadtxt(load_path+'real_world_samples_output_'+str(i+1)+'.csv', delimiter=',')]
            #data_c  = np.r_[data_c,  np.loadtxt(load_path+'cost_'+str(i+1)+'.csv'                     , delimiter=',')]

    np.savetxt(save_path+'real_world_samples_input.csv', data_si, delimiter=',')
    np.savetxt(save_path+'real_world_samples_output.csv',data_so, delimiter=',')
    #np.savetxt(save_path+'cost.csv',                     data_c,  delimiter=',')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--loop_num', type=int, default=1)
    args = arg_parser.parse_args()
    gather_real_world_sample(loop_num=args.loop_num)
