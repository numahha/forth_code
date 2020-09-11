import numpy as np
import argparse
import gather_sample
import GPy

import pendulum_npm
import pendulum_pm

import cartpole_npm
import cartpole_pm

def gpr_without_optimize(x, yi):
    kernel = GPy.kern.RBF(x.shape[1],ARD=True)
    model = GPy.models.GPRegression(x, yi, kernel)
    #model.optimize(messages=True, max_iters=3e5)
    return model

def estimate_log_evidence_npm_pendulum(load_path="./result_apply/",
                              save_path="./result_apply/",
                              loop_num=0):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    npm_class = pendulum_npm.PendulumNPM(X,Y)

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.insert(X, 1, np.sin(X[:,0]), axis=1) # [th,sinth,thdot,u]
    X[:,0] = np.cos(X[:,0])                     # [costh,sinth,thdot,u]
    gprd_m1 = gpr_without_optimize(X, Y[:,0].reshape((Y[:,0].shape[0]),1))
    gprd_m2 = gpr_without_optimize(X, Y[:,1].reshape((Y[:,1].shape[0]),1))
    gprd_m1[:] = npm_class.gpr_m1.param_array
    gprd_m2[:] = npm_class.gpr_m2.param_array

    print("prev_log_evidence =",npm_class.log_evidence())
    print("debug_log_evidence =",gprd_m1.log_likelihood() + gprd_m2.log_likelihood())

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X2 = np.r_[X, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation1/simulation_samples_input.csv' , delimiter=',')]
    Y2 = np.r_[Y, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation1/simulation_samples_output.csv', delimiter=',')]
    X2 = np.insert(X2, 1, np.sin(X2[:,0]), axis=1) # [th,sinth,thdot,u]
    X2[:,0] = np.cos(X2[:,0])                     # [costh,sinth,thdot,u]
    gpr_m1 = gpr_without_optimize(X2, Y2[:,0].reshape((Y2[:,0].shape[0]),1))
    gpr_m2 = gpr_without_optimize(X2, Y2[:,1].reshape((Y2[:,1].shape[0]),1))
    gpr_m1[:] = npm_class.gpr_m1.param_array
    gpr_m2[:] = npm_class.gpr_m2.param_array

    print("prev_log_evidence =",npm_class.log_evidence())
    print("debug_log_evidence =",gprd_m1.log_likelihood() + gprd_m2.log_likelihood())
    temp_le = gpr_m1.log_likelihood() + gpr_m2.log_likelihood()
    print("new_log_evidence =",temp_le)

    temp_data = np.array([npm_class.log_evidence(), temp_le])
    np.savetxt(save_path+"policy"+str(loop_num)+'/simulation1/log_evidence.csv',temp_data,delimiter=',')
    np.savetxt(save_path+"policy"+str(loop_num)+'/log_evidence_for_learn1.csv',np.array([npm_class.log_evidence()]),delimiter=',', fmt ='%.6f')
    return temp_data[1]

def estimate_log_evidence_pm_pendulum(load_path="./result_apply/",
                             save_path="./result_apply/",
                             loop_num=0):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    pm_class = pendulum_pm.PendulumPM(X,Y)

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')

    X2 = np.r_[X, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation2/simulation_samples_input.csv' , delimiter=',')]
    Y2 = np.r_[Y, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation2/simulation_samples_output.csv', delimiter=',')]

    print("prev_log_evidence =",pm_class.log_evidence())
    print("debug_log_evidence =",pm_class.log_evidence_of_new_data(X,Y))
    print("new_log_evidence =",pm_class.log_evidence_of_new_data(X2,Y2))
    temp_data = np.array([pm_class.log_evidence(), pm_class.log_evidence_of_new_data(X2,Y2)])
    np.savetxt(save_path+"policy"+str(loop_num)+'/simulation2/log_evidence.csv',temp_data,delimiter=',')
    np.savetxt(save_path+"policy"+str(loop_num)+'/log_evidence_for_learn2.csv',np.array([pm_class.log_evidence()]),delimiter=',', fmt ='%.6f')
    return temp_data[1]

def estimate_log_evidence_npm_cartpole(load_path="./result_apply/",
                              save_path="./result_apply/",
                              loop_num=0):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    npm_class = cartpole_npm.CartPoleNPM(X,Y)

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.insert(X, 3, np.sin(X[:,2]), axis=1) # [z,zdot,th,sinth,thdot,u]
    X[:,2] = np.cos(X[:,2])                     # [z,zdot,costh,sinth,thdot,u]

    gprd_m1 = gpr_without_optimize(X, Y[:,0].reshape((Y[:,0].shape[0]),1))
    gprd_m2 = gpr_without_optimize(X, Y[:,1].reshape((Y[:,1].shape[0]),1))
    gprd_m3 = gpr_without_optimize(X, Y[:,2].reshape((Y[:,2].shape[0]),1))
    gprd_m4 = gpr_without_optimize(X, Y[:,3].reshape((Y[:,3].shape[0]),1))
    gprd_m1[:] = npm_class.gpr_m1.param_array
    gprd_m2[:] = npm_class.gpr_m2.param_array
    gprd_m3[:] = npm_class.gpr_m3.param_array
    gprd_m4[:] = npm_class.gpr_m4.param_array

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X2 = np.r_[X, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation1/simulation_samples_input.csv' , delimiter=',')]
    Y2 = np.r_[Y, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation1/simulation_samples_output.csv', delimiter=',')]
    X2 = np.insert(X2, 3, np.sin(X2[:,2]), axis=1) # [z,zdot,th,sinth,thdot,u]
    X2[:,2] = np.cos(X2[:,2])                     # [z,zdot,costh,sinth,thdot,u]
    gpr_m1 = gpr_without_optimize(X2, Y2[:,0].reshape((Y2[:,0].shape[0]),1))
    gpr_m2 = gpr_without_optimize(X2, Y2[:,1].reshape((Y2[:,1].shape[0]),1))
    gpr_m3 = gpr_without_optimize(X2, Y2[:,2].reshape((Y2[:,2].shape[0]),1))
    gpr_m4 = gpr_without_optimize(X2, Y2[:,3].reshape((Y2[:,3].shape[0]),1))
    gpr_m1[:] = npm_class.gpr_m1.param_array
    gpr_m2[:] = npm_class.gpr_m2.param_array
    gpr_m3[:] = npm_class.gpr_m3.param_array
    gpr_m4[:] = npm_class.gpr_m4.param_array

    print("prev_log_evidence =",npm_class.log_evidence())
    print("debug_log_evidence =",gprd_m1.log_likelihood() + gprd_m2.log_likelihood() + gprd_m3.log_likelihood() + gprd_m4.log_likelihood())
    temp_le = gpr_m1.log_likelihood() + gpr_m2.log_likelihood() + gpr_m3.log_likelihood() + gpr_m4.log_likelihood()
    print("new_log_evidence =",temp_le)

    temp_data = np.array([npm_class.log_evidence(), temp_le])
    np.savetxt(save_path+"policy"+str(loop_num)+'/simulation1/log_evidence.csv',temp_data,delimiter=',')
    np.savetxt(save_path+"policy"+str(loop_num)+'/log_evidence_for_learn1.csv',np.array([npm_class.log_evidence()]),delimiter=',', fmt ='%.6f')
    return temp_data[1]    

def estimate_log_evidence_pm_cartpole(load_path="./result_apply/",
                             save_path="./result_apply/",
                             loop_num=0):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    pm_class = cartpole_pm.CartPolePM(X,Y)

    X2 = np.r_[X, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation2/simulation_samples_input.csv' , delimiter=',')]
    Y2 = np.r_[Y, np.loadtxt(load_path+"policy"+str(loop_num)+'/simulation2/simulation_samples_output.csv', delimiter=',')]

    print("prev_log_evidence =",pm_class.log_evidence())
    print("debug_log_evidence =",pm_class.log_evidence_of_new_data(X,Y))
    print("new_log_evidence =",pm_class.log_evidence_of_new_data(X2,Y2))
    temp_data = np.array([pm_class.log_evidence(), pm_class.log_evidence_of_new_data(X2,Y2)])
    np.savetxt(save_path+"policy"+str(loop_num)+'/simulation2/log_evidence.csv',temp_data,delimiter=',')
    np.savetxt(save_path+"policy"+str(loop_num)+'/log_evidence_for_learn2.csv',np.array([pm_class.log_evidence()]),delimiter=',', fmt ='%.6f')
    return temp_data[1]

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--loop_num', type=int, default=1)
    arg_parser.add_argument('--env', type=str,default="CustomPendulum-v0")
    args = arg_parser.parse_args()

    if args.env=="CustomPendulum-v0" or args.env=="CustomPendulum-v1":
        npm_evi = estimate_log_evidence_npm_pendulum(loop_num=args.loop_num)
        pm_evi  = estimate_log_evidence_pm_pendulum(loop_num=args.loop_num)

    if args.env=="CustomCartPole-v0":
        npm_evi = estimate_log_evidence_npm_cartpole(loop_num=args.loop_num)
        pm_evi  = estimate_log_evidence_pm_cartpole(loop_num=args.loop_num)


    if npm_evi>pm_evi:
        recomended_model=1
    else:
        recomended_model=2
    np.savetxt("./result_apply/policy"+str(args.loop_num)+"/recomended_model.csv",np.array([recomended_model]),delimiter=',', fmt="%.0f")

