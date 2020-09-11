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
                              loop_num=0,
                              i=5):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.delete(X,i,0)
    Y = np.delete(Y,i,0)
    print("X.shape =",X.shape)
    npm_class = pendulum_npm.PendulumNPM(X,Y)

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.delete(X,i,0)
    Y = np.delete(Y,i,0)
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
    X = np.delete(X,i,0)
    Y = np.delete(Y,i,0)
    X2= np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y2= np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
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
    print("diff =",temp_le-npm_class.log_evidence())
    return (temp_le-npm_class.log_evidence())

def estimate_log_evidence_pm_pendulum(load_path="./result_apply/",
                             save_path="./result_apply/",
                             loop_num=0,
                             i=5):
    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.delete(X,i,0)
    Y = np.delete(Y,i,0)
    pm_class = pendulum_pm.PendulumPM(X,Y)

    gather_sample.gather_real_world_sample(loop_num=loop_num)
    X = np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y = np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')
    X = np.delete(X,i,0)
    Y = np.delete(Y,i,0)
    X2= np.loadtxt(load_path+'real_world_samples_input.csv',  delimiter=',')
    Y2= np.loadtxt(load_path+'real_world_samples_output.csv', delimiter=',')

    print("prev_log_evidence =",pm_class.log_evidence())
    print("debug_log_evidence =",pm_class.log_evidence_of_new_data(X,Y))
    print("new_log_evidence =",pm_class.log_evidence_of_new_data(X2,Y2))
    print("diff =",pm_class.log_evidence_of_new_data(X2,Y2)-pm_class.log_evidence())
    temp_data = np.array([pm_class.log_evidence(), pm_class.log_evidence_of_new_data(X2,Y2)])
    return (pm_class.log_evidence_of_new_data(X2,Y2)-pm_class.log_evidence())

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--loop_num', type=int, default=1)
    arg_parser.add_argument('--env', type=str,default="CustomPendulum-v0")
    args = arg_parser.parse_args()


    num=[]
    npm_cv=[]
    pm_cv=[]
    for i in range(0,50):
        npm_evi = estimate_log_evidence_npm_pendulum(loop_num=args.loop_num, i=i)
        pm_evi  = estimate_log_evidence_pm_pendulum(loop_num=args.loop_num, i=i)
        num.append(i)
        npm_cv.append(npm_evi)
        pm_cv.append(pm_evi)
    import matplotlib.pyplot as plt
    plt.plot(num,npm_cv,label='NPM',color="C1")
    plt.plot(num,pm_cv,label='PM',color="C2")
    print(sum(npm_cv))
    print(sum(pm_cv))
    plt.ylim([0, 5])
    plt.xlabel('Number of training samples', fontsize=18)
    plt.ylabel('Predictive likelihood', fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig('validation.pdf')
    plt.savefig('validation.svg')
    plt.show()

