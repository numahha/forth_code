import numpy as np
import matplotlib.pyplot as plt

# This scripts is only for pendulum.

input_dir="./c250_proposed_trial1b/result_apply/"




if __name__ == '__main__':

    data = np.loadtxt(input_dir+"real_world_samples_input_1.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="black", label="$D^{real}$")

    data = np.loadtxt(input_dir+"candidate_policy1_1/simulation1/simulation_samples_input.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C0", label="$d(\pi_{NPM},\hat{m}_{NPM})$",linestyle="dashed")
    plt.plot(data[-1,0],data[-1,1], color="C0", marker='s', markersize=8)
    data = np.loadtxt(input_dir+"candidate_policy1_1/simulation2/simulation_samples_input.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C0", label="$d(\pi_{NPM},\hat{m}_{PM})$",linestyle="dotted")
    plt.plot(data[-1,0],data[-1,1], color="C0", marker='s', markersize=8)
    data = np.loadtxt(input_dir+"real_world_samples_input_2if.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C0", label="$d^{real}(\pi_{NPM})$")
    plt.plot(data[-1,0],data[-1,1], color="C0", marker='s', markersize=8)


    data = np.loadtxt(input_dir+"candidate_policy1_2/simulation1/simulation_samples_input.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C3", label="$d(\pi_{PM},\hat{m}_{NPM})$",linestyle="dashed")
    plt.plot(data[-1,0],data[-1,1], color="C3", marker='s', markersize=8)
    data = np.loadtxt(input_dir+"candidate_policy1_2/simulation2/simulation_samples_input.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C3", label="$d(\pi_{PM},\hat{m}_{PM})$",linestyle="dotted")
    plt.plot(data[-1,0],data[-1,1], color="C3", marker='s', markersize=8)
    data = np.loadtxt(input_dir+"real_world_samples_input_2.csv", delimiter=',')
    data[:,0]=-data[:,0]+2*np.pi
    data[:,1]=-data[:,1]
    plt.plot(data[:,0],data[:,1], color="C3", label="$d^{real}(\pi_{PM})$")
    plt.plot(data[-1,0],data[-1,1], color="C3", marker='s', markersize=8)


    plt.xlabel('Angle',fontsize=18)
    #plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="x")
    plt.ylabel('Angler velocity',fontsize=18)
    plt.legend(fontsize=11.5)
    plt.savefig("./phase250_sample.pdf")
    plt.savefig("./phase250_sample.svg")

    plt.show()
    plt.close()

