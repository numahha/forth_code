import numpy as np
import matplotlib.pyplot as plt

# This scripts is only for pendulum.



if __name__ == '__main__':

    data = np.loadtxt("./c500_npm_trial1/result_apply/real_world_samples_input.csv", delimiter=',')
    plt.plot(data[:,0],data[:,1], color="C1",label="MBRL(NPM)")
    data = np.loadtxt("./c500_pm_trial1/result_apply/real_world_samples_input.csv", delimiter=',')
    plt.plot(data[:,0],data[:,1], color="C2",label="MBRL(PM)")



    plt.xlabel('Angle',fontsize=18)
    #plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="x")
    plt.ylabel('Angler velocity',fontsize=18)
    plt.legend(fontsize=12)
    plt.savefig("./phase500.pdf")
    plt.savefig("./phase500.svg")
    plt.close()

