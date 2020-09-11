import numpy as np
import matplotlib.pyplot as plt

def phase_portrait(input_filename,output_filename):
    data = np.loadtxt(input_filename, delimiter=',')

    data[:,0]=data[:,0]-np.pi # modified for comparing to pilco
    plt.plot(data[:,0],data[:,1])
    plt.xlabel('Angle')
    #plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="x")
    plt.ylabel('Angler Velocity')
    plt.savefig(output_filename)
    plt.close()


if __name__ == '__main__':
    phase_portrait(input_filename="pilco_converted_result_of_1st_policy.csv",
                   output_filename="pilco_converted_result_of_1st_policy.eps")
    phase_portrait(input_filename="pilco_converted_result_of_2nd_policy.csv",
                   output_filename="pilco_converted_result_of_2nd_policy.eps")
