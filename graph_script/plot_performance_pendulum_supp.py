import numpy as np
import matplotlib.pyplot as plt

loop_num=9#7


def total_cost_npm(dirname='./', trial=1, c_zerodot=0):
    ret_val=0.
    for i in range(1,loop_num):
        data = np.loadtxt(dirname+'c'+str(c_zerodot)+'_npm_trial'+str(trial)+'/result_apply/cost_'+str(i)+'.csv',delimiter=',')
        ret_val += data.sum()
    return ret_val

def total_cost_pm(dirname='./', trial=1, c_zerodot=0):
    ret_val=0.
    for i in range(1,loop_num):
        data = np.loadtxt(dirname+'c'+str(c_zerodot)+'_pm_trial'+str(trial)+'/result_apply/cost_'+str(i)+'.csv',delimiter=',')
        ret_val += data.sum()
    return ret_val

def total_cost_meta(dirname='./', trial=1, c_zerodot=0):
    ret_val=0.
    for i in range(1,loop_num):
        data = np.loadtxt(dirname+'c'+str(c_zerodot)+'_bayesiandp_trial'+str(trial)+'/result_apply/cost_'+str(i)+'.csv',delimiter=',')
        ret_val += data.sum()
    return ret_val

def total_cost_pro(dirname='./', trial=1, c_zerodot=0):
    ret_val=0.
    for i in range(1,loop_num):
        data = np.loadtxt(dirname+'c'+str(c_zerodot)+'_epopt_trial'+str(trial)+'/result_apply/cost_'+str(i)+'.csv',delimiter=',')
        ret_val += data.sum()
    return ret_val


c_list=[]
npm=[]
pm=[]
meta=[]
ave=[]
pro=[]
npm_std=[]
pm_std=[]
meta_std=[]
pro_std=[]

for c in [0,250,500]:

    temp_list_npm=[]
    temp_list_pm=[]
    temp_list_meta=[]
    temp_list_pro=[]
    for d in range(0,1):

        if 0==d:
            trial_num=10
            dn='./'
        if 1==d:
            trial_num=10
            dn='./temp/'
        
        #temp_list_npm.extend( [total_cost_npm(dirname=dn, trial=(t+1), c_zerodot=c)   for t in range(0,trial_num)] )
        temp_list_pm.extend( [total_cost_pm(dirname=dn, trial=(t+1), c_zerodot=c)   for t in range(0,trial_num)] )
        temp_list_meta.extend( [total_cost_meta(dirname=dn, trial=(t+1), c_zerodot=c)   for t in range(0,trial_num)] )
        temp_list_pro.extend( [total_cost_pro(dirname=dn, trial=(t+1), c_zerodot=c)   for t in range(0,trial_num)] )

    c_list.append(0.001*c)
    #temp_npm  = np.array( temp_list_npm )
    temp_pm   = np.array( temp_list_pm )
    temp_meta = np.array( temp_list_meta )
    temp_pro  = np.array( temp_list_pro )

    #npm.append(-temp_npm.mean())
    pm.append(-temp_pm.mean())
    meta.append(-temp_meta.mean())
    pro.append(-temp_pro.mean())

    #npm_std.append(temp_npm.std())
    pm_std.append(temp_pm.std())
    meta_std.append(temp_meta.std())
    pro_std.append(temp_pro.std())
    #print("npm",c,temp_npm)
    #print("meta",c,temp_meta)
    #print("pro",c,temp_pro)

#print("npm",npm)
#print("meta",meta)
#print("pro",pro)


#plt.plot(c_list,npm,color='red',label='$\pi_1$')
#plt.plot(c_list,pm,color='limegreen',label='$\pi_2$')
#plt.plot(c_list,proposed,color='black',label='$\pi_3$')

#plt.errorbar(c_list,npm,yerr=npm_std,capsize=5,color="C1",label='MBRL(NPM)')
plt.errorbar(c_list,pm,yerr=pm_std,capsize=5,color="C2",label='MBRL(PM)')
plt.errorbar(c_list,meta,yerr=meta_std,capsize=5,color="C6",label='Bayesian DP')
plt.errorbar(c_list,pro,yerr=pro_std,capsize=5,color="C5",label='EPOpt')

plt.xlabel('Viscosity coefficient (modeling error of PM)', fontsize=18)
plt.ylabel('Episode cost', fontsize=18)
plt.legend(fontsize=12)
plt.savefig('performance_pendulum_supp.pdf')
plt.savefig('performance_pendulum_supp.svg')
plt.show()

