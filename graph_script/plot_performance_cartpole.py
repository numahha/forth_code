import numpy as np
import matplotlib.pyplot as plt

c=50
loop_num=6
trial_num=20

loop_list=[]
cost_mean=[]
cost_std=[]
for loop in range(1,loop_num+1):
    epi_cost=[]
    for trial in range(1,trial_num+1):
        try:
            data = np.loadtxt('./c'+str(c)+'_npm_trial'+str(trial)+'/result_apply/cost_'+str(loop)+'.csv',delimiter=',')
            epi_cost.append(data.sum())
        except:
            pass
    if len(epi_cost)>0:
        temp_cost = np.array(epi_cost)
        cost_mean.append(-temp_cost.mean())
        cost_std.append(temp_cost.std())
        loop_list.append(loop)
        print("npm",epi_cost)
plt.errorbar(loop_list,cost_mean,yerr=cost_std,color="C1",label='MBRL(NPM)')

loop_list=[]
cost_mean=[]
cost_std=[]
for loop in range(1,loop_num+1):
    epi_cost=[]
    for trial in range(1,trial_num+1):
        try:
            data = np.loadtxt('./c'+str(c)+'_pm_trial'+str(trial)+'/result_apply/cost_'+str(loop)+'.csv',delimiter=',')
            epi_cost.append(data.sum())
        except:
            pass
    if len(epi_cost)>0:
        temp_cost = np.array(epi_cost)
        cost_mean.append(-temp_cost.mean())
        cost_std.append(temp_cost.std())
        loop_list.append(loop)
        print("pm",epi_cost)
plt.errorbar(loop_list,cost_mean,yerr=cost_std,color="C2",label='MBRL(PM)')

loop_list=[]
cost_mean=[]
cost_std=[]
for loop in range(1,loop_num+1):
    epi_cost=[]
    for trial in range(1,trial_num+1):
        try:
            data = np.loadtxt('./c'+str(c)+'_meta_trial'+str(trial)+'/result_apply/cost_'+str(loop)+'.csv',delimiter=',')
            epi_cost.append(data.sum())
        except:
            pass
    if len(epi_cost)>0:
        temp_cost = np.array(epi_cost)
        cost_mean.append(-temp_cost.mean())
        cost_std.append(temp_cost.std())
        loop_list.append(loop)
        print("meta",epi_cost)
plt.errorbar(loop_list,cost_mean,yerr=cost_std,color="C0",label='Plain') 


loop_list=[]
cost_mean=[]
cost_std=[]
for loop in range(1,loop_num+1):
    epi_cost=[]
    for trial in range(1,trial_num+1):
        try:
            data = np.loadtxt('./c'+str(c)+'_proposed_trial'+str(trial)+'/result_apply/cost_'+str(loop)+'.csv',delimiter=',')
            epi_cost.append(data.sum())
        except:
            pass
    if len(epi_cost)>0:
        temp_cost = np.array(epi_cost)
        cost_mean.append(-temp_cost.mean())
        cost_std.append(temp_cost.std())
        loop_list.append(loop)
        print("proposed",epi_cost)
plt.errorbar(loop_list,cost_mean,yerr=cost_std,color="C3",label='Proposed') 


plt.xlabel('Episode Number', fontsize=18)
plt.ylabel('Episode cost', fontsize=18)
plt.legend(fontsize=12)

#plt.ylim([90,200])
plt.ylim([50,250])
plt.savefig('performance_cartpole'+str(c)+'.pdf')
plt.savefig('performance_cartpole'+str(c)+'.svg')
plt.show()


