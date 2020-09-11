#!/bin/bash

## 1st episode is obtained applying random policy.
## 2nd episode is obtained applying a policy planned using MB-RL, whose model is estimated usint real-world samples of 1st episode.
## 3rd episode is obtained applying a policy planned using MB-RL, whose model is estimated usint real-world samples of 1st and 2nd episodes.
## Real-world samples for model estimation are obtained using the original PILCO implementation. Thus, in this comparison, models are estimated using the same data.


policy_mode=111  # 11 or 111: (2nd episode) or (3rd episode)
i=2             #  1 or   2: (2nd episode) or (3rd episode)


hidden_num=32
python3 train.py --alg=trpo_mpi --env=CustomPendulum-v1 --num_timesteps=1e7 --save_path "./result_apply/policy${i}/" --train_env_type $policy_mode --num_hidden=$hidden_num --timesteps_per_batch 2000
cd result_apply
cp plot_result.py ./policy${i}/plot_result.py
cd policy${i}
python3 plot_result.py
cd ../../
python3 apply_policy_pendulum_pilco.py --loop_num $((i+1))

