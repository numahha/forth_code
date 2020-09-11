#!/bin/bash

policy_mode=2 # 1 (NPM),  2 (PM),  or 3 (meta)

hidden_num=32



python3 apply_policy_pendulum.py --loop_num 1 --num_hidden=$hidden_num
cp plot_phase.py  ./result_apply/plot_phase.py
cp plot_result.py ./result_apply/plot_result.py

for i in {1..7}; do
    cd result_apply
    python3 plot_phase.py
    cd ../

    python3 train.py --alg=trpo_mpi --env=CustomPendulum-v0 --num_timesteps=1e6 --save_path "./result_apply/policy${i}pre/" --train_env_type $policy_mode --num_hidden=$hidden_num --timesteps_per_batch 5000
    python3 train.py --alg=trpo_mpi --env=CustomPendulum-v0 --num_timesteps=1e6 --load_path "./result_apply/policy${i}pre/policy/" --save_path "./result_apply/policy${i}/" --train_env_type $policy_mode --num_hidden=$hidden_num --timesteps_per_batch 5000
	
    cd result_apply
    cp plot_result.py ./policy${i}/plot_result.py
    cd policy${i}
    python3 plot_result.py
    cd ../../
    python3 apply_policy_pendulum.py --loop_num $((i+1)) --num_hidden=$hidden_num
done

cd result_apply
python3 plot_phase.py
cd ../
