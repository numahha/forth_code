#!/bin/bash

policy_mode=5 # 1 (NPM),  2 (PM), 3 (plain), 5 (proposed)

hidden_num=32


python3 apply_policy_pendulum.py --loop_num 1 --num_hidden=$hidden_num
cp plot_phase.py  ./result_apply/plot_phase.py
cp plot_result.py ./result_apply/plot_result.py


for i in {1..7}; do
  python3 gather_sample.py --loop_num $((i))
  cd result_apply
  rm current_state.csv
  cp current_state_$i.csv current_state.csv
  python3 plot_phase.py
  cd ../

  if [ $i -lt 4 ]; then
    batch=25000
    timestep_num=10000000
  else
    batch=5000
    timestep_num=2000000    
  fi

  if [ $policy_mode = 5 ]; then
    for j in {1..2}; do
      python3 train.py --alg=trpo_mpi --env=CustomPendulum-v0 --num_timesteps=$timestep_num --save_path "./result_apply/policy${i}/" --train_env_type $j --num_hidden=$hidden_num --timesteps_per_batch $batch
      python3 gather_sample.py --loop_num $i
      python3 apply_policy_pendulum_simulation.py --loop_num $i --sim_env_type 1 --num_hidden=$hidden_num
      python3 apply_policy_pendulum_simulation.py --loop_num $i --sim_env_type 2 --num_hidden=$hidden_num
      python3 estimate_log_evidence.py --loop_num $i --env=CustomPendulum-v0
      mv ./result_apply/policy${i}/ ./result_apply/candidate_policy${i}_${j}/
    done

    selected_model1=$(cat ./result_apply/candidate_policy${i}_1/selected_model.csv)
    selected_model2=$(cat ./result_apply/candidate_policy${i}_2/selected_model.csv)
    recomended_model1=$(cat ./result_apply/candidate_policy${i}_1/recomended_model.csv)
    recomended_model2=$(cat ./result_apply/candidate_policy${i}_2/recomended_model.csv)

    if [ $selected_model1 = $recomended_model1 ]; then
      if [ $selected_model2 = $recomended_model2 ]; then
        # Both policies are desirable. Thus, compare cost.
        reward1=$(cat ./result_apply/candidate_policy${i}_1/simulation1/simulation_total_cost.csv)
        reward2=$(cat ./result_apply/candidate_policy${i}_2/simulation2/simulation_total_cost.csv)
        result=`echo "$reward1 > $reward2" | bc`
        if [ $result -eq 1 ]; then
          cp -rf ./result_apply/candidate_policy${i}_1/  ./result_apply/policy${i}/
        else
          cp -rf ./result_apply/candidate_policy${i}_2/  ./result_apply/policy${i}/
        fi
      else
        cp -rf ./result_apply/candidate_policy${i}_1/  ./result_apply/policy${i}/
      fi
    else
      if [ $selected_model2 = $recomended_model2 ]; then
        cp -rf ./result_apply/candidate_policy${i}_2/  ./result_apply/policy${i}/
      else
        # Both policies are undesirable. Thus, compare maginal likelihood.
        log_evi1=$(cat ./result_apply/candidate_policy${i}_1/log_evidence_for_learn1.csv)
        log_evi2=$(cat ./result_apply/candidate_policy${i}_2/log_evidence_for_learn2.csv)
        result=`echo "$log_evi1 > $log_evi2" | bc`
        if [ $result -eq 1 ]; then
          cp -rf ./result_apply/candidate_policy${i}_1/  ./result_apply/policy${i}/
        else
          cp -rf ./result_apply/candidate_policy${i}_2/  ./result_apply/policy${i}/
        fi
      fi
    fi


  else
    python3 train.py --alg=trpo_mpi --env=CustomPendulum-v0 --num_timesteps=$timestep_num --save_path "./result_apply/policy${i}/" --train_env_type $policy_mode --num_hidden=$hidden_num --timesteps_per_batch $batch
  fi

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
