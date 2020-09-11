#!/bin/bash

hidden_num=32

rm -r ./result_apply/policy1/
cp -r ./result_apply/candidate_policy1_1/ ./result_apply/policy1/

mv ./result_apply/current_state.csv   ./result_apply/current_state.csv.bak
cp ./result_apply/current_state_1.csv ./result_apply/current_state.csv

mv ./result_apply/real_world_samples_input_2.csv  ./result_apply/real_world_samples_input_2.csv.bak
mv ./result_apply/real_world_samples_output_2.csv ./result_apply/real_world_samples_output_2.csv.bak
mv ./result_apply/current_state_2.csv             ./result_apply/current_state_2.csv.bak
mv ./result_apply/cost_2.csv                      ./result_apply/cost_2.csv.bak 

python3 apply_policy_pendulum.py --loop_num 2 --num_hidden=$hidden_num

mv ./result_apply/real_world_samples_input_2.csv  ./result_apply/real_world_samples_input_2if.csv
mv ./result_apply/real_world_samples_output_2.csv ./result_apply/real_world_samples_output_2if.csv
mv ./result_apply/current_state_2.csv             ./result_apply/current_state_2if.csv
mv ./result_apply/cost_2.csv                      ./result_apply/cost_2if.csv

mv ./result_apply/real_world_samples_input_2.csv.bak  ./result_apply/real_world_samples_input_2.csv
mv ./result_apply/real_world_samples_output_2.csv.bak ./result_apply/real_world_samples_output_2.csv
mv ./result_apply/current_state_2.csv.bak             ./result_apply/current_state_2.csv
mv ./result_apply/cost_2.csv.bak                      ./result_apply/cost.csv

rm ./result_apply/current_state.csv
mv ./result_apply/current_state.csv.bak   ./result_apply/current_state.csv


rm -r ./result_apply/policy1/
cp -r ./result_apply/candidate_policy1_2/ ./result_apply/policy1/


