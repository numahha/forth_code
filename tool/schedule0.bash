#!/bin/bash

d=0

for i in {1..10}; do

    cp -r c${d}_npm_trial$i c${d}_npm_trial$((i+1))    
    cd c${d}_npm_trial$i    
    MSG=`bash ./run_pendulum.bash`
    echo $MSG
    cd ../

    cp -r c${d}_pm_trial$i c${d}_pm_trial$((i+1))        
    cd c${d}_pm_trial$i
    MSG=`bash ./run_pendulum.bash`
    echo $MSG
    cd ../

    cp -r c${d}_meta_trial$i c${d}_meta_trial$((i+1))    
    cd c${d}_meta_trial$i
    MSG=`bash ./run_pendulum.bash`
    echo $MSG
    cd ../

    cp -r c${d}_proposed_trial$i c${d}_proposed_trial$((i+1))    
    cd c${d}_proposed_trial$i
    MSG=`bash ./run_pendulum.bash`
    echo $MSG
    cd ../

    
done
