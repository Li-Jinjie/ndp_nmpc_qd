#!/bin/zsh

python_script="nn_train.py"
SNs=(0 2 4 8)
epochs=(20000 20000 20000 20000)

# loop over SNs and epochs
for i in {1..4}
do
    SN=${SNs[$i]}
    epoch=${epochs[$i]}
    echo "Running with configuration: SN: $SN, epoch: $epoch"
    python $python_script --SN $SN --epoch $epoch
done
