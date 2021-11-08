# mini-ImageNet-LT

# Ours
#----------------------------------------------------------
# -[x] CosineCE + TailCalib (1.2)   # Pre-requisite: Experiment "0.2" - Check "run_all_mini-ImageNet-LT.sh"
# -[x] CosineCE + TailCalibX (2.2)  # Pre-requisite: Experiment "0.2" - Check "run_all_mini-ImageNet-LT.sh"
#----------------------------------------------------------

#----mini-ImageNet_LT
actual_dataset=2
seeds=1

#----CosineCE + TailCalib
experiment_no=1.2
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --generate --retraining

wait

#----CosineCE + TailCalibX
experiment_no=2.2
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train

wait
#--------------------------------------------------- Paper experiments end here
