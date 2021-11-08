# mini-ImageNet-LT

# Table 4
#----------------------------------------------------------
# Baseline
# -[x] CE (0.1)                     # Pre-requisite: None 
# -[x] CosineCE (0.2)               # Pre-requisite: None 
#----------------------------------------------------------
# Decouple
# -[x] cRT (0.3)                   # Pre-requisite: Experiment "0.1" 
#----------------------------------------------------------
# Distillation  
# -[x] CBD (0.4)                   # Pre-requisite: Experiment "0.2" but with seeds 10
# For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
#----------------------------------------------------------
# Generation
# -[x] MODALS (0.5)                 # Pre-requisite: Experiment "0.1" 
#----------------------------------------------------------
# Ours
# -[x] CosineCE + TailCalib (1.2)   # Pre-requisite: Experiment "0.2" 
# -[x] CosineCE + TailCalibX (2.2)  # Pre-requisite: Experiment "0.2" 
#----------------------------------------------------------


#----mini-ImageNet_LT
actual_dataset=2
seeds=1

#----CE
experiment_no=0.1 
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --seed=$seeds --train

wait 

# ----CosineCE
experiment_no=0.2
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --seed=$seeds --train
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --seed=10 --train #For CBD

wait

#----cRT
experiment_no=0.3
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --seed=$seeds --train 

wait

#----CBD
experiment_no=0.4
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --seed=$seeds --cv1=0.4 --cv2=100 --train

wait 

#----MODALS 
experiment_no=0.5
python main.py  --experiment=$experiment_no  --gpu="1" --dataset=$actual_dataset --seed=$seeds --generate --cv1=0.01 --train 

wait

#----CosineCE + TailCalib
experiment_no=1.2
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --generate --retraining

wait

#----CosineCE + TailCalibX
experiment_no=2.2
python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train

wait
#--------------------------------------------------- Paper experiments end here
