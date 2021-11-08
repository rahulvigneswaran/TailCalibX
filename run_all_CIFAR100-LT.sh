# CIFAR100_LT

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
# -[x] CBD (0.4)                   # Pre-requisite: Experiment "0.2" but with seeds 10, 20, 30 
# For repo specific to CBD paper with much more detailed instructions, check https://github.com/rahulvigneswaran/Class-Balanced-Distillation-for-Long-Tailed-Visual-Recognition.pytorch
#----------------------------------------------------------
# Generation
# -[x] MODALS (0.5)                 # Pre-requisite: Experiment "0.1" 
#----------------------------------------------------------
# Ours
# -[x] CosineCE + TailCalib (1.2)   # Pre-requisite: Experiment "0.2" 
# -[x] CosineCE + TailCalibX (2.2)  # Pre-requisite: Experiment "0.2" 
# -[x] CBD + TailCalibX (2.4)       # Pre-requisite: Experiment "0.4" 
#----------------------------------------------------------

#----Cifar100_LT
actual_dataset=1

#----CE
experiment_no=0.1 
for seeds in 1 2 3 
do
    python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=1 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=2 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=3 --seed=$seeds --train &
done

wait 

# ----CosineCE
experiment_no=0.2
for seeds in 1 2 3 10 20 30
do
    python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=1 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=2 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=3 --seed=$seeds --train &
done

wait

#----cRT
experiment_no=0.3
for seeds in 1 2 3 
do
    python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=1 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=2 --seed=$seeds --train &
    python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=3 --seed=$seeds --train &
done

wait

#----CBD
experiment_no=0.4
for seeds in 1 2 3
do
    python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=1 --seed=$seeds --cv1=0.8 --cv2=100 --train &
    python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=2 --seed=$seeds --cv1=0.8 --cv2=200 --train &
    python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=3 --seed=$seeds --cv1=0.8 --cv2=100 --train &
done

wait 

#----MODALS 
experiment_no=0.5
for seeds in 1 2 3 
do
    python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=1 --seed=$seeds --cv1=0.01 --generate --retraining &
    python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=2 --seed=$seeds --cv1=0.01 --generate --retraining &
    python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=3 --seed=$seeds --cv1=0.01 --generate --retraining &
done

wait

#----CosineCE + TailCalib
experiment_no=1.2
for imb in 1
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --generate --retraining &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --generate --retraining &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --generate --retraining &
done
wait
for imb in 2
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --generate --retraining &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --generate --retraining &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --generate --retraining &
done
wait
for imb in 3
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --generate --retraining &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --generate --retraining &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --generate --retraining &
done

wait

#----CosineCE + TailCalibX
experiment_no=2.2
for imb in 1
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
done
wait
for imb in 2
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
done
wait
for imb in 3
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
done

wait

#----CBD + TailCalibX
experiment_no=2.4
for imb in 1
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=1.0 --cv2=0.01 --cv3=0.7 --cv4=0.0 --cv5=3 --train &
done
wait
for imb in 2
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.0 --cv4=0.2 --cv5=3 --train &
done
wait
for imb in 3
do
    taskset --cpu-list 10-19 python main.py  --experiment=$experiment_no --gpu="1" --dataset=$actual_dataset --imbalance=$imb --seed=1 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
    taskset --cpu-list 20-29 python main.py  --experiment=$experiment_no --gpu="2" --dataset=$actual_dataset --imbalance=$imb --seed=2 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
    taskset --cpu-list 30-39 python main.py  --experiment=$experiment_no --gpu="3" --dataset=$actual_dataset --imbalance=$imb --seed=3 --cv1=0.9 --cv2=0.01 --cv3=0.9 --cv4=0.0 --cv5=2 --train &
done

#--------------------------------------------------- Paper experiments end here
