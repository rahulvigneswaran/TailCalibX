# CIFAR100_LT

# Ours
#----------------------------------------------------------
# -[x] CosineCE + TailCalib (1.2)   # Pre-requisite: Experiment "0.2" - Check "run_all_CIFAR100-LT.sh"
# -[x] CosineCE + TailCalibX (2.2)  # Pre-requisite: Experiment "0.2" - Check "run_all_CIFAR100-LT.sh"
# -[x] CBD + TailCalibX (2.4)       # Pre-requisite: Experiment "0.4" - Check "run_all_CIFAR100-LT.sh"
#----------------------------------------------------------

#----Cifar100_LT
actual_dataset=1

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
