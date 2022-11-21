
Reproducibility is an issue when it comes to CIFAR100_LT. This is because different seeds select different data points when creating the imbalanced dataset. This will not be resolved even after choosing the same seed as ours `[1,2,3]`, because the randomness caused by the system (ours vs yours) still exists.

To evade this issue, we fixed and saved the data points we used for each imbalance factor and seed. 

So, if you want to exactly reproduce our results, we suggest you to load the data, target tuple for respective phase, imb_factor, seed from here.
