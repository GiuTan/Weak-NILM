# Weak-NILM
This repository contains source code to perform multi-label classification using a deep learning approach trained with weakly labeled data. This work refers to [Multilabel Appliance Classification With Weakly Labeled Data for Non-Intrusive Load Monitoring](https://ieeexplore.ieee.org/document/9831435).
The implemented network is a Convolutional Recurrent Neural Network. 
Both strongly and weakly labeled data are generated from two real-world benchmark datasets: REFIT and UK-DALE.

Two experiments are proposed:
- one based on UK-DALE data where it is possible to vary the percentage of bags with strong annotations as well as weak annotations
- one based on mixed dataset where the network can be trained with a percentage of strong annotations from UK-DALE and a quantity of weak annotations from REFIT, testing on both datasets the performance of the network. 

In dataset_creation folder code for synthetic aggregates creation is available for both UK-DALE and REFIT. Modules noise_extraction.py and noised_aggregate_creation.py have to be used to create noised aggregate vectors, adding noise to synthetic vectors.
Appliances taken into consideration are kettle, microwave, fridge, washing machine and dishwasher.

Data will be created with both types of label. Appliance states are set to 1 in strong annotations when the specific appliance is ON and 0 when is OFF, based on the on_power_threshold parameter; weak annotations are set to 1 when at least one time appliance is active inside the window. 
Quantity of strong and weak annotations to be used in the experiments can be defined in the experiment modules. In fact, in ukdale_experiment_1_2 and mixed_training_experiment can be set:

- quantity of data previously generated from UKDALE house 1
- quantity of data previously generated from UKDALE house 2
- quantity of data previously generated from UKDALE house 3
- quantity of data previously generated from UKDALE house 4
- quantity of data previously generated from UKDALE house 5
- strong annotations percentage
- weak annotations percentage
- control of strong quantity selected 
- clip smoothing post-processing. This flag refers only to fully supervised + weak supervised experiment 
- the use of weakly labeled dataset
- type of experiment to be performed (fully supervised or fully supervised + weak supervised) 
- path to synthetic data for ANE computation
- flag to perform train or inference. If train is selected also the prediction on the test set and metrics estimation will be performed while if inference is chosen weights of a trained model will be loaded from the path to perform the inference.  

To perform the mixed experiment, in the proposed work REFIT was resample from 8s to 6s period. Specifically, refit_resampling.py can be used for this purpose. 
For mixed training experiment there is the possibility to set also the testing dataset desired, choosing between REFIT and UK-DALE.

Required packages to prepare the enviroment are listed in environment.yml file.

Structure for the linear softmax pooling layer is inspired by https://github.com/marl/autopool.







