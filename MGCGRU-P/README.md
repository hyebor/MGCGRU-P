# Ridesplitting demand prediction via spatiotemporal multi-graph convolution network

This is an implementation of the methodology in "Ridesplitting demand prediction via spatiotemporal multi-graph convolution network". 

## To train the model

1. generate your training data: python generate_training_data.py
2. train model with script (e.g.) : ```python training.py --dataset beijing --rnn_type dcgru--cost mae --gpu_id 0  --trial_id  --ckpt_dir ./ckpt --data_dir ./data ```
3. evaluate with script: ```python test.py --dataset beijing --rnn_type dcgru--cost mae --gpu_id 0 --trial_id 0 --ckpt_dir ./ckpt --output_dir ./result```
   &nbsp;

Note that:

- the setting of ```--dataset```, ```--rnn_type```, ```--cost```, ```--trial_id``` and ```--ckpt_dir``` should be consistent in training and evaluation.
- ```--trial_id```  controls the random seed; by changing it, we can get multiple-trials of results with different random seeds.

&nbsp;


### Settings

```--rnn_type```:

- ```dcgru```

```--cost```:

- ```mae``` (for MAE and RMSE computation)
- ```nll``` (for CRPS, P10QL and P90QL computation)

### Other Settings

The adjacency matrices of the graphs of the datasets are stored in ```./data/sensor_graph``` (```--graph_dir```). 

The model configs are stored in ```./data/model_config``` (```--config_dir```)

If the folders are changed, please make the arguments of ```train.py``` and ```inference.py``` consistent.

&nbsp;
&nbsp;
&nbsp;

----



