# forCar
time-series-forecasting &amp; anomaly_detection
## Installation
```
git clone https://github.com/KNUAI/forCar.git
```
```
cd forCar && pip install -r requirements.txt
```

If you want to forecaste, put your data in datasets/forecasting/  
ex)
![image](https://user-images.githubusercontent.com/86586602/169826930-85e57487-62e3-4998-a472-beb7a288188b.png)

If you want to anomaly detection, put your data in datasets/Anomaly_Detection/  
ex)
![image](https://user-images.githubusercontent.com/86586602/169827535-a0871177-bc63-4d19-9b7a-f1e82ece500b.png)
**Notice**
anomaly detection dataset must have 'MaterialID', 'is_test', 'target' columns

## Usage
**If you want to run forecasting**
```
python forecasting.py --model SCINet --data ETTh1.csv --cols HUFL HULL MUFL MULL LUFL LULL OT --seq_len 96 --pred_len 48
```
Write the model you want to use in model!
Write the file_name you want to use in data!  
Write the variables you want to use in cols!  
Write the length you want to input in seq_len!  
Write the length you want to predict in pred_len!  
**If you want to run anomaly detection**
```
python FD.py --data D1.csv --cols feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15
```
Write the file_name you want to use in data!  
Write the variables you want to use in cols!  

**The detailed descriptions for forecasting**
| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment. This can be set to `SCINet`, `Informer`, `LSTM` |
| data           | The data_file_name                                             |
| inverse | Whether to inverse output data, using this argument means inversing output data (defaults to `True`) |
| gpu | The gpu no, used for training and inference (defaults to 0) |
| seq_len | Input sequence length of Informer encoder (defaults to 96) |
| pred_len | Prediction sequence length (defaults to 48) |
| cols | Certain cols from the data files as the input features |
| train_epochs | Train epochs (defaults to 100) |
| batch_size | The batch size of training input data (defaults to 32) |
| patience | Early stopping patience (defaults to 5) |
| learning_rate | Optimizer learning rate (defaults to 0.0001) |
| loss | Loss function (defaults to `mae`) |
| lradj | Ways to adjust the learning rate (defaults to `type1`) |
| evaluate | Evaluate the trained model |
| hidden_size | N_channel of module (defaults to 4) |
| kernel | Window_size: 3, 5, 7 (defaults to 5) |
| dropout | Dropout (defaults to 0.5) |
| num_decoder_layer | Evaluate the trained model (defaults to 1) |

**The detailed descriptions for anomaly detection**
| Parameter name | Description of parameter |
| --- | --- |
| data           | The data_file_name                                             |
| cols | Certain cols from the data files as the input features |
| fold | 5-fold: 1, 2, 3, 4, 5 (defaults to 5) |
| latent_size | Dimension of latent vector |
| threshold_rate | Threshold_rate (defaults to 5) |
| n_layer | n_layers of rnn model |
| epoch | Train epochs (defaults to 200) |
| batch_size | The batch size of training input data (defaults to 32) |
| lr | Optimizer learning rate (defaults to 0.0001) |
| r_model | RNN model: `LSTM`, `GRU` (defaults to `LSTM`) |
| evaluate | Evaluate the trained model |
| patience | Early stopping patience (defaults to 3) |
| gpus | The gpu no, used for training and inference (defaults to 0) |




