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
python forecasting.py --data ETTh1.csv --cols HUFL HULL MUFL MULL LUFL LULL OT --seq_len 96 --pred_len 48
```
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
