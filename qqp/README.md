# qqp models 
It contains one folders ouput by [train_sts_qqp_indomain.py](../train_sts_qqp_indomain.py):     
The folder of bi-encoder that only trains on qqp train data.  

If we run [train_sts_qqp_crossdomain.py](../train_sts_qqp_crossdomain.py), there will be:  
The folder of cross-encoder that trains on stsb data and predict a certain fraction of qqp train data.     
The folder of bi-encoder that trains on the predicted qqp train data.<br>

The classification folder contains the qqp data.
