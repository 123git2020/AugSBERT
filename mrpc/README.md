# mrpc models 
It contains three folders ouput by [train_mrpc_bm25.py](../train_mrpc_bm25.py):  
The folder of cross-encoder that trains on gold samples and predict silver samples.    
The folder of bi-encoder that trains on augmented data composed of gold and silver samples.    
The folder of bi-encoder that only trains on gold samples.  <br>

The data.tsv is the cleaned original mrpc data.

When you want to retrain the model, please delete all the folders in this directory but not data.

