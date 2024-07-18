settings--> Epoch: 40 || Batch_size: 16 || Sequence len.: 500
Test Loss(mse): 0.05843295529484749
Test MAE: 0.24145260453224182
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 500, 100)          40800     
                                                                 
 dropout (Dropout)           (None, 500, 100)          0         
                                                                 
 lstm_1 (LSTM)               (None, 500, 50)           30200     
                                                                 
 dropout_1 (Dropout)         (None, 500, 50)           0         
                                                                 
 lstm_2 (LSTM)               (None, 50)                20200     
                                                                 
 dropout_2 (Dropout)         (None, 50)                0         
                                                                 
 dense (Dense)               (None, 1)                 51        
                                                                 
=================================================================
Total params: 91,251
Trainable params: 91,251
Non-trainable params: 0
_________________________________________________________________

graph1(loss vs epoch):![lstm_1_loss](https://github.com/user-attachments/assets/71f92254-9bea-40d1-ba79-8128f899c098)
