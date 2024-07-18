
settings--> Epoch: 40 || Batch_size: 16 || Sequence len.: 500
Test Loss(mse): 0.06370624899864197
Test MAE: 0.25212547183036804
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 500, 100)          30900     
                                                                 
 dropout (Dropout)           (None, 500, 100)          0         
                                                                 
 gru_1 (GRU)                 (None, 500, 50)           22800     
                                                                 
 dropout_1 (Dropout)         (None, 500, 50)           0         
                                                                 
 gru_2 (GRU)                 (None, 50)                15300     
                                                                 
 dropout_2 (Dropout)         (None, 50)                0         
                                                                 
 dense (Dense)               (None, 1)                 51        
                                                                 
=================================================================
Total params: 69,051
Trainable params: 69,051
Non-trainable params: 0
_________________________________________________________________

graph1(loss vs epoch):
![gru_1_loss](https://github.com/user-attachments/assets/07d55da9-68ac-4890-9481-6a45a5f56c54)
