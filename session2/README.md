**********************************************************************************************************************
1. Model Accuracy
**********************************************************************************************************************

[0.03015573327351458, 0.9942]

**********************************************************************************************************************
2. Logs from 20 Epocs:
**********************************************************************************************************************

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 8s 130us/step - loss: 0.0308 - acc: 0.9906 - val_loss: 0.0287 - val_acc: 0.9921
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0103 - acc: 0.9963 - val_loss: 0.0289 - val_acc: 0.9915
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0076 - acc: 0.9973 - val_loss: 0.0265 - val_acc: 0.9932
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 8s 127us/step - loss: 0.0065 - acc: 0.9976 - val_loss: 0.0321 - val_acc: 0.9917
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 8s 129us/step - loss: 0.0049 - acc: 0.9983 - val_loss: 0.0321 - val_acc: 0.9926
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0044 - acc: 0.9986 - val_loss: 0.0267 - val_acc: 0.9938
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 8s 130us/step - loss: 0.0040 - acc: 0.9987 - val_loss: 0.0284 - val_acc: 0.9932
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 8s 131us/step - loss: 0.0034 - acc: 0.9988 - val_loss: 0.0299 - val_acc: 0.9937
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 8s 132us/step - loss: 0.0035 - acc: 0.9988 - val_loss: 0.0341 - val_acc: 0.9929
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0036 - acc: 0.9989 - val_loss: 0.0316 - val_acc: 0.9932
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 8s 129us/step - loss: 0.0030 - acc: 0.9990 - val_loss: 0.0323 - val_acc: 0.9929
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 8s 129us/step - loss: 0.0025 - acc: 0.9992 - val_loss: 0.0373 - val_acc: 0.9924
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0029 - acc: 0.9992 - val_loss: 0.0304 - val_acc: 0.9935
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0024 - acc: 0.9992 - val_loss: 0.0272 - val_acc: 0.9942
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0022 - acc: 0.9994 - val_loss: 0.0297 - val_acc: 0.9936
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0021 - acc: 0.9994 - val_loss: 0.0306 - val_acc: 0.9931
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 8s 130us/step - loss: 0.0026 - acc: 0.9992 - val_loss: 0.0308 - val_acc: 0.9938
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0026 - acc: 0.9991 - val_loss: 0.0305 - val_acc: 0.9934
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 8s 128us/step - loss: 0.0019 - acc: 0.9995 - val_loss: 0.0306 - val_acc: 0.9936
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 8s 130us/step - loss: 0.0015 - acc: 0.9996 - val_loss: 0.0302 - val_acc: 0.9942
<keras.callbacks.History at 0x7ff438745f60>

**********************************************************************************************************************
3. Approach
**********************************************************************************************************************

1. Balance the number of channels and accuracy based on trial and error to restrict the model within 15k params
2. Better results on bacth size of 128 against batch size of 512
3. Reduce dropout value again through trial and error to balance accuracy and overfillting 





