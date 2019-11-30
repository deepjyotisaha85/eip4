**********************************************************************************************************************
1. Base Network Validation Accuracy:
**********************************************************************************************************************

Accuracy on test data is: 78.49

**********************************************************************************************************************
2. Model Definition:
**********************************************************************************************************************


model.add(SeparableConv2D(128, (3,3) , strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3))) 
#O/P: 32, RF: 3 

model.add(BatchNormalization())

model.add(SeparableConv2D(128, (3,3) , strides=(1, 1), padding='same', activation='relu'))
#O/P: 32, RF: 5 

model.add(BatchNormalization())

model.add(Dropout(0.20))

model.add(SeparableConv2D(64, (2,2) , strides=(2, 2), activation='relu'))   
#O/P: 16, RF: 6

model.add(BatchNormalization())

model.add(SeparableConv2D(128, (3,3) , strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3))) 
#O/P: 16, RF: 10 

model.add(BatchNormalization())

model.add(SeparableConv2D(128, (3,3) , strides=(1, 1), padding='same', activation='relu')) 
#O/P: 16, RF: 14 

model.add(BatchNormalization())

model.add(Dropout(0.20))

model.add(SeparableConv2D(64, (2,2) , strides=(2, 2), activation='relu'))  
#O/P: 8, RF: 16 

model.add(BatchNormalization())

model.add(SeparableConv2D(64, (3,3) , strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3))) 
#O/P: 8, RF: 24 

model.add(BatchNormalization())

model.add(SeparableConv2D(128, (3,3) , strides=(1, 1), padding='same', activation='relu')) 
#O/P: 8, RF: 32 

model.add(BatchNormalization())

model.add(SeparableConv2D(10, (1,1) , strides=(1, 1), activation='relu')) 
model.add(BatchNormalization())
model.add(SeparableConv2D(10, (8,8) , strides=(1, 1), activation='relu'))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))

**********************************************************************************************************************
3. 50 epoch logs:
**********************************************************************************************************************


Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
  2/390 [..............................] - ETA: 31s - loss: 0.4573 - acc: 0.8594
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, callbacks=[<keras.ca..., steps_per_epoch=390, epochs=50)`
  if sys.path[0] == '':
390/390 [==============================] - 24s 61ms/step - loss: 0.6508 - acc: 0.7718 - val_loss: 0.9117 - val_acc: 0.7134
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 24s 62ms/step - loss: 0.5566 - acc: 0.8046 - val_loss: 0.7891 - val_acc: 0.7429
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 24s 61ms/step - loss: 0.4979 - acc: 0.8252 - val_loss: 0.6538 - val_acc: 0.7839
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 24s 61ms/step - loss: 0.4518 - acc: 0.8417 - val_loss: 0.6670 - val_acc: 0.7855
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 24s 61ms/step - loss: 0.4132 - acc: 0.8545 - val_loss: 0.6438 - val_acc: 0.7931
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 24s 61ms/step - loss: 0.3795 - acc: 0.8670 - val_loss: 0.6291 - val_acc: 0.7996
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 24s 61ms/step - loss: 0.3485 - acc: 0.8764 - val_loss: 0.6779 - val_acc: 0.7887
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 24s 61ms/step - loss: 0.3244 - acc: 0.8867 - val_loss: 0.6271 - val_acc: 0.8034
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 24s 61ms/step - loss: 0.3032 - acc: 0.8923 - val_loss: 0.6363 - val_acc: 0.8093
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 25s 63ms/step - loss: 0.2791 - acc: 0.9000 - val_loss: 0.6485 - val_acc: 0.8063
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 25s 63ms/step - loss: 0.2634 - acc: 0.9063 - val_loss: 0.6694 - val_acc: 0.8060
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 24s 62ms/step - loss: 0.2501 - acc: 0.9106 - val_loss: 0.6431 - val_acc: 0.8159
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 24s 61ms/step - loss: 0.2314 - acc: 0.9179 - val_loss: 0.6822 - val_acc: 0.8132
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 24s 61ms/step - loss: 0.2217 - acc: 0.9211 - val_loss: 0.6949 - val_acc: 0.8133
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 24s 61ms/step - loss: 0.2102 - acc: 0.9255 - val_loss: 0.6950 - val_acc: 0.8109
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 24s 61ms/step - loss: 0.2006 - acc: 0.9286 - val_loss: 0.7095 - val_acc: 0.8094
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 24s 61ms/step - loss: 0.1898 - acc: 0.9314 - val_loss: 0.7223 - val_acc: 0.8091
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 24s 61ms/step - loss: 0.1845 - acc: 0.9339 - val_loss: 0.7237 - val_acc: 0.8145
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 24s 61ms/step - loss: 0.1765 - acc: 0.9365 - val_loss: 0.7290 - val_acc: 0.8125
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 24s 61ms/step - loss: 0.1687 - acc: 0.9394 - val_loss: 0.7401 - val_acc: 0.8141
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 24s 61ms/step - loss: 0.1613 - acc: 0.9403 - val_loss: 0.7582 - val_acc: 0.8153
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 24s 61ms/step - loss: 0.1536 - acc: 0.9439 - val_loss: 0.7643 - val_acc: 0.8136
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 24s 61ms/step - loss: 0.1474 - acc: 0.9468 - val_loss: 0.7786 - val_acc: 0.8123
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 24s 61ms/step - loss: 0.1452 - acc: 0.9482 - val_loss: 0.7870 - val_acc: 0.8143
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 24s 61ms/step - loss: 0.1373 - acc: 0.9505 - val_loss: 0.8006 - val_acc: 0.8148
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 24s 61ms/step - loss: 0.1387 - acc: 0.9503 - val_loss: 0.8018 - val_acc: 0.8171
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 24s 62ms/step - loss: 0.1297 - acc: 0.9530 - val_loss: 0.8153 - val_acc: 0.8132
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 24s 62ms/step - loss: 0.1302 - acc: 0.9520 - val_loss: 0.8194 - val_acc: 0.8118
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 25s 63ms/step - loss: 0.1235 - acc: 0.9557 - val_loss: 0.8364 - val_acc: 0.8114
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 24s 62ms/step - loss: 0.1203 - acc: 0.9574 - val_loss: 0.8239 - val_acc: 0.8163
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 24s 61ms/step - loss: 0.1220 - acc: 0.9550 - val_loss: 0.8316 - val_acc: 0.8129
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 24s 61ms/step - loss: 0.1147 - acc: 0.9581 - val_loss: 0.8524 - val_acc: 0.8156
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 24s 61ms/step - loss: 0.1125 - acc: 0.9584 - val_loss: 0.8546 - val_acc: 0.8140
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 24s 61ms/step - loss: 0.1092 - acc: 0.9601 - val_loss: 0.8747 - val_acc: 0.8190
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 24s 61ms/step - loss: 0.1045 - acc: 0.9624 - val_loss: 0.8789 - val_acc: 0.8159
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 24s 62ms/step - loss: 0.1054 - acc: 0.9614 - val_loss: 0.8752 - val_acc: 0.8149
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 24s 61ms/step - loss: 0.1037 - acc: 0.9631 - val_loss: 0.8859 - val_acc: 0.8155
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 24s 61ms/step - loss: 0.1008 - acc: 0.9637 - val_loss: 0.8993 - val_acc: 0.8161
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 24s 61ms/step - loss: 0.0973 - acc: 0.9650 - val_loss: 0.8954 - val_acc: 0.8187
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 24s 61ms/step - loss: 0.0977 - acc: 0.9660 - val_loss: 0.9135 - val_acc: 0.8180
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 24s 61ms/step - loss: 0.0937 - acc: 0.9669 - val_loss: 0.9047 - val_acc: 0.8184
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 24s 61ms/step - loss: 0.0917 - acc: 0.9669 - val_loss: 0.9165 - val_acc: 0.8166
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 24s 61ms/step - loss: 0.0905 - acc: 0.9671 - val_loss: 0.9190 - val_acc: 0.8174
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 24s 61ms/step - loss: 0.0894 - acc: 0.9678 - val_loss: 0.9246 - val_acc: 0.8170
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 25s 63ms/step - loss: 0.0876 - acc: 0.9692 - val_loss: 0.9200 - val_acc: 0.8162
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 25s 65ms/step - loss: 0.0812 - acc: 0.9707 - val_loss: 0.9492 - val_acc: 0.8159
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 24s 62ms/step - loss: 0.0814 - acc: 0.9710 - val_loss: 0.9322 - val_acc: 0.8191
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 24s 62ms/step - loss: 0.0802 - acc: 0.9713 - val_loss: 0.9523 - val_acc: 0.8177
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 24s 62ms/step - loss: 0.0833 - acc: 0.9694 - val_loss: 0.9573 - val_acc: 0.8171
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.000180386.
390/390 [==============================] - 24s 62ms/step - loss: 0.0821 - acc: 0.9700 - val_loss: 0.9600 - val_acc: 0.8182
Model took 1198.47 seconds to train


Accuracy on test data is: 81.82







