import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, MaxPool1D,MaxPool2D, Conv2D, Dropout,\
    BatchNormalization, Flatten, Conv2D, AveragePooling1D, MaxPooling2D, GlobalMaxPooling2D, Conv1D, Reshape, MaxPooling1D,\
    PReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.utils import to_categorical
from keras import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.initializers import Constant


import numpy as np
import pandas as pd
import argparse
import os
import seaborn as sns

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6"


def NN_cl(X_data,y_data,X_train, y_train, X_test, y_test,
    otherfeatures_data_train, otherfeatures_data_test,
    model_num, data_size, epoch_size,batch_size, learning_rate):
    if not os.path.exists(
            'TrainedModel/cl-model{}_{}t_{}te_{}_LR{}'.format(str(model_num), str(data_size), str(epoch_size),
                                                              str(batch_size),
                                                              str(learning_rate))):
        os.makedirs('TrainedModel/cl-model{}_{}t_{}te_{}_LR{}'.format(str(model_num),
                                                                      str(data_size),
                                                                      str(epoch_size),
                                                                      str(batch_size), str(learning_rate)))

    if not os.path.exists(
            'TrainedModel/cl-model{}_{}t_{}te_{}_LR{}/plot'.format(str(model_num),
                                                                   str(data_size),
                                                                   str(epoch_size),
                                                                   str(batch_size), str(learning_rate))):
        os.makedirs('TrainedModel/cl-model{}_{}t_{}te_{}_LR{}/plot'.format(str(model_num),
                                                                           str(data_size),
                                                                           str(epoch_size),
                                                                           str(batch_size), str(learning_rate)))

    if not os.path.exists('TrainedModel/cl-model{}_{}t_{}te_{}_LR{}/model'.format(str(model_num),
                                                                                  str(data_size),
                                                                                  str(epoch_size),
                                                                                  str(batch_size),
                                                                                  str(learning_rate))):
        os.makedirs('TrainedModel/cl-model{}_{}t_{}te_{}_LR{}/model'.format(str(model_num),
                                                                            str(data_size),
                                                                            str(epoch_size),
                                                                            str(batch_size),
                                                                            str(learning_rate)))

    X_data_index=X_data.index


    X_train = np.expand_dims(X_train, axis=2)  # reshape (14400, 18500) to (14400, 18500, 1)
    X_test = np.expand_dims(X_test, axis=2)
    X_data = np.expand_dims(X_data, axis=2)
    y_train=pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    y_data = pd.DataFrame(y_data)
    print("---------------------------y_test", y_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_data = np.array(y_data)




    ####model ZOZZ+ before-overfitting+conv3-dense+allprelu
    model=Sequential()
    model.add(Conv1D(16, 9, activation='relu', input_shape=X_train.shape[1:]))  # X_train.shape[1:] ))#
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.adam(learning_rate=learning_rate), 
                  metrics=['accuracy'])



    history = model.fit(X_train, y_train, epochs=epoch_size, batch_size=batch_size,
                        validation_data=(X_test, y_test))  # steps_per_epoch=10,
    model.save("TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) +
               "te_" + str(batch_size) + "_LR" + str(learning_rate) + "/model/Dust_NN_.h5")



    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) +
                "te_" + str(batch_size) + "_LR" + str(learning_rate) + "/plot/loss.png")
    print("after fitting")

    # Evaluate
    loss_test, accuracy_test = model.evaluate(X_test, y_test)
    print('Accuracy', accuracy_test)

    loss_train, accuracy_train = model.evaluate(X_train, y_train)

    # Prediction
    prediction = model.predict(X_test)
    train_pred = model.predict(X_train)

    total_pred=model.predict(X_data)
    # df_tot=pd.DataFrame(columns={"cl_pred": total_pred,
    #                              "index": X_data_index})
    df_tot=pd.DataFrame(X_data_index)
    df_tot["cl_pred"]= total_pred
    df_tot.to_pickle("TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) +
               "te_" + str(batch_size) + "_LR" + str(learning_rate) + "cl_pred.pkl")


    prediction = [item for elem in prediction for item in elem]  # [i for i in prediction[:]]
    train_pred = [item for elem in train_pred for item in elem]

    y_test = [item for elem in y_test for item in elem]  # [i for i in prediction[:]]
    y_train = [item for elem in y_train for item in elem]


    y_test = np.array(y_test)
    y_train = np.array(y_train)

    #######---plot test----------------------------------------------------
    plt.figure()
    plt.scatter(y_test, prediction, s=3)
    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test))  # +
    plt.xlim(-1, 2)
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(batch_size) + "_LR"
        + str(learning_rate) +  "/plot/test_actualvspred.png")

    ####-------------

    plt.figure()
    plt.scatter(y_test, prediction, s=3)
    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test))  # +
    plt.xlim(-1, 2)
    plt.yscale('log')
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
        + str(learning_rate) + "/plot/test_log_actualvspred.png")
 
    #---------------
    plt.figure()

    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test)) 
    sns.violinplot(x=y_test, y=prediction) 
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
        + str(learning_rate) + "/plot/test_violin_actualvspred.png")

    #######-------
    plt.figure()

    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test))  
    sns.violinplot(x=y_test, y=prediction) 
    plt.yscale('log')
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
        + str(learning_rate) + "/plot/test_log_violin_actualvspred.png")

    #####plot train----------------------------------------------------------------------

    plt.figure()
    plt.scatter(y_train, train_pred, s=3)
    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_train)) 
    plt.xlim(-1, 2)
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(batch_size) + "_LR"
        + str(learning_rate)  + "/plot/train_actualvspred.png")


    ####---
    plt.figure()
    plt.scatter(y_train, train_pred, s=3)
    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_train))  
    plt.xlim(-1, 2)
    plt.yscale('log')
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
        + str(learning_rate) + "/plot/train_log_actualvspred.png")


    #######----
    plt.figure()

    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test)) 
    sns.violinplot(x=y_train, y=train_pred)
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(batch_size) + "_LR"
        + str(learning_rate) +  "/plot/train_violin_actualvspred.png")

    #######----
    plt.figure()

    plt.xlabel("actual value")
    plt.ylabel("predicted value")
    plt.title("Accuracy is:" + str(accuracy_test))  
    sns.violinplot(x=y_train, y=train_pred)  
    plt.yscale('log')
    plt.savefig(
        "TrainedModel/cl-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
        + str(learning_rate) + "/plot/train_log_violin_actualvspred.png")
