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
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"


def NN_reg(X,y_,X_train, y_train, X_test, y_test, otherfeatures_data_train, otherfeatures_data_test,
        model_num, data_size,
        epoch_size,
        batch_size, learning_rate
        ):

    if not os.path.exists(
            'TrainedModel/reg-model{}_{}t_{}te_{}_LR{}'.format(str(model_num), str(data_size), str(epoch_size),
                                                              str(batch_size),
                                                              str(learning_rate))):
        os.makedirs('TrainedModel/reg-model{}_{}t_{}te_{}_LR{}'.format(str(model_num),
                                                                      str(data_size),
                                                                      str(epoch_size),
                                                                      str(batch_size), str(learning_rate)))

    if not os.path.exists(
            'TrainedModel/reg-model{}_{}t_{}te_{}_LR{}/plot'.format(str(model_num),
                                                                   str(data_size),
                                                                   str(epoch_size),
                                                                   str(batch_size), str(learning_rate))):
        os.makedirs('TrainedModel/reg-model{}_{}t_{}te_{}_LR{}/plot'.format(str(model_num),
                                                                           str(data_size),
                                                                           str(epoch_size),
                                                                           str(batch_size), str(learning_rate)))

    if not os.path.exists('TrainedModel/reg-model{}_{}t_{}te_{}_LR{}/model'.format(str(model_num),
                                                                                  str(data_size),
                                                                                  str(epoch_size),
                                                                                  str(batch_size),
                                                                                  str(learning_rate))):
        os.makedirs('TrainedModel/reg-model{}_{}t_{}te_{}_LR{}/model'.format(str(model_num),
                                                                            str(data_size),
                                                                            str(epoch_size),
                                                                            str(batch_size),
                                                                            str(learning_rate)))




    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    #For Conv as first layer
    X_train = np.expand_dims(X_train, axis=2)  # reshape (14400, 18500) to (14400, 18500, 1)
    X_test = np.expand_dims(X_test, axis=2)
    # print(y_train)
    # print(y_test)


    y_train=pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    # print("---------------------------y_test", y_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # print("X_train shape:",X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("y_train:", y_train)
    # print("y_test:", y_test)
    # print("X_train shape after reshape:",X_train.shape)
    # print("y_train shape after reshape:", y_train.shape)
    # print(y_train)

    try:


        model=Sequential()


        
        model.add(Conv1D(64, 7, input_shape=X_train.shape[1:]))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(64, 4, kernel_regularizer=keras.regularizers.l2(0.5)))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv1D(128, 2, kernel_regularizer=keras.regularizers.l2(0.5)))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv1D(256, 1, kernel_regularizer=keras.regularizers.l2(0.5)))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv1D(128, 1, kernel_regularizer=keras.regularizers.l2(0.5)))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Dense(16))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Dense(1, activation='linear'))  


        len_data_train=len(X_train)
        model.compile(loss=keras.losses.Huber(delta=1),
                      optimizer=keras.optimizers.adam(lr=learning_rate),
                      metrics=['mae'])


        history=model.fit(X_train, y_train, epochs=epoch_size, batch_size=batch_size, validation_data=(X_test, y_test)) 
        model.save("TrainedModel/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+
                   "te_"+str(batch_size)+"_LR"+str(learning_rate)+"/model/Dust_NN_.h5")




#############################plot history---------------------------------------------------------------------------------------------
        # summarize history for accuracy
        plt.figure()
        plt.plot(np.log10(history.history['mae']))
        plt.plot(np.log10(history.history['val_mae']))
        plt.title('log mean squared error')
        plt.ylabel('log_mae')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("TrainedModel/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+
                   "te_"+str(batch_size)+"_LR"+str(learning_rate)+"/plot/mae.png")
        # summarize history for loss
        plt.figure()
        plt.plot(np.log10(history.history['loss']))
        plt.plot(np.log10(history.history['val_loss']))
        plt.title('loss')
        plt.ylabel('log_loss')
        plt.xlabel('epoch')
        # plt.yscale('log')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("TrainedModel/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+
                   "te_"+str(batch_size)+"_LR"+str(learning_rate)+"/plot/log_loss.png")


        plt.figure()
        plt.plot(np.log10(history.history['loss']))
        plt.plot(np.log10(history.history['val_loss']))
        plt.title('loss')
        plt.ylabel('log_loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("TrainedModel/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+
                   "te_"+str(batch_size)+"_LR"+str(learning_rate)+"/plot/logscale_log_loss.png")
        print("after fitting")

#############################--------------------------------------------------------------------------------------------------------------

        # Evaluate
        loss_test, accuracy_test = model.evaluate(X_test, y_test)
        print('Accuracy', accuracy_test)

        loss_train, accuracy_train = model.evaluate(X_train, y_train)


        # Prediction
        prediction = model.predict(X_test)
        train_pred= model.predict(X_train)

        print("prediction----------", prediction)
        print("y_test----------", y_test)


        prediction=[ item for elem in prediction for item in elem]#[i for i in prediction[:]]
        train_pred = [item for elem in train_pred for item in elem]

        y_test = [item for elem in y_test for item in elem]  # [i for i in prediction[:]]
        y_train = [item for elem in y_train for item in elem]


        y_test=np.array(y_test)
        y_train=np.array(y_train)



##############saving predictions--------------------------------------------------------------------------------------------------------------
        result=pd.DataFrame(prediction)
        result["y_test"]=y_test
        result.to_pickle("TrainedModel/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+"te_"+str(batch_size)+"_LR"
                         +str(learning_rate)+"/plot/result.pkl")

        result_train=pd.DataFrame(train_pred)
        result_train["y_train"]=y_train
        result_train.to_pickle("TrainedModel/reg-model"+str(model_num)+ "_"+str(data_size)+"t_"+str(epoch_size)+"te_"+str(batch_size)+
                               "_LR"+str(learning_rate)+"/plot/result_train.pkl")












##############plotting---------------------------------------------------------------------------------------------------------------------------



        # #________
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        plt.scatter(y_test, prediction, label='MassDust', s=3)
        # plt.plot((y_test[3]) * 1e-4, (prediction[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_test) + "for "+str(len(y_test))+" objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test_actualvspredMassDust.png")


        #####------
        plt.figure(figsize=(12, 10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        plt.scatter(y_test, prediction, label='MassDust', s=3)
        # plt.plot((y_test[3]) * 1e-4, (prediction[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.xscale('log')
        plt.yscale('log')
        plt.xscale('symlog', linthreshx=1e-1)
        plt.yscale('symlog', linthreshy=1e-1)
        plt.legend()
        plt.title("loss is:" + str(loss_test) + "for " + str(len(y_test)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test_Log_actualvspredMassDust.png")
        ###----

        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', c=otherfeatures_data_test["tempSN"])  # , size=3)
        sns.scatterplot(y_test, prediction,  hue=otherfeatures_data_test["tempSN"])

        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test_actualvspredMassDust_tempSN.png")

        ####----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', c=otherfeatures_data_test["tempDust"])  # , size=3)
        sns.scatterplot(y_test, prediction, hue=otherfeatures_data_test["tempDust"])
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test_actualvspredMassDust_tempDust.png")

        ####----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', c=otherfeatures_data_test["radiusSN"])  # , size=3)
        sns.scatterplot(y_test, prediction, hue=otherfeatures_data_test["radiusSN"])
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test_actualvspredMassDust_radiusSN.png")



        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        my_cmap = plt.cm.jet
        my_cmap.set_under('w', 1)
        y_test_=y_test#.ravel()#np.array(y_test)
        prediction_=prediction#.ravel()#np.array(prediction)
        print(y_test_.shape)


        plt.hist2d(y_test_,prediction_ , 500, cmap=my_cmap, cmin=1)  # , size=3)
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        plt.colorbar()
        # plt.plot((y_test[3]) * 1e-4, (prediction[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0, 2)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        plt.title("loss is:" + str(loss_test) + "for "+str(len(y_test))+" objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/test-hist_actualvspredMassDust.png")
        #
     ###---------train
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        plt.scatter(y_train, train_pred,label='MassDust', s=3)#, size=3)
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+"te_"+str(batch_size)+"_LR"
                    +str(learning_rate)+"/plot/train_actualvspredMassDust.png")

        #----

        plt.figure(figsize=(12, 10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        plt.scatter(y_train, train_pred, label='MassDust', s=3)  # , size=3)
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xscale('symlog', linthreshx=1e-1)
        plt.yscale('symlog', linthreshy=1e-1)
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/train_Log_actualvspredMassDust.png")




        ####-----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', color=otherfeatures_data_train["tempSN"])  # , size=3)
        sns.scatterplot(y_train, train_pred, hue=otherfeatures_data_train["tempSN"])
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/train_actualvspredMassDust_tempSN.png")


        ####----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', color=otherfeatures_data_train["tempDust"])  # , size=3)
        sns.scatterplot(y_train, train_pred, hue=otherfeatures_data_train["tempDust"])
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/train_actualvspredMassDust_tempDust.png")

        ####----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        # plt.scatter(y_train, train_pred, label='MassDust', color=otherfeatures_data_train["radiusSN"])  # , size=3)
        sns.scatterplot(y_train, train_pred, hue=otherfeatures_data_train["radiusSN"])
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model" + str(model_num) + "_" + str(data_size) + "t_" + str(epoch_size) + "te_" + str(
            batch_size) + "_LR"
                    + str(learning_rate) +  "/plot/train_actualvspredMassDust_radiusSN.png")

        ####----
        plt.figure(figsize=(12,10))
        plt.plot([0, 100], [0, 100], ls="--", c=".3")
        my_cmap = plt.cm.jet
        my_cmap.set_under('w', 1)
        y_train_=y_train#.ravel()
        train_pred_= train_pred#.ravel()

        plt.hist2d(y_train_,train_pred_,500, cmap=my_cmap, cmin=1)#, size=3)
        plt.colorbar()
        # plt.plot((y_train[3]) * 1e-4, (train_pred[3]) * 1e-4, label='MassDust')

        plt.xlabel(r"$M_{Dust} (10^{-4} M_{sun})$")
        plt.ylabel(r"Pred $M_{Dust} (10^{-4} M_{sun})$")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        # plt.ylim(0,2)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.legend()
        plt.title("loss is:" + str(loss_train) + "for " + str(len(y_train)) + " objects")
        plt.savefig("dataPhoto-cl+reg/reg-model"+str(model_num)+"_"+str(data_size)+"t_"+str(epoch_size)+"te_"+str(batch_size)+"_LR"
                    +str(learning_rate)+"/plot/train-hist_actualvspredMassDust.png")




    except Exception as e:
        print("no profile", e)


