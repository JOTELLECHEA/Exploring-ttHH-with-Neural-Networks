# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that trains and test a Keras NN.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
# Imported packages.
import csv, sys
import uproot
import pandas as pd
import numpy as np
from numpy import array

np.set_printoptions(threshold=sys.maxsize)
import shap
import argparse
import tensorflow as tf
import tkinter
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import math
import time
from math import log, sqrt
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

status = len(tf.config.experimental.list_physical_devices("GPU"))

from sklearn.metrics import (
    precision_recall_curve,
    plot_precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.metrics import confusion_matrix
from datetime import datetime
import slug  # Library with common functions used in multiple scripts.

parser = argparse.ArgumentParser(description="number of jets")
parser.add_argument("--num", type=str, help="Use '--num=' followed by a Number of jets")
args = parser.parse_args()
# numofjets = int(args.num)
numofjets = 10

# Fixed values.
tree = "OutputTree"
seed = 42

phase = 1

branches = slug.dataCol(phase,numofjets)

# Number of features.
numBranches = len(branches) - 2

mikeHancePATH = '/data/users/mhance/tthh/'
jTellecheaPATH = '~/neural_networks/data/flat_btageff_0.77/'
# jTellecheaPATH = '~/neural_networks/data/flat_btageff_0.85/'
# Data read from file.
signal = uproot.open(jTellecheaPATH+"new_TTHH.root")[tree]
df_signal = signal.pandas.df(branches)

bkgTTBB = uproot.open(jTellecheaPATH+"new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)

bkgTTH = uproot.open(jTellecheaPATH+"new_TTH.root")[tree]
df_bkgTTH = bkgTTH.pandas.df(branches)

bkgTTZ = uproot.open(jTellecheaPATH+"new_TTZ.root")[tree]
df_bkgTTZ = bkgTTZ.pandas.df(branches)

df_background = pd.concat([df_bkgTTBB, df_bkgTTH, df_bkgTTZ])

# The 3 backgrounds are concatenated we shuffle to make sure they are not sorted.
shuffleBackground = shuffle(df_background, random_state=seed)

# Signal and shuffle background data.
rawdata = pd.concat([df_signal, shuffleBackground])

X = rawdata.drop(["weights", "truth"], axis=1)

X = slug.scaleData(X,phase)


# Signal
scalefactor = 0.00232 * 0.608791
sigw = rawdata["weights"][: len(signal)] * scalefactor
bkgw = rawdata["weights"][len(signal) :]

# Labeling data with 1's and 0's to distinguish.(1/positve/signal and 0/negative/background)
# Truth Labels.
y = np.concatenate((np.ones(len(df_signal)), np.zeros(len(shuffleBackground))))


# Shuffle full data and split into train/test and validation set.
X_dev, X_eval, y_dev, y_eval = train_test_split(
    X, y, test_size=0.01, random_state=seed, stratify=y
)
X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=seed, stratify=y_dev
)

def main(LAYER, BATCH, RATE):
    """
    NN structure ex. [5,5,5,5,1] 4 layers with 5 neurons each and one output layer. LAYER value is
    the number of hidden layers excluding the output layer. Each hidden layer will contain the same
    amount of neurons (It is hard coded to be the number of features). The BATCH is the batch size,
    powers of 2 are perfered but any positive number works. RATE is the drop out rate; so a RATE = .5
    is half of the neurons being randomly turned off.
    """
    network = []
    numEpochs =  150  # Number of times the NN gets trained.
    batchSize = BATCH
    numLayers = LAYER
    neurons = numBranches

    # This creates a list that has the stucture of the NN.
    for i in range(numLayers - 1):
        network.append(neurons)
    network.append(1)
    numNeurons = sum(network)

    # This is a conformation that the script is starting and the NN structure is displayed.
    print("Script starting....\n", network)

    # This tags the output files with either GPU or CPU.
    if status == 1:
        print("GPU")
        sufix = ".GPU"
    else:
        sufix = ".CPU"
        print("CPU")

    # Start time for file name.
    startTime = datetime.now()
    pre = time.strftime("%Y.%m.%d_") + time.strftime("%H.%M.%S.")

    # Filename for keras model to be saved as.
    h5name = (
        "numLayers"
        + str(LAYER)
        + ".numBranches"
        + str(neurons)
        + ".batchSize"
        + str(BATCH)
    )
    modelName = "data/" + pre + h5name + sufix + ".h5"

    # NN model defined as a function.
    def build_model():

        # Create a NN model. Barebones model with no layers.
        model = Sequential()

        # Best option for most NN.
        opt = keras.optimizers.Nadam()
        # opt = keras.optimizers.Adam()

        # Activation function other options possible.
        act = "relu"  # Relu is 0 for negative values, linear for nonzero values.

        # Use model.add() to add one layer at a time, 1st layer needs input shape, So we pass the 1st element of network.
        # Dense Layers are fully connected and most common.

        model.add(Dense(network[0], input_dim=numBranches))

        # Loop through and add layers (1,(n-2)) where n is the number of layers. We end at n-2 because we start at 1 not zero and
        # we  the input layer is added above with input dimension. Therefore we must remove 2 from layers.
        for i in range(1, numLayers - 2):
            model.add(Dense(network[i], activation=act))  # Hidden layers.
            # Turning off nuerons of layer above in loop with probability = 1-r, so r = 0.25, then 75% of nerouns are kept.
            model.add(Dropout(RATE, seed=seed))

        # Last layer needs to have one neuron for a binary classification(BC) which yields from 0 to 1.
        model.add(
            Dense(network[-1], activation="sigmoid")
        )  # Output layer's activation function for BC needs to be sigmoid.

        # Last step is compiling.
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=tf.keras.metrics.Precision(),
        )
        return model

    def compare_train_test(kModel, X_train, y_train, X_test, y_test, bins=30):
        """
        This creates the signal and background distrubution.
        """
        decisions = []
        for X, y in ((X_train, y_train), (X_test, y_test)):
            d1 = model.predict(X[y > 0.5]).ravel()  # signal
            d2 = model.predict(X[y < 0.5]).ravel()  # background
            decisions += [d1, d2]
        low = min(np.min(d) for d in decisions)
        high = max(np.max(d) for d in decisions)
        low_high = array([low, high])

        plt.subplot(212)
        plt.hist(
            decisions[0],
            color="r",
            alpha=0.5,
            range=low_high,
            bins=bins,
            histtype="stepfilled",
            density=False,
            label="S (train)",
        )
        plt.hist(
            decisions[1],
            color="b",
            alpha=0.5,
            range=low_high,
            bins=bins,
            histtype="stepfilled",
            density=False,
            label="B (train)",
        )

        hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="S (test)")

        hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="B (test)")
        plt.show()

    # Using model and setting parameters.
    model = build_model()

    # This checkpoint is used for recovery of trained weights incase of interuption.
    checkPointsCallBack = ModelCheckpoint("temp.h5", save_best_only=True)

    # This terminates early if the monitor does not see an improvement after a certain
    # amount of epochs given by the patience.
    earlyStopCallBack = EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )

    # This is where the training starts.
    kModel = model.fit(
        X_train,
        y_train,
        epochs=numEpochs,
        batch_size=batchSize,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[earlyStopCallBack, checkPointsCallBack],
    )
    plt.subplot(211)
    plt.plot(kModel.history['precision'])
    plt.plot(kModel.history['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('precision.jpg')
    plt.subplot(212)
    plt.plot(kModel.history['loss'])
    plt.plot(kModel.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.jpg')
    # This is the predicted score. Values range between [0,1]
    y_predicted = model.predict(X_test)

    # The score is rounded; values are 0 or 1.
    y_predicted_round = [1 * (x[0] >= 0.5) for x in y_predicted]

    # Prediction, fpr,tpr and threshold values for ROC.
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    aucroc = auc(fpr, tpr)
    precision, recall, thresRecall = precision_recall_curve(y_test, y_predicted)

    # plt.xlabel("Score")
    # plt.ylabel("Distribution")
    # plt.yscale("log")
    # plt.legend(loc="upper right")
    # plt.subplot(211)
    # plt.plot(fpr, tpr, "k-", label="All, AUC = %0.3f" % (aucroc))
    # plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic")
    # plt.legend(loc="lower right")
    # plt.grid()


    # AUC
    
    # plot1 = plt.figure(1)
    # slug.plotROC(fpr, tpr, aucroc)
    # slug.plotPR(precision,recall,thresRecall)
    # compare_train_test(kModel, X_train, y_train, X_test, y_test)

    if False:
        # This plots the important features.
        plot2 = plt.figure(2)
        backgrounds = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, backgrounds)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(
            shap_values,
            X_train,
            plot_type="bar",
            feature_names=branches[:-1],
            max_display=25,
        )

    # computes max signif
    numbins = 1000000
    allScore = model.predict(X)
    sigScore = model.predict(X[y > 0.5]).ravel()
    bkgScore = model.predict(X[y < 0.5]).ravel()
    sigSUM = len(sigScore)
    bkgSUM = len(bkgScore)

    xlimit = (0, 1)
    tp = []
    fp = []
    hist, bins = np.histogram(sigScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for i in range(numbins - 1, -1, -1):
        count += hist[i] / sigSUM
        tp.append(count)
    hist, bins = np.histogram(bkgScore, bins=numbins, range=xlimit, density=False)
    count = 0
    for j in range(numbins - 1, -1, -1):
        count += hist[j] / bkgSUM
        fp.append(count)
    area = auc(fp, tp)
    xplot = tp
    yplot = fp
    # computes max signif
    sigSUM = len(sigScore) * scalefactor
    tp = np.array(tp) * sigSUM
    fp = np.array(fp) * bkgSUM
    syst = 0.0
    stat = 0.0
    maxsignif = 0.0
    maxs = 0
    maxb = 0
    bincounter = numbins - 1
    bincountatmaxsignif = 999
    for t, f in zip(tp, fp):
        signif = slug.getZPoisson(t, f, stat, syst)
        if f >= 10 and signif > maxsignif:
            maxsignif = signif
            maxs = t
            maxb = f
            bincountatmaxsignif = bincounter
            score = bincountatmaxsignif / numbins
        bincounter -= 1
    print(
        "\n Score = %6.3f\n Signif = %5.2f\n nsig = %d\n nbkg = %d\n"
        % (score, maxsignif, maxs, maxb)
    )
    runtime = datetime.now() - startTime
    areaUnderCurve = "{:.4f}".format(aucroc)
    maxsignif = "{:5.2f}".format(maxsignif)
    average_precision = average_precision_score(y_test, y_predicted)
    avgPer = "{0:0.4f}".format(average_precision)
    score = "{0:6.3f}".format(score)
    maxs = "%10d" % (maxs)
    maxb = "%10d" % (maxb)
    # cm = confusion_matrix(y_test, y_predicted_round)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted_round,normalize='all').ravel()
    CM = '[{0:0.2f}, {1:0.2f}, {2:0.2f}, {3:0.2f}]'.format(tp,fp,fn,tn)
    modelParam = [
        "FileName",
        "ConfusionMatrix [TP FP] [FN TN]",
        "Run Time",
        "AUC",
        "Avg.P",
        "Score",
        "Max Signif",
        "nsig",
        "nbkg",
    ]
    df = pd.DataFrame(
        np.array(
            [
                [
                    modelName[5:],
                    CM,
                    runtime,
                    areaUnderCurve,
                    avgPer,
                    score,
                    maxsignif,
                    maxs,
                    maxb,
                ]
            ]
        ),
        columns=modelParam,
    )
    # df.to_csv("csv/testelep2.csv", mode="a", header=False, index=False)
    # df.to_csv("csv/highlevelvariables.csv", mode="a", header=False, index=False)
    df.to_csv("csv/aug.csv", mode="a", header=False, index=False)
    print(df.to_string(justify="left", columns=modelParam, header=True, index=False))
    print("Saving model.....")
    model.save(modelName)  # Save Model as a HDF5 filein Data folder
    print("Model Saved")
main(5,512,0)
