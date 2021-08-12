# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that has frequently used functions.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
# Imported packages.
#import tkinter as tk
import math
import matplotlib
import numpy as np
import pandas as pd
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
# Fixed values.
tree = "OutputTree"
seed = 42

# These are the names of the High level features that are stored as branches in the ROOT files.
HighLevel = [
    "numjet",
    "numlep",
    "btag",
    "srap",
    "cent",
    "m_bb",
    "h_b",
    # "mt1",
    # "mt2",
    # "mt3",
    'met',
    'metPhi',
    "dr1",
    "dr2",
    # "dr3",
]

def dataCol(phase,numberofjets):
    ''' 
    List of names of features. 

    Three Phases : 
    1. High Level Variables only.
    2. Lepton and Jet Variables only.
    3. 1 AND 2. 
    
    The number of jets range from 0 to 21.

    '''
    types = ["flav", "pT", "eta", "phi", "btag"]
    LeptonVAR = []
    JetVAR = []
    for i in range(4):
        for j in range(2):
            LeptonVAR.append("lepton" + str(j + 1) + types[i])
    for i in range(1, 5):
        for j in range(numberofjets):
            JetVAR.append("jet" + str(j + 1) + types[i])
    if phase == 1:
        branches = sorted(HighLevel) + ["weights", "truth"]
    elif phase == 2:
        branches = sorted(LeptonVAR + JetVAR) + ["weights", "truth"]
    elif phase == 3:
        if numberofjets == 0:
            branches = sorted(HighLevel + LeptonVAR) + ["weights", "truth"]
        elif numberofjets > 0:
            branches = sorted(HighLevel + JetVAR + LeptonVAR) + ["weights", "truth"]

    return branches

def scaleData(data,phase):
    '''
    Scales data to have 0 mean with a variance of unity.
    
    Returns data scaled and saves stats as pxl to unscale data.

    '''
    scaler = StandardScaler()
    scaler.fit(data)
    sData = scaler.transform(data)
    joblib.dump(scaler,'phase' + str(phase) + '-Scaler.pxl')
    print('Saved scaler!')
    return sData

def loadScaledData(data,phase):
    '''
    UnScales data.
    
    Returns data and loads stats from pxl file.

    '''
    scaler = joblib.load('phase' + str(phase) + '-Scaler.pxl')
    sData = scaler.transform(data)
    print('Loaded scaler and transformed data')
    return sData


def plotPR(x, y, t):
    plt.subplot(411)
    plt.plot(t, x[:-1], "b--", label="Precision")
    plt.plot(t, y[:-1], "g-", label="Recall")
    plt.ylim([0.00, 1.05])
    plt.xlabel("Threshold")
    plt.title("Precision/Recall vs. Threshold Curve")
    plt.legend(loc="lower right")
    plt.grid()


def plotROC(x, y, AUC):
    plt.subplot(412)
    plt.plot(x, y, lw=1, label="ROC (area = %0.6f)" % (AUC))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()

def confusedMatrix(Matrix):
    label = ['Signal','Background']
    df_CM = pd.DataFrame(Matrix, index=label, columns=label)
    sn.heatmap(df_CM,cmap='Blues',annot=True)
    plt.title('Confusion matrix, with normalization')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def getZPoisson(s, b, stat, syst):
    """
    The significance for optimisation.

    s: total number of signal events
    b: total number of background events
    stat: relative MC stat uncertainty for the total bkg.
    syst: relative syst uncertainty on background

    Note that the function already accounts for the sqrt(b)
    uncertainty from the data, so we only need to pass in additional
    stat and syst terms.  e.g. the stat term above is really only
    characterizing the uncertainty due to limited MC statistics used
    to estimate the background yield.
    """
    n = s + b

    # this is a relative uncertainty
    sigma = math.sqrt(stat ** 2 + syst ** 2)

    # turn into the total uncertainty
    sigma = sigma * b

    if s <= 0 or b <= 0:
        return 0

    factor1 = 0
    factor2 = 0

    if sigma < 0.01:
        # In the limit where the total BG uncertainty is zero,
        # this reduces to approximately s/sqrt(b)
        factor1 = n * math.log((n / b))
        factor2 = n - b
    else:
        factor1 = n * math.log((n * (b + sigma ** 2)) / ((b ** 2) + n * sigma ** 2))
        factor2 = ((b ** 2) / (sigma ** 2)) * math.log(
            1 + ((sigma ** 2) * (n - b)) / (b * (b + sigma ** 2))
        )

    signif = 0
    try:
        signif = math.sqrt(2 * (factor1 - factor2))
    except ValueError:
        signif = 0

    return signif
