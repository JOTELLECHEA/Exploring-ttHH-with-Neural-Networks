# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that loads NN weights and makes 1D plots that apply NN score for a cut.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################\
import uproot
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
import matplotlib
import slug
import datetime
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import array
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix,roc_curve

sc = StandardScaler()
seed = 42
tree = "OutputTree"
phase = 3



branches = slug.dataCol(phase)
numBranches = len(branches)

parser = argparse.ArgumentParser(description="Plot 1D plots of sig/bac")
parser.add_argument("--file", type=str, help="Use '--file=' followed by a *.h5 file")
args = parser.parse_args()
file = "data/" + str(args.file)

# Data read from file.
signal = uproot.open("data/new_TTHH.root")[tree]
df_signal = signal.pandas.df(branches)

bkgTTBB = uproot.open("data/new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)

bkgTTH = uproot.open("data/new_TTH.root")[tree]
df_bkgTTH = bkgTTH.pandas.df(branches)

bkgTTZ = uproot.open("data/new_TTZ.root")[tree]
df_bkgTTZ = bkgTTZ.pandas.df(branches)

df_background = pd.concat([df_bkgTTBB, df_bkgTTH, df_bkgTTZ])

# The 3 backgrounds are concatenated we shuffle to make sure they are not sorted.
shuffleBackground = shuffle(df_background, random_state=seed)


# signal and limited shuffle background data to counter inbalanced data problem.
rawdata = pd.concat([df_signal, shuffleBackground])

X = rawdata.drop(["weights", "truth"], axis=1)

X = sc.fit_transform(X)

# signal
scalefactor = 0.00232 * 0.608791
sigw = rawdata["weights"][: len(signal)] * scalefactor
bkgw = rawdata["weights"][len(signal) :]

# Labeling data with 1's and 0's to distinguish.
y = np.concatenate((np.ones(len(df_signal)), np.zeros(len(shuffleBackground))))

# Shuffle full data and split into train/test and validation set.
X_dev, X_eval, y_dev, y_eval = train_test_split(
    X, y, test_size=0.001, random_state=seed
)
X_train, X_test, y_train, y_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=seed
)

neuralNet = keras.models.load_model(file)

allScore = neuralNet.predict(X).ravel()
fpr, tpr, thresholds = roc_curve(y, allScore)
area = auc(fpr, tpr)

scoresig=[]
score1=[]
score2=[]
score3=[]
if False:
    for i in range(len(X)):
        if i < len(signal):
            scoresig.append(allScore[i])
        else:
            if rawdata['truth'].values[i] == 1:
                score1.append(allScore[i])
            if rawdata['truth'].values[i] == 2:
                score2.append(allScore[i])
            if rawdata['truth'].values[i] == 3:
                score3.append(allScore[i])
    plt.hist(
            [scoresig,score1,score2,score3],
            bins=50,
            histtype="stepfilled",
            label=['Signal',"TTH","TTZ","TTBB"],
            linestyle="solid",
            color=['red','blue','mediumorchid','green'],
            stacked=True,
            # weights=[NNb3weights,NNb2weights,NNb1weights],
        )
    # plt.hist(
    #     allScore,
    #     color="k",
    #     bins=50,
    #     histtype="stepfilled",
    #     label='Score Distribution (Test Data)',
    # )
    plt.ylabel('Events')
    plt.xlabel('Score')
    plt.yscale("log")
    plt.legend()
    plt.show()
# The score is rounded; values are 0 or 1.
# y_pred = [1 * (x[0] >= 0.5) for x in allScore]

if False:
    bins = numbins = 30
    decisions = []

    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = neuralNet.predict(X[y>0.5]).ravel()
        d2 = neuralNet.predict(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    xlimit = array([low,high])
    hist, bins = np.histogram(decisions[3], bins=numbins, range=xlimit, density=True)
    plt.figure(figsize=(8, 6))
    plt.subplot(212)
    plt.hist(
        decisions[0],
        color="r",
        alpha=0.5,
        range=xlimit,
        bins=bins,
        histtype="stepfilled",
        # density=False,
        density=True,
        label='S (train)',
        # label="Signal Distribution",
        # weights=sigw,
    )
    plt.hist(
        decisions[1],
        color="b",
        alpha=0.5,
        range=xlimit,
        bins=bins,
        histtype="stepfilled",
        # density=False,
        density=True,
        # label="Background Distribution",
        label='B (train)',
        # weights=bkgw,
    )
    hist, bins = np.histogram(decisions[2], bins=numbins, range=xlimit, density=True)

    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3], bins=numbins, range=xlimit, density=True)

    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    
    # plt.axvline(x= score,color='k')

    plt.xlabel("Score")
    plt.ylabel("Events")
    plt.yscale("log")
    plt.axvline(x= 0.958,color='k')
    plt.legend(loc="upper right")
    plt.subplot(211)
    plt.plot(fpr, tpr, "k-", label="All, AUC = %0.3f" % (area))
    plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


if True:
    totalscore = neuralNet.predict(X).ravel()

    numbins = len(totalscore)

    sigScore = neuralNet.predict(X[y > 0.5]).ravel()
    bkgScore = neuralNet.predict(X[y < 0.5]).ravel()
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

    # precision = tp/(fp+tp)
    # plt.plot(precision,totalscore, "k-",)
    # plt.show()
    # print(len(tp),len(fp),numbins)
flag = 1
if flag == 1:
    hl = ['weights','numjet','numlep','btag','srap','cent','m_bb','h_b']
    sample = ['s','b']
    for i in range(1, 10 + 1):
        for k in range(2): 
            for j in range(1,4): 
                if (k == 0 and j > 1): continue # This makes sure only one signal is added.
                command = "" # This line is here to clear out the previous command.
                command = "NN" + sample[k] + str(j) + "jeteta" + str(i) + " = []"
                exec(command)
                command = "" # This line is here to clear out the previous command.
                command = "NN" + sample[k] + str(j) + "jetphi" + str(i) + " = []"
                exec(command)
                command = "" #  This line is here to clear out the previous command.
                command = "NN" + sample[k] + str(j) + "jetpt" + str(i) + " = []"
                exec(command)
                for q in range(len(hl)):
                    command = "" #  This line is here to clear out the previous command.
                    command = "NN" + sample[k] + str(j) + hl[q] + " = []"
                    exec(command)
                for w in range(1,4):
                    command = "" #  This line is here to clear out the previous command.
                    command = "NN" + sample[k] + str(j) + "mt" + str(w) + " = []"
                    exec(command)
                    command = "" #  This line is here to clear out the previous command.
                    command = "NN" + sample[k] + str(j) + "dr" + str(w) + " = []"
                    exec(command)

    for i in range(len(X)):
        if i < len(signal):
            if allScore[i] > score:
                NNs1numjet.append(rawdata["numjet"].values[i])
                NNs1numlep.append(rawdata["numlep"].values[i])
                NNs1btag.append(rawdata["btag"].values[i])
                NNs1srap.append(rawdata["srap"].values[i])
                NNs1cent.append(rawdata["cent"].values[i])
                NNs1m_bb.append(rawdata["m_bb"].values[i])
                NNs1h_b.append(rawdata["h_b"].values[i])
    #             NNs1mt1.append(rawdata["mt1"].values[i])
    #             NNs1mt2.append(rawdata["mt2"].values[i])
    #             NNs1mt3.append(rawdata["mt3"].values[i])
    #             NNs1dr1.append(rawdata["dr1"].values[i])
    #             NNs1dr2.append(rawdata["dr2"].values[i])
    #             NNs1dr3.append(rawdata["dr3"].values[i])
    #             NNs1jetpt1.append(rawdata["jet1pT"].values[i])
    #             NNs1jetpt2.append(rawdata["jet2pT"].values[i])
    #             NNs1jetpt3.append(rawdata["jet3pT"].values[i])
    #             NNs1jetpt4.append(rawdata["jet4pT"].values[i])
    #             NNs1jetpt5.append(rawdata["jet5pT"].values[i])
    #             NNs1jetpt6.append(rawdata["jet6pT"].values[i])
    #             NNs1jetpt7.append(rawdata["jet7pT"].values[i])
    #             NNs1jetpt8.append(rawdata["jet8pT"].values[i])
    #             NNs1jetpt9.append(rawdata["jet9pT"].values[i])
    #             NNs1jetpt10.append(rawdata["jet10pT"].values[i])
    #             NNs1jeteta1.append(rawdata["jet1eta"].values[i])
    #             NNs1jeteta2.append(rawdata["jet2eta"].values[i])
    #             NNs1jeteta3.append(rawdata["jet3eta"].values[i])
    #             NNs1jeteta4.append(rawdata["jet4eta"].values[i])
    #             NNs1jeteta5.append(rawdata["jet5eta"].values[i])
    #             NNs1jeteta6.append(rawdata["jet6eta"].values[i])
    #             NNs1jeteta7.append(rawdata["jet7eta"].values[i])
    #             NNs1jeteta8.append(rawdata["jet8eta"].values[i])
    #             NNs1jeteta9.append(rawdata["jet9eta"].values[i])
    #             NNs1jeteta10.append(rawdata["jet10eta"].values[i])
    #             NNs1jetphi1.append(rawdata["jet1phi"].values[i])
    #             NNs1jetphi2.append(rawdata["jet2phi"].values[i])
    #             NNs1jetphi3.append(rawdata["jet3phi"].values[i])
    #             NNs1jetphi4.append(rawdata["jet4phi"].values[i])
    #             NNs1jetphi5.append(rawdata["jet5phi"].values[i])
    #             NNs1jetphi6.append(rawdata["jet6phi"].values[i])
    #             NNs1jetphi7.append(rawdata["jet7phi"].values[i])
    #             NNs1jetphi8.append(rawdata["jet8phi"].values[i])
    #             NNs1jetphi9.append(rawdata["jet9phi"].values[i])
    #             NNs1jetphi10.append(rawdata["jet10phi"].values[i])
                NNs1weights.append(scalefactor * rawdata["weights"].values[i])
        else:
            if allScore[i] > score:
                if rawdata['truth'].values[i] == 1:
                    NNb1numjet.append(rawdata["numjet"].values[i])
                    NNb1numlep.append(rawdata["numlep"].values[i])
                    NNb1btag.append(rawdata["btag"].values[i])
                    NNb1srap.append(rawdata["srap"].values[i])
                    NNb1cent.append(rawdata["cent"].values[i])
                    NNb1m_bb.append(rawdata["m_bb"].values[i])
                    NNb1h_b.append(rawdata["h_b"].values[i])
    #                 NNb1mt1.append(rawdata["mt1"].values[i])
    #                 NNb1mt2.append(rawdata["mt2"].values[i])
    #                 NNb1mt3.append(rawdata["mt3"].values[i])
    #                 NNb1dr1.append(rawdata["dr1"].values[i])
    #                 NNb1dr2.append(rawdata["dr2"].values[i])
    #                 NNb1dr3.append(rawdata["dr3"].values[i])
    #                 NNb1jetpt1.append(rawdata["jet1pT"].values[i])
    #                 NNb1jetpt2.append(rawdata["jet2pT"].values[i])
    #                 NNb1jetpt3.append(rawdata["jet3pT"].values[i])
    #                 NNb1jetpt4.append(rawdata["jet4pT"].values[i])
    #                 NNb1jetpt5.append(rawdata["jet5pT"].values[i])
    #                 NNb1jetpt6.append(rawdata["jet6pT"].values[i])
    #                 NNb1jetpt7.append(rawdata["jet7pT"].values[i])
    #                 NNb1jetpt8.append(rawdata["jet8pT"].values[i])
    #                 NNb1jetpt9.append(rawdata["jet9pT"].values[i])
    #                 NNb1jetpt10.append(rawdata["jet10pT"].values[i])
    #                 NNb1jeteta1.append(rawdata["jet1eta"].values[i])
    #                 NNb1jeteta2.append(rawdata["jet2eta"].values[i])
    #                 NNb1jeteta3.append(rawdata["jet3eta"].values[i])
    #                 NNb1jeteta4.append(rawdata["jet4eta"].values[i])
    #                 NNb1jeteta5.append(rawdata["jet5eta"].values[i])
    #                 NNb1jeteta6.append(rawdata["jet6eta"].values[i])
    #                 NNb1jeteta7.append(rawdata["jet7eta"].values[i])
    #                 NNb1jeteta8.append(rawdata["jet8eta"].values[i])
    #                 NNb1jeteta9.append(rawdata["jet9eta"].values[i])
    #                 NNb1jeteta10.append(rawdata["jet10eta"].values[i])
    #                 NNb1jetphi1.append(rawdata["jet1phi"].values[i])
    #                 NNb1jetphi2.append(rawdata["jet2phi"].values[i])
    #                 NNb1jetphi3.append(rawdata["jet3phi"].values[i])
    #                 NNb1jetphi4.append(rawdata["jet4phi"].values[i])
    #                 NNb1jetphi5.append(rawdata["jet5phi"].values[i])
    #                 NNb1jetphi6.append(rawdata["jet6phi"].values[i])
    #                 NNb1jetphi7.append(rawdata["jet7phi"].values[i])
    #                 NNb1jetphi8.append(rawdata["jet8phi"].values[i])
    #                 NNb1jetphi9.append(rawdata["jet9phi"].values[i])
    #                 NNb1jetphi10.append(rawdata["jet10phi"].values[i])
                    NNb1weights.append(rawdata["weights"].values[i])

                if rawdata['truth'].values[i] == 2:
                    NNb2numjet.append(rawdata["numjet"].values[i])
                    NNb2numlep.append(rawdata["numlep"].values[i])
                    NNb2btag.append(rawdata["btag"].values[i])
                    NNb2srap.append(rawdata["srap"].values[i])
                    NNb2cent.append(rawdata["cent"].values[i])
                    NNb2m_bb.append(rawdata["m_bb"].values[i])
                    NNb2h_b.append(rawdata["h_b"].values[i])
    #                 NNb2mt1.append(rawdata["mt1"].values[i])
    #                 NNb2mt2.append(rawdata["mt2"].values[i])
    #                 NNb2mt3.append(rawdata["mt3"].values[i])
    #                 NNb2dr1.append(rawdata["dr1"].values[i])
    #                 NNb2dr2.append(rawdata["dr2"].values[i])
    #                 NNb2dr3.append(rawdata["dr3"].values[i])
    #                 NNb2jetpt1.append(rawdata["jet1pT"].values[i])
    #                 NNb2jetpt2.append(rawdata["jet2pT"].values[i])
    #                 NNb2jetpt3.append(rawdata["jet3pT"].values[i])
    #                 NNb2jetpt4.append(rawdata["jet4pT"].values[i])
    #                 NNb2jetpt5.append(rawdata["jet5pT"].values[i])
    #                 NNb2jetpt6.append(rawdata["jet6pT"].values[i])
    #                 NNb2jetpt7.append(rawdata["jet7pT"].values[i])
    #                 NNb2jetpt8.append(rawdata["jet8pT"].values[i])
    #                 NNb2jetpt9.append(rawdata["jet9pT"].values[i])
    #                 NNb2jetpt10.append(rawdata["jet10pT"].values[i])
    #                 NNb2jeteta1.append(rawdata["jet1eta"].values[i])
    #                 NNb2jeteta2.append(rawdata["jet2eta"].values[i])
    #                 NNb2jeteta3.append(rawdata["jet3eta"].values[i])
    #                 NNb2jeteta4.append(rawdata["jet4eta"].values[i])
    #                 NNb2jeteta5.append(rawdata["jet5eta"].values[i])
    #                 NNb2jeteta6.append(rawdata["jet6eta"].values[i])
    #                 NNb2jeteta7.append(rawdata["jet7eta"].values[i])
    #                 NNb2jeteta8.append(rawdata["jet8eta"].values[i])
    #                 NNb2jeteta9.append(rawdata["jet9eta"].values[i])
    #                 NNb2jeteta10.append(rawdata["jet10eta"].values[i])
    #                 NNb2jetphi1.append(rawdata["jet1phi"].values[i])
    #                 NNb2jetphi2.append(rawdata["jet2phi"].values[i])
    #                 NNb2jetphi3.append(rawdata["jet3phi"].values[i])
    #                 NNb2jetphi4.append(rawdata["jet4phi"].values[i])
    #                 NNb2jetphi5.append(rawdata["jet5phi"].values[i])
    #                 NNb2jetphi6.append(rawdata["jet6phi"].values[i])
    #                 NNb2jetphi7.append(rawdata["jet7phi"].values[i])
    #                 NNb2jetphi8.append(rawdata["jet8phi"].values[i])
    #                 NNb2jetphi9.append(rawdata["jet9phi"].values[i])
    #                 NNb2jetphi10.append(rawdata["jet10phi"].values[i])
                    NNb2weights.append(rawdata["weights"].values[i]) 

                if rawdata['truth'].values[i] == 3:
                    NNb3numjet.append(rawdata["numjet"].values[i])
                    NNb3numlep.append(rawdata["numlep"].values[i])
                    NNb3btag.append(rawdata["btag"].values[i])
                    NNb3srap.append(rawdata["srap"].values[i])
                    NNb3cent.append(rawdata["cent"].values[i])
                    NNb3m_bb.append(rawdata["m_bb"].values[i])
                    NNb3h_b.append(rawdata["h_b"].values[i])
    #                 NNb3mt1.append(rawdata["mt1"].values[i])
    #                 NNb3mt2.append(rawdata["mt2"].values[i])
    #                 NNb3mt3.append(rawdata["mt3"].values[i])
    #                 NNb3dr1.append(rawdata["dr1"].values[i])
    #                 NNb3dr2.append(rawdata["dr2"].values[i])
    #                 NNb3dr3.append(rawdata["dr3"].values[i])
    #                 NNb3jetpt1.append(rawdata["jet1pT"].values[i])
    #                 NNb3jetpt2.append(rawdata["jet2pT"].values[i])
    #                 NNb3jetpt3.append(rawdata["jet3pT"].values[i])
    #                 NNb3jetpt4.append(rawdata["jet4pT"].values[i])
    #                 NNb3jetpt5.append(rawdata["jet5pT"].values[i])
    #                 NNb3jetpt6.append(rawdata["jet6pT"].values[i])
    #                 NNb3jetpt7.append(rawdata["jet7pT"].values[i])
    #                 NNb3jetpt8.append(rawdata["jet8pT"].values[i])
    #                 NNb3jetpt9.append(rawdata["jet9pT"].values[i])
    #                 NNb3jetpt10.append(rawdata["jet10pT"].values[i])
    #                 NNb3jeteta1.append(rawdata["jet1eta"].values[i])
    #                 NNb3jeteta2.append(rawdata["jet2eta"].values[i])
    #                 NNb3jeteta3.append(rawdata["jet3eta"].values[i])
    #                 NNb3jeteta4.append(rawdata["jet4eta"].values[i])
    #                 NNb3jeteta5.append(rawdata["jet5eta"].values[i])
    #                 NNb3jeteta6.append(rawdata["jet6eta"].values[i])
    #                 NNb3jeteta7.append(rawdata["jet7eta"].values[i])
    #                 NNb3jeteta8.append(rawdata["jet8eta"].values[i])
    #                 NNb3jeteta9.append(rawdata["jet9eta"].values[i])
    #                 NNb3jeteta10.append(rawdata["jet10eta"].values[i])
    #                 NNb3jetphi1.append(rawdata["jet1phi"].values[i])
    #                 NNb3jetphi2.append(rawdata["jet2phi"].values[i])
    #                 NNb3jetphi3.append(rawdata["jet3phi"].values[i])
    #                 NNb3jetphi4.append(rawdata["jet4phi"].values[i])
    #                 NNb3jetphi5.append(rawdata["jet5phi"].values[i])
    #                 NNb3jetphi6.append(rawdata["jet6phi"].values[i])
    #                 NNb3jetphi7.append(rawdata["jet7phi"].values[i])
    #                 NNb3jetphi8.append(rawdata["jet8phi"].values[i])
    #                 NNb3jetphi9.append(rawdata["jet9phi"].values[i])
    #                 NNb3jetphi10.append(rawdata["jet10phi"].values[i])
                    NNb3weights.append(rawdata["weights"].values[i])

    snumlep = df_signal["numlep"].values
    bnumlep = df_background["numlep"].values

    snumjet = df_signal["numjet"].values
    bnumjet = df_background["numjet"].values

    sbtag = df_signal["btag"].values
    bbtag = df_background["btag"].values

    ssrap = df_signal["srap"].values
    bsrap = df_background["srap"].values

    scent = df_signal["cent"].values
    bcent = df_background["cent"].values

    sm_bb = df_signal["m_bb"].values
    bm_bb = df_background["m_bb"].values

    sh_b = df_signal["h_b"].values
    bh_b = df_background["h_b"].values

    # smt1 = df_signal["mt1"].values
    # bmt1 = df_background["mt1"].values

    # smt2 = df_signal["mt2"].values
    # bmt2 = df_background["mt2"].values

    # smt3 = df_signal["mt3"].values
    # bmt3 = df_background["mt3"].values

    # sdr1 = df_signal["dr1"].values
    # bdr1 = df_background["dr1"].values

    # sdr2 = df_signal["dr2"].values
    # bdr2 = df_background["dr2"].values

    # sdr3 = df_signal["dr3"].values
    # bdr3 = df_background["dr3"].values

    # sjetpt1 = df_signal["jet1pT"].values
    # sjetpt2 = df_signal["jet2pT"].values
    # sjetpt3 = df_signal["jet3pT"].values
    # sjetpt4 = df_signal["jet4pT"].values
    # sjetpt5 = df_signal["jet5pT"].values
    # sjetpt6 = df_signal["jet6pT"].values
    # sjetpt7 = df_signal["jet7pT"].values
    # sjetpt8 = df_signal["jet8pT"].values
    # sjetpt9 = df_signal["jet9pT"].values
    # sjetpt10 = df_signal["jet10pT"].values
    # bjetpt1 = df_background["jet1pT"].values
    # bjetpt2 = df_background["jet2pT"].values
    # bjetpt3 = df_background["jet3pT"].values
    # bjetpt4 = df_background["jet4pT"].values
    # bjetpt5 = df_background["jet5pT"].values
    # bjetpt6 = df_background["jet6pT"].values
    # bjetpt7 = df_background["jet7pT"].values
    # bjetpt8 = df_background["jet8pT"].values
    # bjetpt9 = df_background["jet9pT"].values
    # bjetpt10 = df_background["jet10pT"].values

    # sjeteta1 = df_signal["jet1eta"].values
    # sjeteta2 = df_signal["jet2eta"].values
    # sjeteta3 = df_signal["jet3eta"].values
    # sjeteta4 = df_signal["jet4eta"].values
    # sjeteta5 = df_signal["jet5eta"].values
    # sjeteta6 = df_signal["jet6eta"].values
    # sjeteta7 = df_signal["jet7eta"].values
    # sjeteta8 = df_signal["jet8eta"].values
    # sjeteta9 = df_signal["jet9eta"].values
    # sjeteta10 = df_signal["jet10eta"].values
    # bjeteta1 = df_background["jet1eta"].values
    # bjeteta2 = df_background["jet2eta"].values
    # bjeteta3 = df_background["jet3eta"].values
    # bjeteta4 = df_background["jet4eta"].values
    # bjeteta5 = df_background["jet5eta"].values
    # bjeteta6 = df_background["jet6eta"].values
    # bjeteta7 = df_background["jet7eta"].values
    # bjeteta8 = df_background["jet8eta"].values
    # bjeteta9 = df_background["jet9eta"].values
    # bjeteta10 = df_background["jet10eta"].values

    # sjetphi1 = df_signal["jet1phi"].values
    # sjetphi2 = df_signal["jet2phi"].values
    # sjetphi3 = df_signal["jet3phi"].values
    # sjetphi4 = df_signal["jet4phi"].values
    # sjetphi5 = df_signal["jet5phi"].values
    # sjetphi6 = df_signal["jet6phi"].values
    # sjetphi7 = df_signal["jet7phi"].values
    # sjetphi8 = df_signal["jet8phi"].values
    # sjetphi9 = df_signal["jet9phi"].values
    # sjetphi10 = df_signal["jet10phi"].values
    # bjetphi1 = df_background["jet1phi"].values
    # bjetphi2 = df_background["jet2phi"].values
    # bjetphi3 = df_background["jet3phi"].values
    # bjetphi4 = df_background["jet4phi"].values
    # bjetphi5 = df_background["jet5phi"].values
    # bjetphi6 = df_background["jet6phi"].values
    # bjetphi7 = df_background["jet7phi"].values
    # bjetphi8 = df_background["jet8phi"].values
    # bjetphi9 = df_background["jet9phi"].values
    # bjetphi10 = df_background["jet10phi"].values
    def qPlot(x, y, nx, b1,b2,b3, a, b, c, Name):
        bins = np.arange(a, b + 1.5) - .5
        plt.hist(
            y,
            bins=bins,
            histtype="step",
            label="Full Background",
            linestyle="solid",
            color="black",
            weights=bkgw,
        )
        plt.hist(
            x,
            bins=bins,
            histtype="step",
            label="Full Signal",
            linestyle="solid",
            color="darkred",
            weights=sigw,
            stacked=False,
        )
        plt.hist(
            [b3,b2,b1],
            bins=bins,
            histtype="stepfilled",
            label=["TTH Score > %0.2f" % (score),"TTZ Score > %0.2f" % (score),"TTBB Score > %0.2f" % (score)],
            linestyle="solid",
            color=['blue','mediumorchid','green'],
            stacked=True,
            weights=[NNb3weights,NNb2weights,NNb1weights],
        )
        plt.hist(
            nx,
            bins=bins,
            histtype="step",
            hatch='/',
            label="Signal Score > %0.2f" % (score),
            linestyle="solid",
            color="darkred",
            weights=NNs1weights,
            stacked=False,
        )
        plt.xticks(bins + 0.5)
        plt.legend(loc=1,fontsize = 'x-small')
        plt.xlabel(Name, horizontalalignment='right', x=1.0)
        plt.ylabel('Events', horizontalalignment='right', y=1.0)
        plt.title(r'$\sqrt{s}=$ 14 TeV, $\mathcal{L} =$ 3000 fb${}^{-1}$')
        plt.ylabel("Events")
        plt.yscale("log")
        plt.style.use('classic')
        # plt.show()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
    def hPlot(x, y, nx, b1,b2,b3, a, b, c, Name):
        bins = np.linspace(a, b, c)
        plt.hist(
            y,
            bins=bins,
            histtype="step",
            label="Full Background",
            linestyle="solid",
            color="black",
            weights=bkgw,
        )
        plt.hist(
            x,
            bins=bins,
            histtype="step",
            label="Full Signal",
            linestyle="solid",
            color="darkred",
            weights=sigw,
            stacked=False,
        )
        plt.hist(
            [b3,b2,b1],
            bins=bins,
            histtype="stepfilled",
            label=["TTH Score > %0.2f" % (score),"TTZ Score > %0.2f" % (score),"TTBB Score > %0.2f" % (score)],
            linestyle="solid",
            color=['blue','mediumorchid','green'],
            stacked=True,
            weights=[NNb3weights,NNb2weights,NNb1weights],
        )
        plt.hist(
            nx,
            bins=bins,
            histtype="step",
            hatch='/',
            label="Signal Score > %0.2f" % (score),
            linestyle="solid",
            color="darkred",
            weights=NNs1weights,
            stacked=False,
        )
        
        plt.legend(loc=1,fontsize = 'x-small')
        plt.xlabel(Name, horizontalalignment='right', x=1.0)
        plt.ylabel('Events', horizontalalignment='right', y=1.0)
        plt.title(r'$\sqrt{s}=$ 14 TeV, $\mathcal{L} =$ 3000 fb${}^{-1}$')
        plt.ylabel("Events")
        plt.yscale("log")
        plt.style.use('classic')
        # plt.show()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

    
    # tn, fp, fn, tp = confusion_matrix(y, y_pred,normalize='all').ravel()
    # Matrix = np.matrix([[tp,fn],[fp,tn]])
    # slug.confusedMatrix(Matrix)

    # y_pred2 = [1 * (x[0] >= score) for x in allScore]

    # tn, fp, fn, tp = confusion_matrix(y, y_pred2,normalize='all').ravel()
    # Matrix2 = np.matrix([[tp,fn],[fp,tn]])
    # slug.confusedMatrix(Matrix2)

    pdfname = file[:-2] + 'pdf'
    with PdfPages(pdfname) as pdf:

    #     plt.figure(figsize=(8, 6))
    #     plt.subplot(212)
    #     plt.hist(
    #         sigScore,
    #         color="r",
    #         alpha=0.5,
    #         range=xlimit,
    #         bins=100,
    #         histtype="stepfilled",
    #         # density=False,
    #         density=True,
    #         label='S (train)',
    #         # label="Signal Distribution",
    #         weights=sigw,
    #     )
    #     plt.hist(
    #         bkgScore,
    #         color="b",
    #         alpha=0.5,
    #         range=xlimit,
    #         bins=100,
    #         histtype="stepfilled",
    #         # density=False,
    #         density=True,
    #         # label="Background Distribution",
    #         label='B (train)',
    #         weights=bkgw,
    #     )
    #     plt.axvline(x= score,color='k')
    #     plt.xlabel("Score")
    #     plt.ylabel("Events")
    #     plt.yscale("log")
    #     plt.legend(loc="upper right")
    #     plt.subplot(211)
    #     plt.plot(yplot, xplot, "k-", label="All, AUC = %0.3f" % (area))
    #     plt.plot(maxs,maxb,'ko') 
    #     plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck")
    #     plt.xlim([-0.05, 1.05])
    #     plt.ylim([-0.05, 1.05])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("Receiver operating characteristic")
    #     plt.legend(loc="lower right")
    #     plt.grid()
    #     pdf.savefig()  # saves the current figure into a pdf page
    #     plt.close()

        qPlot(snumjet, bnumjet, NNs1numjet, NNb1numjet, NNb2numjet, NNb3numjet, 1, 21, 22, 'Jet multiplicity')
        
        qPlot(snumlep, bnumlep, NNs1numlep, NNb1numlep,NNb2numlep,NNb3numlep, 0, 3, 5, 'Lepton multiplicity')

        qPlot(sbtag, bbtag, NNs1btag, NNb1btag,NNb2btag, NNb3btag,0, 10, 10, 'N b-tagged jets')

        hPlot(ssrap, bsrap, NNs1srap, NNb1srap, NNb2srap,NNb3srap, 0, 10, 20, r'$ < \eta(b_{i},b_{j}) >$')

        hPlot(scent, bcent, NNs1cent, NNb1cent,NNb2cent, NNb3cent,0, 1, 20, 'Centrality')

        hPlot(sm_bb, bm_bb, NNs1m_bb, NNb1m_bb,NNb2m_bb,NNb3m_bb, 0, 250, 25, r'${M}_{bb}$ [GeV]')

        hPlot(sh_b, bh_b, NNs1h_b, NNb1h_b,NNb2h_b,NNb3h_b, 0, 1500, 60, r'${H}_{B}$ [GeV]')
 
        # hPlot(smt1, bmt1, NNs1mt1, NNb1mt1,NNb2mt1,NNb3mt1, 0, 300, 100, r'${m}_{T}1$ [GeV]')

        # hPlot(smt2, bmt2, NNs1mt2, NNb1mt2,NNb2mt2,NNb3mt2, 0, 300, 100, r'${m}_{T}2$ [GeV]')

        # hPlot(smt3, bmt3, NNs1mt3, NNb1mt3,NNb2mt3,NNb3mt3, 0, 300, 100, r'${m}_{T}3$ [GeV]')

        # hPlot(sdr1, bdr1, NNs1dr1, NNb1dr1,NNb2dr1,NNb3dr1, 0, 7, 100, r'$\Delta$R1')

        # hPlot(sdr2, bdr2, NNs1dr2, NNb1dr2,NNb2dr2,NNb3dr2, 0, 7, 100, r'$\Delta$R2')

        # hPlot(sdr3, bdr3, NNs1dr3, NNb1dr3,NNb2dr3,NNb3dr3, 0, 7, 100, r'$\Delta$R3')

        # hPlot(sjetpt1, bjetpt1, NNs1jetpt1, NNb1jetpt1,NNb2jetpt1,NNb3jetpt1, 0, 1e6, 100, r'Jet1 pT')

        # hPlot(sjetpt2, bjetpt2, NNs1jetpt2, NNb1jetpt2,NNb2jetpt2,NNb3jetpt2, 0, 1e6, 100, r'Jet2 pT')

        # hPlot(sjetpt3, bjetpt3, NNs1jetpt3, NNb1jetpt3,NNb2jetpt3,NNb3jetpt3, 0, 1e6, 100, r'Jet3 pT')
  
        # hPlot(sjetpt4, bjetpt4, NNs1jetpt4, NNb1jetpt4,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet4 pT')

        # hPlot(sjetpt5, bjetpt5, NNs1jetpt5, NNb1jetpt5,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet5 pT')

        # hPlot(sjetpt6, bjetpt6, NNs1jetpt6, NNb1jetpt6,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet6 pT')

        # hPlot(sjetpt7, bjetpt7, NNs1jetpt7, NNb1jetpt7,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet7 pT')

        # hPlot(sjetpt8, bjetpt8, NNs1jetpt8, NNb1jetpt8,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet8 pT')

        # hPlot(sjetpt9, bjetpt9, NNs1jetpt9, NNb1jetpt9,NNb2jetpt4,NNb3jetpt4, 0, 1e6, 100, r'Jet9 pT')

        # hPlot(sjetpt10, bjetpt10, NNs1jetpt10, NNb1jetpt10,NNb2jetpt10,NNb3jetpt10, 0, 1e6, 100, r'Jet10 pT')

        # hPlot(sjeteta1, bjeteta1, NNs1jeteta1, NNb1jeteta1,NNb2jeteta1,NNb3jeteta1, -6, 6, 12, r'Jet1 $\eta$')

        # hPlot(sjeteta2, bjeteta2, NNs1jeteta2, NNb1jeteta2,NNb2jeteta2,NNb3jeteta2, -6, 6, 12, r'Jet2 $\eta$')

        # hPlot(sjeteta3, bjeteta3, NNs1jeteta3, NNb1jeteta3,NNb2jeteta3,NNb3jeteta3, -6, 6, 12, r'Jet3 $\eta$')
 
        # hPlot(sjeteta4, bjeteta4, NNs1jeteta4, NNb1jeteta4,NNb2jeteta4,NNb3jeteta4, -6, 6, 12, r'Jet4 $\eta$')

        # hPlot(sjeteta5, bjeteta5, NNs1jeteta5, NNb1jeteta5,NNb2jeteta5,NNb3jeteta5, -6, 6, 12, r'Jet5 $\eta$')
   
        # hPlot(sjeteta6, bjeteta6, NNs1jeteta6, NNb1jeteta6,NNb2jeteta6,NNb3jeteta6, -6, 6, 12, r'Jet6 $\eta$')

        # hPlot(sjeteta7, bjeteta7, NNs1jeteta7, NNb1jeteta7,NNb2jeteta7,NNb3jeteta7, -6, 6, 12, r'Jet7 $\eta$')

        # hPlot(sjeteta8, bjeteta8, NNs1jeteta8, NNb1jeteta8,NNb2jeteta8,NNb3jeteta8, -6, 6, 12, r'Jet8 $\eta$')
 
        # hPlot(sjeteta9, bjeteta9, NNs1jeteta9, NNb1jeteta9,NNb2jeteta9,NNb3jeteta9, -6, 6, 12, r'Jet9 $\eta$')

        # hPlot(sjeteta10, bjeteta10, NNs1jeteta10, NNb1jeteta10,NNb2jeteta10,NNb3jeteta10, -6, 6, 12, r'Jet10 $\eta$')
      
        # hPlot(sjetphi1, bjetphi1, NNs1jetphi1, NNb1jetphi1,NNb2jetphi1,NNb3jetphi1, -4, 4, 8, r'Jet1 $\phi$')
        
        # hPlot(sjetphi2, bjetphi2, NNs1jetphi2, NNb1jetphi2,NNb2jetphi1,NNb3jetphi2, -4, 4, 8, r'Jet2 $\phi$')
        
        # hPlot(sjetphi3, bjetphi3, NNs1jetphi3, NNb1jetphi3,NNb2jetphi1,NNb3jetphi3, -4, 4, 8, r'Jet3 $\phi$')

        # hPlot(sjetphi4, bjetphi4, NNs1jetphi4, NNb1jetphi4,NNb2jetphi1,NNb3jetphi4, -4, 4, 8, r'Jet4 $\phi$')
       
        # hPlot(sjetphi5, bjetphi5, NNs1jetphi5, NNb1jetphi5,NNb2jetphi1,NNb3jetphi5, -4, 4, 8, r'Jet5 $\phi$')
      
        # hPlot(sjetphi6, bjetphi6, NNs1jetphi6, NNb1jetphi6,NNb2jetphi1,NNb3jetphi6, -4, 4, 8, r'Jet6 $\phi$')
      
        # hPlot(sjetphi7, bjetphi7, NNs1jetphi7, NNb1jetphi7,NNb2jetphi1,NNb3jetphi7, -4, 4, 8, r'Jet7 $\phi$')
    
        # hPlot(sjetphi8, bjetphi8, NNs1jetphi8, NNb1jetphi8,NNb2jetphi1,NNb3jetphi8, -4, 4, 8, r'Jet8 $\phi$')
       
        # hPlot(sjetphi9, bjetphi9, NNs1jetphi9, NNb1jetphi9,NNb2jetphi1,NNb3jetphi9, -4, 4, 8, r'Jet9 $\phi$')
    
        # hPlot(sjetphi10, bjetphi10, NNs1jetphi10, NNb1jetphi10,NNb2jetphi1,NNb3jetphi10, -4, 4, 8, r'Jet10 $\phi$')
        

        d = pdf.infodict()
        d['Title'] = 'LoadNN'
        d['Author'] = u'Jonathan O. Tellechea\xe4nen'
        d['Subject'] = '1D plots that apply NN score for a cut.'
        d['Keywords'] = 'ttHH'
        # d['CreationDate'] = datetime.datetime(2009, 11, 13)
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        print(pdfname)
