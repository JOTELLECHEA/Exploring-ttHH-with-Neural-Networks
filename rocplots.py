# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that creates a ROC plot for Low, High, and Low + High VARS (Background rejection vs signal efficiency).
##############################################################################################################################
# Imported packages.
import pandas as pd
import tkinter as tk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Load High and Low level csv file created by rocs.py script.
hnlv = pd.read_csv("csv/highandlowlvlvars.csv")
f_hnlv = hnlv["fpr"]
t_hnlv = hnlv["tpr"]
bkgR_hnlv = hnlv["bkgR"]

# Load Low level VARS csv file created by rocs.py script.
llv = pd.read_csv("csv/lowlvlvars.csv")
f_llv = llv["fpr"]
t_llv = llv["tpr"]
bkgR_llv = llv["bkgR"]

# Load High Level VARS csv file created by rocs.py script.
hlv = pd.read_csv("csv/highlvlvars.csv")
f_hlv = hlv["fpr"]
t_hlv = hlv["tpr"]
bkgR_hlv = hlv["bkgR"]

# Plots of the three ROC.
fig1 = plt.figure(1)
plt.plot(t_hlv, bkgR_hlv, "r-", label="High level variables, AUC = %0.3f" % (auc(f_hlv, t_hlv)))
plt.plot(t_llv, bkgR_llv, "b-", label="Low level variables, AUC = %0.3f" % (auc(f_llv, t_llv)))
plt.plot(t_hnlv, bkgR_hnlv, "k-", label="All variables, AUC = %0.3f" % (auc(f_hnlv, t_hnlv)))

# Plot parameters and title/labels.
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.title("Modified Receiver operating characteristic")
plt.legend(loc="lower left")
plt.grid()


fig2 = plt.figure(2)
plt.plot(f_hlv, t_hlv, "r-", label="High level variables, AUC = %0.3f" % (auc(f_hlv, t_hlv)))
plt.plot(f_llv, t_llv, "b-", label="Low level variables, AUC = %0.3f" % (auc(f_llv, t_llv)))
plt.plot(f_hnlv, t_hnlv, "k-", label="All variables, AUC = %0.3f" % (auc(f_hnlv, t_hnlv)))
plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Luck, AUC = 0.500")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.grid()
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()
