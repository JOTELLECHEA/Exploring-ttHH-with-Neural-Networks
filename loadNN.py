# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
seed = 42
tree = 'OutputTree'
# branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1']
branches = ['numjet','numlep','btag','srap','cent','m_bb','h_b','mt1','dr1','lep1pT','lep1eta','lep1phi',
'lep2pT','lep2eta','lep2phi','lep3pT','lep3eta','lep3phi','jet1pT','jet1eta','jet1phi','jet1b','jet1c',
'jet2pT','jet2eta','jet2phi','jet2b','jet2c','jet3pT','jet3eta','jet3phi','jet3b','jet3c',
'jet4pT','jet4eta','jet4phi','jet4b','jet4c','jet5pT','jet5eta','jet5phi','jet5b','jet5c',
'jet6pT','jet6eta','jet6phi','jet6b','jet6c','jet7pT','jet7eta','jet7phi','jet7b','jet7c',
'jet8pT','jet8eta','jet8phi','jet8b','jet8c','jet9pT','jet9eta','jet9phi','jet9b','jet9c',
'jet10pT','jet10eta','jet10phi','jet10b','jet10c','jet11pT','jet11eta','jet11phi','jet11b','jet11c',
'jet12pT','jet12eta','jet12phi','jet12b','jet12c','jet13pT','jet13eta','jet13phi','jet13b','jet13c',
'jet14pT','jet14eta','jet14phi','jet14b','jet14c','jet15pT','jet15eta','jet15phi','jet15b','jet15c',
'jet16pT','jet16eta','jet16phi','jet16b','jet16c','jet17pT','jet17eta','jet17phi','jet17b','jet17c',
'jet18pT','jet18eta','jet18phi','jet18b','jet18c','jet19pT','jet19eta','jet19phi','jet19b','jet19c',
'jet20pT','jet20eta','jet20phi','jet20b','jet20c','jet21pT','jet21eta','jet21phi','jet21b','jet21c']
numBranches = len(branches)
###########################################################################################################################
parser = argparse.ArgumentParser(description= 'Plot 1D plots of sig/bac')
parser.add_argument("--file", type=str, help= "Use '--file=' followed by a *.h5 file")
args = parser.parse_args()
file = str(args.file)
###########################################################################################################################
# Data read from file.
signal         = uproot.open('data/new_signal_tthh.root')[tree]
df_signal      = signal.pandas.df(branches)
background     = uproot.open('data/background.root')[tree]
df_background  = background.pandas.df(branches)
shuffleBackground = shuffle(df_background,random_state=seed)
#signal and limited shuffle background data to counter inbalanced data problem.
X = pd.concat([df_signal,shuffleBackground])
z = sc.fit_transform(X)

neuralNet = keras.models.load_model(file)

score = neuralNet.predict(z).ravel()

NNsnumjet = []
NNsnumlep = []
NNsbtag   = []
NNssrap   = []
NNscent   = []
NNsm_bb   = []
NNsh_b    = []
NNsmt1    = []
NNsdr1    = []
NNbnumjet = []
NNbnumlep = []
NNbbtag   = []
NNbsrap   = []
NNbcent   = []
NNbm_bb   = []
NNbh_b    = []
NNbmt1    = []
NNbdr1    = []

ScoreForMaxSignif = 0.997

for i in range(len(X)):
    if i<len(signal):
        if score[i]>ScoreForMaxSignif: 
            NNsnumjet.append(X['numjet'].values[i])
            NNsnumlep.append(X['numlep'].values[i])
            NNsbtag.append(X['btag'].values[i])
            NNssrap.append(X['srap'].values[i])
            NNscent.append(X['cent'].values[i])
            NNsm_bb.append(X['m_bb'].values[i])
            NNsh_b.append(X['h_b'].values[i])
            NNsmt1.append(X['mt1'].values[i])
            NNsdr1.append(X['dr1'].values[i])
    else:
        if score[i]>ScoreForMaxSignif:
            NNbnumjet.append(X['numjet'].values[i])
            NNbnumlep.append(X['numlep'].values[i])
            NNbbtag.append(X['btag'].values[i])
            NNbsrap.append(X['srap'].values[i])
            NNbcent.append(X['cent'].values[i])
            NNbm_bb.append(X['m_bb'].values[i])
            NNbh_b.append(X['h_b'].values[i])
            NNbmt1.append(X['mt1'].values[i])
            NNbdr1.append(X['dr1'].values[i])

snumlep = df_signal['numlep'].values
bnumlep = df_background['numlep'].values

snumjet = df_signal['numjet'].values
bnumjet = df_background['numjet'].values

sbtag = df_signal['btag'].values
bbtag = df_background['btag'].values

ssrap = df_signal['srap'].values
bsrap = df_background['srap'].values

scent = df_signal['cent'].values
bcent = df_background['cent'].values

sm_bb = df_signal['m_bb'].values
bm_bb = df_background['m_bb'].values

sh_b = df_signal['h_b'].values
bh_b = df_background['h_b'].values

smt1 = df_signal['mt1'].values
bmt1 = df_background['mt1'].values

sdr1 = df_signal['dr1'].values
bdr1 = df_background['dr1'].values
def hPlot(x,y,nx,ny,a,b,c,Name):
    bins = np.linspace(a,b,c)
    plt.hist(y, bins=bins,histtype='step',label='background',linestyle='solid',color='steelblue')
    plt.hist(x, bins=bins,histtype='step',label='signal',linestyle='solid',color='firebrick')
    plt.hist(ny, bins=bins,histtype='step',label='NN-background',linestyle='dashed',color='steelblue')
    plt.hist(nx, bins=bins,histtype='step',label='NN-signal',linestyle='dashed',color='firebrick')
    plt.legend(loc=1)
    plt.xlabel(Name)
    plt.ylabel('Events')
    plt.yscale('log')


fig1 = plt.figure()

ax1 = fig1.add_subplot(221)
hPlot(snumjet,bnumjet,NNsnumjet,NNbnumjet,1,21,21,branches[0])

ax2 = fig1.add_subplot(222)
hPlot(snumlep,bnumlep,NNsnumlep,NNbnumlep,1,3,3,branches[1])

ax3 = fig1.add_subplot(223)
hPlot(sbtag,bbtag,NNsbtag,NNbbtag,0,10,10,branches[2])

ax4 = fig1.add_subplot(224)
hPlot(ssrap,bsrap,NNssrap,NNbsrap,0,10,10,branches[3])

fig2 = plt.figure(2)

ax1 = fig2.add_subplot(221)
hPlot(scent,bcent,NNscent,NNbcent,0,1,10,branches[4])

ax2 = fig2.add_subplot(222)
hPlot(sm_bb,bm_bb,NNsm_bb,NNbm_bb,0,250,10,branches[5])

ax3 = fig2.add_subplot(223)
hPlot(sh_b,bh_b,NNsh_b,NNbh_b,0,1500,10,branches[6])

ax4 = fig2.add_subplot(224)
hPlot(smt1,bmt1,NNsmt1,NNbmt1,0,300,100,branches[7])

plot3 = plt.figure(3)
hPlot(sdr1,bdr1,NNsdr1,NNbdr1,0,7,100,branches[8])

plt.show()