# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: This script keeps a record of trained NN; Keeps track of time , AUC , lenght of NN etc. 
# The filename for the saved weights is displayed to be used in loadNN.py to create plots.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
# Imported packages.
import pandas as pd
import numpy as np
import argparse


####work in progress to automate script
# parser = argparse.ArgumentParser(description="Plot 1D plots of sig/bac")
# parser.add_argument("--file", type=str, help="Use '--file=' followed by a *.h5 file")
# args = parser.parse_args()
# file = "data/" + str(args.file)
# file = 'hyperparameterRecord_v3.csv'
# file = 'fiveLayerDropout_2.csv'
# file = 'fiveLayerDropout_3.csv'
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Y/M/D @ H:M','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']
# modelParam  = ['NN Archi.','#Br.','LR','Batch','AUC','Avg.P','Run Time','ConfusionMatrix [TP FP] [FN TN]','Score','Max Signif','nsig','nbkg']#######

file = 'csv/jet.csv'
modelParam = [
        'FileName',
        "ConfusionMatrix [TP FP] [FN TN]",
        "Run Time",
        "AUC",
        "Avg.P",
        "Score",
        "Max Signif",
        "nsig",
        "nbkg"
    ]
data = pd.read_csv(file)
print(data.to_string(justify='right',columns=modelParam,header=True,index=1))