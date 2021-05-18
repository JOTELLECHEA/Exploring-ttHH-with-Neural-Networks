# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script plots sherpa data vs atlas data.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
import uproot
import numpy as np
import pandas as pd
import slug
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
seed = 42
tree = "OutputTree"
treesherpa ='allev/hftree'
phase = 3

High = [
    "numjet",
    "numlep",
    "btag",
    "srap",
    "cent",
    "m_bb",
    "h_b",
    "mt1",
    "mt2",
    "mt3",
    "dr1",
    "dr2",
    "dr3",
    'weights'
]
types = ["pT", "eta", "phi", "btag"]
LeptonVAR = []
JetVAR = []
for i in range(3):
    for j in range(3):
        LeptonVAR.append("lepton" + str(j + 1) + types[i])
for i in range(len(types)):
    for j in range(10):
        JetVAR.append("jet" + str(j + 1) + types[i])
###                                               -END

branches = High + LeptonVAR + JetVAR

bkgTTBB = uproot.open("data/new_TTBB.root")[tree]
df_bkgTTBB = bkgTTBB.pandas.df(branches)

bkgsherpa = uproot.open("/data/users/mhance/tthh/ttbbjj_histograms.root")[treesherpa]
df_bkgsherpa = bkgsherpa.pandas.df(branches)


def qPlot(x, y, a, b, c, Name):
        # bins = np.linspace(a, b, c)
        bins = np.arange(a, b + 1.5) - .5
        plt.hist(
            [x,y],
            bins=bins,
            histtype="step",
            label=["SHERPA Background",'ATLAS Background'],
            linestyle="solid",
            color=["black",'green'],
            weights=[weightsSHERPA,weightsATLAS],
        )
        plt.xticks(bins + 0.5)
        plt.legend(loc=1,fontsize = 'x-small')
        plt.xlabel(Name, horizontalalignment='right', x=1.0)
        plt.ylabel('Events', horizontalalignment='right', y=1.0)
        plt.title(r'$\sqrt{s}=$ 14 TeV, $\mathcal{L} =$ 3000 fb${}^{-1}$')
        plt.style.use('classic')
        plt.yscale('log')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close() 

def hPlot(x, y, a, b, c, Name):
        bins = np.linspace(a, b, c)
        plt.hist(
            [x,y],
            bins=bins,
            histtype="step",
            label=["SHERPA Background",'ATLAS Background'],
            linestyle="solid",
            color=["black",'green'],
            weights=[weightsSHERPA,weightsATLAS],
        )
        # plt.xticks(bins)
        plt.legend(loc=1,fontsize = 'x-small')
        plt.xlabel(Name, horizontalalignment='right', x=1.0)
        plt.ylabel('Events', horizontalalignment='right', y=1.0)
        plt.title(r'$\sqrt{s}=$ 14 TeV, $\mathcal{L} =$ 3000 fb${}^{-1}$')
        plt.style.use('classic')
        plt.yscale('log')
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close() 

hl = ['weights','numjet','numlep','btag','srap','cent','m_bb','h_b']
sample = ['ATLAS','SHERPA']
for i in range(1, 10 + 1):
    for k in range(2): 
        for j in range(1,4): 
            if (k == 0 and j > 1): continue # This makes sure only one signal is added.
            command = "" # This line is here to clear out the previous command.
            command = "jeteta" + str(i) + sample[k] + " = []"
            exec(command)
            command = "" # This line is here to clear out the previous command.
            command = "jetphi" + str(i) +sample[k] + " = []"
            exec(command)
            command = "" #  This line is here to clear out the previous command.
            command = "jetpt" + str(i) + sample[k] + " = []"
            exec(command)
            command = "" #  This line is here to clear out the previous command.
            command = "jetbtag" + str(i) + sample[k] + " = []"
            exec(command)
            for q in range(len(hl)):# High Level and weights inti...
                command = "" #  This line is here to clear out the previous command.
                command = hl[q] +sample[k] + " = []"
                exec(command)
            for w in range(1,4):
                command = "" #  This line is here to clear out the previous command.
                command = "mt" + str(w) + sample[k] + " = []"
                exec(command)
                command = "" #  This line is here to clear out the previous command.
                command = "dr" + str(w) +sample[k] + " = []"
                exec(command)
                # command = "" #  This line is here to clear out the previous command.
                # command = "jet" + str(w) + 'btag' +sample[k] + " = []"
                # exec(command)


for i in range(len(df_bkgsherpa)):
    if df_bkgsherpa['numlep'].values[i] > 0 :
        numjetSHERPA.append(df_bkgsherpa['numjet'].values[i])
        numlepSHERPA.append(df_bkgsherpa['numlep'].values[i])
        weightsSHERPA.append(df_bkgsherpa['weights'].values[i]/1000)
        btagSHERPA.append(df_bkgsherpa["btag"].values[i])
        srapSHERPA.append(df_bkgsherpa["srap"].values[i])
        centSHERPA.append(df_bkgsherpa["cent"].values[i])
        m_bbSHERPA.append(df_bkgsherpa["m_bb"].values[i]/1000)
        h_bSHERPA.append(df_bkgsherpa["h_b"].values[i]/1000)
        mt1SHERPA.append(df_bkgsherpa["mt1"].values[i]/1000)
        mt2SHERPA.append(df_bkgsherpa["mt2"].values[i]/1000)
        mt3SHERPA.append(df_bkgsherpa["mt3"].values[i]/1000)
        dr1SHERPA.append(df_bkgsherpa["dr1"].values[i])
        dr2SHERPA.append(df_bkgsherpa["dr2"].values[i])
        dr3SHERPA.append(df_bkgsherpa["dr3"].values[i])
        jetpt1SHERPA.append(df_bkgsherpa["jet1pT"].values[i])
        jetpt2SHERPA.append(df_bkgsherpa["jet2pT"].values[i])
        jetpt3SHERPA.append(df_bkgsherpa["jet3pT"].values[i])
        jetpt4SHERPA.append(df_bkgsherpa["jet4pT"].values[i])
        jetpt5SHERPA.append(df_bkgsherpa["jet5pT"].values[i])
        jetpt6SHERPA.append(df_bkgsherpa["jet6pT"].values[i])
        jetpt7SHERPA.append(df_bkgsherpa["jet7pT"].values[i])
        jetpt8SHERPA.append(df_bkgsherpa["jet8pT"].values[i])
        jetpt9SHERPA.append(df_bkgsherpa["jet9pT"].values[i])
        jetpt10SHERPA.append(df_bkgsherpa["jet10pT"].values[i])
        jetbtag1SHERPA.append(df_bkgsherpa["jet1btag"].values[i])
        jetbtag2SHERPA.append(df_bkgsherpa["jet2btag"].values[i])
        jetbtag3SHERPA.append(df_bkgsherpa["jet3btag"].values[i])
        jetbtag4SHERPA.append(df_bkgsherpa["jet4btag"].values[i])
        jetbtag5SHERPA.append(df_bkgsherpa["jet5btag"].values[i])
        jetbtag6SHERPA.append(df_bkgsherpa["jet6btag"].values[i])
        jetbtag7SHERPA.append(df_bkgsherpa["jet7btag"].values[i])
        jetbtag8SHERPA.append(df_bkgsherpa["jet8btag"].values[i])
        jetbtag9SHERPA.append(df_bkgsherpa["jet9btag"].values[i])
        jetbtag10SHERPA.append(df_bkgsherpa["jet10btag"].values[i])
        jeteta1SHERPA.append(df_bkgsherpa["jet1eta"].values[i])
        jeteta2SHERPA.append(df_bkgsherpa["jet2eta"].values[i])
        jeteta3SHERPA.append(df_bkgsherpa["jet3eta"].values[i])
        jeteta4SHERPA.append(df_bkgsherpa["jet4eta"].values[i])
        jeteta5SHERPA.append(df_bkgsherpa["jet5eta"].values[i])
        jeteta6SHERPA.append(df_bkgsherpa["jet6eta"].values[i])
        jeteta7SHERPA.append(df_bkgsherpa["jet7eta"].values[i])
        jeteta8SHERPA.append(df_bkgsherpa["jet8eta"].values[i])
        jeteta9SHERPA.append(df_bkgsherpa["jet9eta"].values[i])
        jeteta10SHERPA.append(df_bkgsherpa["jet10eta"].values[i])
        jetphi1SHERPA.append(df_bkgsherpa["jet1phi"].values[i])
        jetphi2SHERPA.append(df_bkgsherpa["jet2phi"].values[i])
        jetphi3SHERPA.append(df_bkgsherpa["jet3phi"].values[i])
        jetphi4SHERPA.append(df_bkgsherpa["jet4phi"].values[i])
        jetphi5SHERPA.append(df_bkgsherpa["jet5phi"].values[i])
        jetphi6SHERPA.append(df_bkgsherpa["jet6phi"].values[i])
        jetphi7SHERPA.append(df_bkgsherpa["jet7phi"].values[i])
        jetphi8SHERPA.append(df_bkgsherpa["jet8phi"].values[i])
        jetphi9SHERPA.append(df_bkgsherpa["jet9phi"].values[i])
        jetphi10SHERPA.append(df_bkgsherpa["jet10phi"].values[i])

for i in range(len(df_bkgTTBB)):
    numjetATLAS.append(df_bkgTTBB['numjet'].values[i])
    numlepATLAS.append(df_bkgTTBB['numlep'].values[i])
    weightsATLAS.append(df_bkgTTBB['weights'].values[i])
    btagATLAS.append(df_bkgTTBB["btag"].values[i])
    srapATLAS.append(df_bkgTTBB["srap"].values[i])
    centATLAS.append(df_bkgTTBB["cent"].values[i])
    m_bbATLAS.append(df_bkgTTBB["m_bb"].values[i])
    h_bATLAS.append(df_bkgTTBB["h_b"].values[i])
    mt1ATLAS.append(df_bkgTTBB["mt1"].values[i])
    mt2ATLAS.append(df_bkgTTBB["mt2"].values[i])
    mt3ATLAS.append(df_bkgTTBB["mt3"].values[i])
    dr1ATLAS.append(df_bkgTTBB["dr1"].values[i])
    dr2ATLAS.append(df_bkgTTBB["dr2"].values[i])
    dr3ATLAS.append(df_bkgTTBB["dr3"].values[i])
    jetpt1ATLAS.append(df_bkgTTBB["jet1pT"].values[i])
    jetpt2ATLAS.append(df_bkgTTBB["jet2pT"].values[i])
    jetpt3ATLAS.append(df_bkgTTBB["jet3pT"].values[i])
    jetpt4ATLAS.append(df_bkgTTBB["jet4pT"].values[i])
    jetpt5ATLAS.append(df_bkgTTBB["jet5pT"].values[i])
    jetpt6ATLAS.append(df_bkgTTBB["jet6pT"].values[i])
    jetpt7ATLAS.append(df_bkgTTBB["jet7pT"].values[i])
    jetpt8ATLAS.append(df_bkgTTBB["jet8pT"].values[i])
    jetpt9ATLAS.append(df_bkgTTBB["jet9pT"].values[i])
    jetpt10ATLAS.append(df_bkgTTBB["jet10pT"].values[i])
    jetbtag1ATLAS.append(df_bkgTTBB["jet1btag"].values[i])
    jetbtag2ATLAS.append(df_bkgTTBB["jet2btag"].values[i])
    jetbtag3ATLAS.append(df_bkgTTBB["jet3btag"].values[i])
    jetbtag4ATLAS.append(df_bkgTTBB["jet4btag"].values[i])
    jetbtag5ATLAS.append(df_bkgTTBB["jet5btag"].values[i])
    jetbtag6ATLAS.append(df_bkgTTBB["jet6btag"].values[i])
    jetbtag7ATLAS.append(df_bkgTTBB["jet7btag"].values[i])
    jetbtag8ATLAS.append(df_bkgTTBB["jet8btag"].values[i])
    jetbtag9ATLAS.append(df_bkgTTBB["jet9btag"].values[i])
    jetbtag10ATLAS.append(df_bkgTTBB["jet10btag"].values[i])
    jeteta1ATLAS.append(df_bkgTTBB["jet1eta"].values[i])
    jeteta2ATLAS.append(df_bkgTTBB["jet2eta"].values[i])
    jeteta3ATLAS.append(df_bkgTTBB["jet3eta"].values[i])
    jeteta4ATLAS.append(df_bkgTTBB["jet4eta"].values[i])
    jeteta5ATLAS.append(df_bkgTTBB["jet5eta"].values[i])
    jeteta6ATLAS.append(df_bkgTTBB["jet6eta"].values[i])
    jeteta7ATLAS.append(df_bkgTTBB["jet7eta"].values[i])
    jeteta8ATLAS.append(df_bkgTTBB["jet8eta"].values[i])
    jeteta9ATLAS.append(df_bkgTTBB["jet9eta"].values[i])
    jeteta10ATLAS.append(df_bkgTTBB["jet10eta"].values[i])
    jetphi1ATLAS.append(df_bkgTTBB["jet1phi"].values[i])
    jetphi2ATLAS.append(df_bkgTTBB["jet2phi"].values[i])
    jetphi3ATLAS.append(df_bkgTTBB["jet3phi"].values[i])
    jetphi4ATLAS.append(df_bkgTTBB["jet4phi"].values[i])
    jetphi5ATLAS.append(df_bkgTTBB["jet5phi"].values[i])
    jetphi6ATLAS.append(df_bkgTTBB["jet6phi"].values[i])
    jetphi7ATLAS.append(df_bkgTTBB["jet7phi"].values[i])
    jetphi8ATLAS.append(df_bkgTTBB["jet8phi"].values[i])
    jetphi9ATLAS.append(df_bkgTTBB["jet9phi"].values[i])
    jetphi10ATLAS.append(df_bkgTTBB["jet10phi"].values[i])


pdfname = 'sherpa-atlas.pdf'
with PdfPages(pdfname) as pdf:
    qPlot(numjetSHERPA,numjetATLAS,1,21,22,'Jet multiplicity')
    qPlot(numlepSHERPA,numlepATLAS,0,3,5,'Lepton multiplicity')
    qPlot(btagSHERPA, btagATLAS,0, 10, 10, 'N b-tagged jets')
    hPlot(srapSHERPA, srapATLAS, 0, 10, 20, r'$ < \eta(b_{i},b_{j}) >$')
    hPlot(centSHERPA, centATLAS,0, 1, 20, 'Centrality')
    hPlot(m_bbSHERPA, m_bbATLAS, 0, 250, 25, r'${M}_{bb}$ [GeV]')
    hPlot(h_bSHERPA, h_bATLAS,   0, 1500, 60, r'${H}_{B}$ [GeV]')
    hPlot(mt1SHERPA, mt1ATLAS,  0, 300, 100, r'${m}_{T}1$ [GeV]')
    hPlot(mt2SHERPA, mt2ATLAS,  0, 300, 100, r'${m}_{T}2$ [GeV]')
    hPlot(mt3SHERPA, mt3ATLAS,   0, 300, 100, r'${m}_{T}3$ [GeV]')
    hPlot(dr1SHERPA, dr1ATLAS,   0, 7, 100, r'$\Delta$R1')
    hPlot(dr2SHERPA, dr2ATLAS,  0, 7, 100, r'$\Delta$R2')
    hPlot(dr3SHERPA, dr3ATLAS,  0, 7, 100, r'$\Delta$R3')
    # hPlot(dr2SHERPA, dr2ATLAS,  0, 7, 100, r'$\Delta$R2')
    # hPlot(dr3SHERPA, dr3ATLAS,  0, 7, 100, r'$\Delta$R3')

    qPlot(jetbtag1SHERPA,jetbtag1ATLAS, 0, 1,3,'jet1btag')
    qPlot(jetbtag2SHERPA,jetbtag2ATLAS, 0, 1,3,'jet2btag')
    qPlot(jetbtag3SHERPA,jetbtag3ATLAS, 0, 1,3,'jet3btag')
    qPlot(jetbtag4SHERPA,jetbtag4ATLAS, 0, 1,3,'jet4btag')
    qPlot(jetbtag5SHERPA,jetbtag5ATLAS, 0, 1,3,'jet5btag')
    qPlot(jetbtag6SHERPA,jetbtag6ATLAS, 0, 1,3,'jet6btag')
    qPlot(jetbtag7SHERPA,jetbtag7ATLAS, 0, 1,3,'jet7btag')
    qPlot(jetbtag8SHERPA,jetbtag8ATLAS, 0, 1,3,'jet8btag')
    qPlot(jetbtag9SHERPA,jetbtag9ATLAS, 0, 1,3,'jet9btag')
    qPlot(jetbtag10SHERPA,jetbtag10ATLAS, 0, 1,3,'jet10btag')

    hPlot(jetpt1SHERPA, jetpt1ATLAS,  0, 1e6, 100, r'Jet1 pT')
    hPlot(jetpt2SHERPA, jetpt2ATLAS,  0, 1e6, 100, r'Jet2 pT')
    hPlot(jetpt3SHERPA, jetpt3ATLAS,  0, 1e6, 100, r'Jet3 pT')
    hPlot(jetpt4SHERPA, jetpt4ATLAS,  0, 1e6, 100, r'Jet4 pT')
    hPlot(jetpt5SHERPA, jetpt5ATLAS,  0, 1e6, 100, r'Jet5 pT')
    hPlot(jetpt6SHERPA, jetpt6ATLAS,  0, 1e6, 100, r'Jet6 pT')
    hPlot(jetpt7SHERPA, jetpt7ATLAS,  0, 1e6, 100, r'Jet7 pT')
    hPlot(jetpt8SHERPA, jetpt8ATLAS,   0, 1e6, 100, r'Jet8 pT')
    hPlot(jetpt9SHERPA, jetpt9ATLAS,  0, 1e6, 100, r'Jet9 pT')
    hPlot(jetpt10SHERPA, jetpt10ATLAS,  0, 1e6, 100, r'Jet10 pT')

    hPlot(jeteta1SHERPA, jeteta1ATLAS,   -6, 6, 12, r'Jet1 $\eta$')
    hPlot(jeteta2SHERPA, jeteta2ATLAS,   -6, 6, 12, r'Jet2 $\eta$')
    hPlot(jeteta3SHERPA, jeteta3ATLAS,   -6, 6, 12, r'Jet3 $\eta$')
    hPlot(jeteta4SHERPA, jeteta4ATLAS,  -6, 6, 12, r'Jet4 $\eta$')
    hPlot(jeteta5SHERPA, jeteta5ATLAS,  -6, 6, 12, r'Jet5 $\eta$')
    hPlot(jeteta6SHERPA, jeteta6ATLAS,   -6, 6, 12, r'Jet6 $\eta$')
    hPlot(jeteta7SHERPA, jeteta7ATLAS,   -6, 6, 12, r'Jet7 $\eta$')
    hPlot(jeteta8SHERPA, jeteta8ATLAS,   -6, 6, 12, r'Jet8 $\eta$')
    hPlot(jeteta9SHERPA, jeteta9ATLAS,  -6, 6, 12, r'Jet9 $\eta$')
    hPlot(jeteta10SHERPA, jeteta10ATLAS,   -6, 6, 12, r'Jet10 $\eta$')
 
    hPlot(jetphi1SHERPA, jetphi1ATLAS,  -4, 4, 8, r'Jet1 $\phi$') 
    hPlot(jetphi2SHERPA, jetphi2ATLAS, -4, 4, 8, r'Jet2 $\phi$')
    hPlot(jetphi3SHERPA, jetphi3ATLAS,  -4, 4, 8, r'Jet3 $\phi$')
    hPlot(jetphi4SHERPA, jetphi4ATLAS,   -4, 4, 8, r'Jet4 $\phi$')  
    hPlot(jetphi5SHERPA, jetphi5ATLAS,   -4, 4, 8, r'Jet5 $\phi$')
    hPlot(jetphi6SHERPA, jetphi6ATLAS,  -4, 4, 8, r'Jet6 $\phi$')  
    hPlot(jetphi7SHERPA, jetphi7ATLAS,   -4, 4, 8, r'Jet7 $\phi$')
    hPlot(jetphi8SHERPA, jetphi8ATLAS,  -4, 4, 8, r'Jet8 $\phi$')
    hPlot(jetphi9SHERPA, jetphi9ATLAS, -4, 4, 8, r'Jet9 $\phi$')
    hPlot(jetphi10SHERPA, jetphi10ATLAS,  -4, 4, 8, r'Jet10 $\phi$')


    

    d = pdf.infodict()
    d['Title'] = 'sherpa-atlas'
    d['Author'] = u'Jonathan O. Tellechea\xe4nen'
    d['Keywords'] = 'ttHH'
    # d['CreationDate'] = datetime.datetime(2009, 11, 13)
    # d['CreationDate'] = datetime.datetime.today()
    # d['ModDate'] = datetime.datetime.today()

print(pdfname)

if False: plt.show()