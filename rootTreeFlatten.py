#!/usr/bin/env python2
# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: Script that Flattens ROOT branches and adds needed high level VARS.
#######################################################################################

# Imported packages.
import os, time, sys, argparse, math
import numpy as np
from array import array
import shutil
import random
import itertools

seed = 42
random.seed(seed)
import ROOT
rand = random.random()


def augment_rootfile(filepath):

    # Creates a copy of ROOT file and adds 'new_' to the beginning.
    shutil.copyfile(filepath, "new_" + filepath)

    # This automatically detects which data sample is passed.
    truthLabel = 137
    if filepath == "TTHH.root":
        truthLabel = 0
    elif filepath == "TTBB.root":
        truthLabel = 1
    elif filepath == "TTH.root":
        truthLabel = 2
    elif filepath == "TTZ.root":
        truthLabel = 3
    else:
        print("Invalid ROOT file")
        sys.exit()

    # Tree name inside ROOT File.
    treename = "OutputTree"

    # Gives access to new file and allows writting.
    t = ROOT.TFile("new_" + filepath, "UPDATE")

    # ROOT tree object. Retrieving tree object to be accessed.
    tree = t.Get(treename)

    # Define branches as an array.
    numlep = array("f", [0])
    numjet = array("f", [0])
    weights = array("f", [0])
    btag = array("f", [0])
    srap = array("f", [0])
    cent = array("f", [0])
    m_bb = array("f", [0])
    h_b = array("f", [0])
    chi = array("f", [0])
    truth = array("f", [0])
    Met = array('f',[0])
    MetPhi = array('f',[0])

    # Create dictionary to automate filling lepton features.
    leptonpT = {}
    leptoneta = {}
    leptonphi = {}
    leptonflav = {}
    mt = {}
    dr = {}

    # Defining branches from lepton dictionaries.
    maxlepton = 4
    for i in range(1, maxlepton):
        leptonpT[i] = array("f", [0])
        leptoneta[i] = array("f", [0])
        leptonphi[i] = array("f", [0])
        leptonflav[i] = array("f", [0])
        mt[i] = array("f", [0])
        dr[i] = array("f", [0])

    # Create branches to be added to ROOT tree object for lepton features.
    br_leptonpT = {}
    br_leptoneta = {}
    br_leptonphi = {}
    br_leptonflav = {}
    br_mt = {}
    br_dr = {}

    # Assign branches to tree object for lepton features.
    for i in range(1, maxlepton):
        br_leptonpT[i] = tree.Branch("lepton%dpT" % i, leptonpT[i], "lepton%dpT/F" % i)
        br_leptoneta[i] = tree.Branch(
            "lepton%deta" % i, leptoneta[i], "lepton%deta/F" % i
        )
        br_leptonphi[i] = tree.Branch(
            "lepton%dphi" % i, leptonphi[i], "lepton%dphi/F" % i
        )
        br_leptonflav[i] = tree.Branch(
            "lepton%dflav" % i, leptonflav[i], "lepton%dflav/F" % i
        )
        br_mt[i] = tree.Branch("mt%d" % i, mt[i], "mt%d/F" % i)
        br_dr[i] = tree.Branch("dr%d" % i, dr[i], "dr%d/F" % i)

    # Create dictionary to automate filling jet features.
    jetpT = {}
    jeteta = {}
    jetphi = {}
    jetbtag = {}

    # Defining branches from jet dictionaries.
    maxjets = 22
    for i in range(1, maxjets):
        jetpT[i] = array("f", [0])
        jeteta[i] = array("f", [0])
        jetphi[i] = array("f", [0])
        jetbtag[i] = array("f", [0])


    # Create branches to be added to ROOT tree object for jet features.
    br_jetpT = {}
    br_jeteta = {}
    br_jetphi = {}
    br_jetbtag = {}

    # Assign branches to tree object for jet features.
    for i in range(1, maxjets):
        br_jetpT[i] = tree.Branch("jet%dpT" % i, jetpT[i], "jet%dpT/F" % i)
        br_jeteta[i] = tree.Branch("jet%deta" % i, jeteta[i], "jet%deta/F" % i)
        br_jetphi[i] = tree.Branch("jet%dphi" % i, jetphi[i], "jet%dphi/F" % i)
        br_jetbtag[i] = tree.Branch("jet%dbtag" % i, jetbtag[i], "jet%dbtag/F" % i)


    # Assign branches to tree object for High level features.
    br_numlep = tree.Branch("numlep", numlep, "numlep/F")
    br_truth = tree.Branch("truth", truth, "truth/F")
    br_numjet = tree.Branch("numjet", numjet, "numjet/F")
    br_weights = tree.Branch("weights", weights, "weights/F")
    br_btag = tree.Branch("btag", btag, "btag/F")
    br_cent = tree.Branch("cent", cent, "cent/F")
    br_srap = tree.Branch("srap", srap, "srap/F")
    br_m_bb = tree.Branch("m_bb", m_bb, "m_bb/F")
    br_h_b = tree.Branch("h_b", h_b, "h_b/F")
    br_chi = tree.Branch("chi", chi, "chi/F")
    br_Met = tree.Branch('met',Met,'met/F')
    br_MetPhi =tree.Branch('metPhi',MetPhi,'metPhi')

    # Start time for progress bar.
    start_time = time.clock()

    ### Fucnctions START ###

    # Calculates missing transverse momentum.
    def missingPT(x):
        met_value = ROOT.TMath.Sqrt(
            2
            * event.met[0]
            * event.leppT[x]
            / (10 ** 6)
            * (1 - ROOT.TMath.Cos((lepvec[x].DeltaPhi(neutrino[0]))))
        )
        return met_value

    # Calculates average separation in pseudorapidity between two b-tagged jets.
    def etabi_j(x, y):
        distance = abs(jetvec[tracker_btj[x]].Eta() - jetvec[tracker_btj[y]].Eta())
        return distance

    # Vector Pt or M Sum between two b-tagged jets.
    def vectorsum(x, y, c):
        if c == "Pt":
            sum = (jetvec[tracker_btj[x]] + jetvec[tracker_btj[y]]).Pt()
        elif c == "M":
            sum = (jetvec[tracker_btj[x]] + jetvec[tracker_btj[y]]).M()
        return sum

    def btaggedjet(truebhadron,truechadron):
        # 70% of truth-b-jets are labeled as b-tagged
        if truebhadron == 1 and rand <= 0.7:
            return 1

        # 20% mistag rate for c-jets.
        elif truechadron == 1 and rand <= 0.2:
            return 1

        # 0.2% mistag rate for light-jets.
        elif rand <= 0.002:
            return 1

        # Remaining Jets must be tagged c-jets.
        else:
            return 0

    ### Fucnctions END ###

    # Number of events.
    n_entries = tree.GetEntries()

    # Dummy variable. For progress bar.
    i = 1

    # Looping through tree by event(iterator).
    for event in tree:

        ### Initialize VARS START ###

        tracker_btj = []  # List for tracking btagjets.
        tracker_non = []  # List for tracking lightjets.
        dR1 = []  # List for Delta R values for jets, ands lepton 1.
        dR2 = []  # List for Delta R values for jets, ands lepton 2.
        dR3 = []  # List for Delta R values for jets, ands lepton 3.
        chisq = []  # Chi Square list.
        lepvec = {}  # Lepton ROOT Four Vector.
        jetvec = {}  # Jet ROOT Four Vector.
        neutrino = {}  # Neutrino ROOT Fpur Vector.
        HB_sum_Pt = 0.0  # Initialize sum of Pt for all b-tag jets.
        # rand = 0.0  # Initialize rand value.
        cen_sum_Pt = 0.0  # Initialize sum of Pt for all jets.
        cen_sum_E = 0.0  # Initialize sum of E for all jets.
        etasum = 0.0  # Initialize sum for eta seperation.
        etasum_N = 0.0  # Initialize sum for eta separation average.
        btjmaxPt = 0.0  # Initialize empty b-tag vecto for max .Pt().
        btjmaxM = 0.0  # Initialize empty b-tag vecto for max .M().
        vec_sum_Pt = 0.0  # Initialize empty b-tag vector for summing Pt().
        vec_sum_M = 0.0  # Initialize empty b-tag vector for summing M().
        numlep = -9
        numjet = -9
        truth = 137
        weights = 137
        btag = -9
        srap = -9
        cent = -9
        m_bb = -9
        h_b = -9
        chi = 0 
        Met = 0
        MetPhi = -99
        btagjets = -9
        ntagjets = -9



        ### Initialize VARS END ###

        # Show some progress
        if i % 1000 == 0:
            print(
                "   processing entry {:8d}/{:d} [{:5.0f} evts/s]".format(
                    i, n_entries, i / (time.clock() - start_time)
                )
            )

        ### Setting values START ###
        numlep[0] = lep = event.nlep[0]  # Number of Leptons.
        numjet[0] = jet = event.njet[0]  # Number of Jets.
        weights[0] = event.mcweight[0]  # Weights of event.
        truth[0] = truthLabel  # Label to identify data sample.
        Met[0] = event.met[0]
        MetPhi[0]=event.met_phi[0]

        # 1 Lepton minimum requirment.

        # Create ROOT Four Vector for Neutrinos.
        neutrino[0] = ROOT.TLorentzVector()

        # Neutrinos Four Vector set.
        neutrino[0].SetPtEtaPhiM(event.met[0], 0, event.met_phi[0], 0)

        # Loop through Leptons.
        for j in xrange(lep):

            # Create ROOT Four Vector for Leptons.
            lepvec[j] = ROOT.TLorentzVector()

            # Leptons Four Vector set.
            lepvec[j].SetPtEtaPhiM(
                event.leppT[j], event.lepeta[j], event.lepphi[j], 0
            )

        # Loop through Jets.
        for k in xrange(jet):

            # Create ROOT Four Vector for Jets.
            jetvec[k] = ROOT.TLorentzVector()

            # Jets Four Vector set.
            jetvec[k].SetPtEtaPhiM(
                event.jetpT[k], event.jeteta[k], event.jetphi[k], 0
            )

            # Getting Delta R Values. If statments needed to insure lepton exist.
            if lep > 0:
                dR1.append(lepvec[0].DeltaR(jetvec[k]))
            if lep > 1:
                dR2.append(lepvec[1].DeltaR(jetvec[k]))
            if lep > 2:
                dR3.append(lepvec[2].DeltaR(jetvec[k]))

        # Loop through Jets.
        for x in xrange(jet):

            # Scalar sum of E.
            cen_sum_E += jetvec[x].E()

            # Scalar sum of Pt.
            cen_sum_Pt += jetvec[x].Pt()

            # 70% of truth-b-jets are labeled as b-tagged
            if event.jetbhadron[x] == 1 and rand <= 0.7:
                tracker_btj.append(x)

            # 20% mistag rate for c-jets.
            elif event.jetchadron[x] == 1 and rand <= 0.2:
                tracker_btj.append(x)

            # 0.2% mistag rate for light-jets.
            elif rand <= 0.002:
                tracker_btj.append(x)

            # Remaining Jets must be tagged c-jets.
            else:
                tracker_non.append(x)

        # Number of b-tagged jets.
        btagjets = len(tracker_btj)
        btag[0] = btagjets

        # Number of non b-tagged jets.
        ntagjets = len(tracker_non)

        # Prevent division by Zero.
        if cen_sum_E != 0:
            # Scalar sum of Pt/E.
            cent[0] = cen_sum_Pt / cen_sum_E
        else:
            # Dummy Value to avoid empty entry.
            cent[0] = -9999

        # Loop through b-tagged jets.
        for k in xrange(btagjets):

            # Sum of Pt for all b-tag jets.
            HB_sum_Pt += jetvec[tracker_btj[k]].Pt()

            # Loop through b-tagged jets.
            for j in xrange(btagjets):

                # Ignore loop with same jet.
                if k == j:
                    continue

                # Finding separation between all b_jets.
                etasum += etabi_j(k, j)

                # Sum of btagjets Pt.
                vec_sum_Pt = vectorsum(k, j, "Pt")

                # Sum of btagjets M.
                vec_sum_M = vectorsum(k, j, "M")

                # Finds max Pt and M for two btagjets.
                if vec_sum_Pt < btjmaxPt:
                    continue
                btjmaxPt = vec_sum_Pt
                btjmaxM = vec_sum_M

        # Scale to GeV.
        m_bb[0] = btjmaxM / 1000
        h_b[0] = HB_sum_Pt / 1000

        # Average separation in pseudorapidity between two b-tagged jets (srap)
        if btagjets > 1:
            etasum_N = etasum / (btagjets ** 2 - btagjets)
        else:
            # Dummy Value to avoid empty entry.
            etasum_N = -999

        # Srap
        srap[0] = etasum_N

        # Chisquare.
        if btagjets >= 6:
            for o in xrange(0, 6):
                for j in xrange(1, 6):
                    for k in xrange(2, 6):
                        for l in xrange(3, 6):
                            if o == j == k == l:
                                continue
                            if o > j > k > l:
                                continue
                            if j > k > l:
                                continue
                            if k > l:
                                continue
                            if o > j:
                                continue
                            if j > k:
                                continue
                            if o == j == k:
                                continue
                            if o == j:
                                continue
                            if o == k:
                                continue
                            if o == l:
                                continue
                            if j == k == l:
                                continue
                            if j == k:
                                continue
                            if j == l:
                                continue
                            if k == l:
                                continue
                            chisq.append(
                                (vectorsum(o, j, "M") - 120000) ** 2
                                + (vectorsum(k, l, "M") - 120000) ** 2
                            )
        # Checking if chiaquare exist.
        if len(chisq) > 0:
            chi[0] = min(chisq)

        # Dummy Value to avoid empty entry.
        else:
            chi[0] = -999

    # Combine the delta R.
    delta_r = [dR1, dR2, dR3]

    # Loop through leptons: set Four Vector components, delata R and flavor.
    lloop = range(1, maxlepton)
    for n in lloop:
        if n <= lep:

            # Using n-1 beacuse we start count at 1 insted of 0.
            leptonpT[n][0] = event.leppT[n - 1]
            leptoneta[n][0] = event.lepeta[n - 1]
            leptonphi[n][0] = event.lepphi[n - 1]
            leptonflav[n][0] = event.lepflav[n - 1]
            mt[n][0] = missingPT(n - 1)

            if len(delta_r[n - 1]) == 0:
                dr[n][0] = -999
            else:
                dr[n][0] = min(delta_r[n - 1])

        else:

            # Dummy Values to avoid empty entry.
            leptonpT[n][0] = -999
            leptoneta[n][0] = -9
            leptonphi[n][0] = -9
            leptonflav[n][0] = -999
            mt[n][0] = -999
            dr[n][0] = -999

    # Loop through jets: set Four Vector components and flavor.
    jloop = range(1, maxjets)
    for n in jloop:
        if n <= jet:

            # Using n-1 beacuse we start count at 1 insted of 0.
            jetpT[n][0] = event.jetpT[n - 1]
            jeteta[n][0] = event.jeteta[n - 1]
            jetphi[n][0] = event.jetphi[n - 1]
            jetbtag[n][0] = btaggedjet(event.jetbhadron[n - 1],event.jetchadron[n-1])

        else:

            # Dummy Values to avoid empty entry.
            jetpT[n][0] = -999
            jeteta[n][0] = -9
            jetphi[n][0] = -9
            jetbtag[n][0] = -9

    ### Setting values End ###

    # Fill new branches with set values.
    for n in lloop:
        br_leptonpT[n].Fill()
        br_leptoneta[n].Fill()
        br_leptonphi[n].Fill()
        br_leptonflav[n].Fill()
        br_mt[n].Fill()
        br_dr[n].Fill()

    for n in jloop:
        br_jetpT[n].Fill()
        br_jeteta[n].Fill()
        br_jetphi[n].Fill()
        br_jetbtag[n].Fill()


    br_numjet.Fill()
    br_numlep.Fill()
    br_weights.Fill()
    br_btag.Fill()
    br_cent.Fill()
    br_srap.Fill()
    br_m_bb.Fill()
    br_h_b.Fill()
    br_chi.Fill()
    br_truth.Fill()
    br_Met.Fill()
    br_MetPhi.Fill()

    i += 1

    # Write augmented tree to copied file.
    tree.Write("", ROOT.TObject.kOverwrite)

    # Closes ROOT File and frees memory.
    t.Close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="flatten/augment ntuples")
    parser.add_argument("--file", help="input ROOT file")
    args = parser.parse_args()
    augment_rootfile(args.file)
