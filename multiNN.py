# Written By : Jonathan O. Tellechea
# Adviser    : Mike Hance, Phd
# Research   : Using a neural network to maximize the significance of tttHH production.
# Description: This will repeat nnKerasGPU.py multiple times to run it multiple times unsupervised. Must be edited to run scenario of interest.
# Reference  :http://cdsweb.cern.ch/record/2220969/files/ATL-PHYS-PUB-2016-023.pdf
###########################################################################################################################
# Imported packages.
import nnKerasGPU as nn

# This runs nnKerasGPU.py multiple times. A for loop can be used to varry the parameters. 
for i in range(1,4):
	nn.main(i+4,1024,0)




