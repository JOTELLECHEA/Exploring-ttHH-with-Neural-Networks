<img src="https://github.com/JOTELLECHEA/Exploring-ttHH-with-Neural-Networks/blob/master/Exploring_ttHH_with_Neural_Networks.png">

Conduct research involving prospective di-Higgs production at the HL-LHC by using machine learning via Neural networks.\
Advisor: Michael Hance.

# TBD (This readme is in the works last update is 2/2/2021)
# Scripts
These are all the scripts used in this repo. 
```bash
.
├── ./hyperparameterRecord.py
├── ./loadNN.py
├── ./multiNN.py
├── ./nnKerasGPU.py
├── ./plotsNeuralNetResults.py (TBD)
├── ./requirements.txt
├── ./rocCurve.py
├── ./rocplots.py
├── ./rocs.py
├── ./rootroot.py
├── ./rootTreeFlatten.py 
├── ./slug.py

```
#
1. To run scripts in repo a virtual enviorment is created.
   - Create a virtual enviroment run the scripts in this repo.
   Choose a name for your enviroment(env), replace ### with the name of your env.
   When you are done with the env, you can exit by typing 'off', to start the env 
   you have to source it. Then the required packages are installed.
  ```bash 
    $ python3 -m venv ###
    $ source ###/bin/activate
    (###)$
    (###)$ pip install -r requirements.txt
  ```
2. Programs Descriptions
   - nnKerasGPU.py \
     NN via Keras and tensorflow with GPU.The main(LAYER, BATCH, RATE), LAYER is the number of layers inlcuding the output layer, BATCH is the batch size, and 
     RATE is the dropout rate (The % of neurons randomly turned off) value must be between [0,1]. To run script follow example below:
     ```bash
     $ python -i nnKerasGPU.py
     >>> main(5,512,.01) 
     ```
   - multiNN.py \
     This will repeat nnKerasGPU.py multiple times to run it multiple times unsupervised.
     Must be edited to run scenario of interest.
     
   - slug.py \
     This script has functions that are used frquently and can be called to reduce code. 
     Functions of intrest are getZPoisson and ROC plot.
     
   - hyperparameterRecord.py \
     This script keeps a record of trained NN; Keeps track of time , AUC , lenght of NN etc. The filename for the saved weights is displayed to be used
     in loadNN.py to create plots. To run script follow example below:
       ```bash
           $ python hyperparameterRecord.py

                                                                   FileName    ConfusionMatrix [TP FP] [FN TN]                Run Time     AUC   Avg.P  Score  Max Signif  nsig   nbkg
       0    2020.12.21_15.16.10.numLayers5.numBranches69.batchSize512.GPU.h5   ([753820, 7884], [38234, 47062])  0 days 00:35:04.329170  0.9347  0.7506  0.921        2.21   107   2321
       1    2020.12.21_15.51.15.numLayers5.numBranches69.batchSize512.GPU.h5   ([754397, 7307], [38822, 46474])  0 days 00:39:51.617728  0.9349  0.7518  0.898        2.25   115   2590
       2    2020.12.21_16.31.08.numLayers5.numBranches69.batchSize512.GPU.h5   ([753857, 7847], [38537, 46759])  0 days 00:22:12.332931  0.9336  0.7478  0.939        2.18    86   1546
       3    2020.12.21_16.53.21.numLayers5.numBranches69.batchSize512.GPU.h5   ([754285, 7419], [38703, 46593])  0 days 00:34:29.965689  0.9348  0.7512  0.927        2.29    87   1419

       ```
     
   - loadNN.py 
     Script that uses NN score to create cuts on signal & background and plot the features and ROC. nnKerasGPU.py saves NN weights
     as HDF which has the extention 'h5'. These Files are saved in the ~/data/, for more information on specific h5 files. Using 
     hyperparameterRecord.py one can choose a NN that has been trained and have a record of the conditions the NN was trained in.
     (Currently a Jupyter NB version exist which displays the plots neatly and has a different statistical calculation). To run script 
     look at example below:
   
     ```bash
        $ python -i loadNN.py --file filename.h5
     ```
   - rootroot.py *** This scrip Requires ROOT *** \
     This script is used to check the data by plotting the lepton/jet four Energy-momentum components individually.
     The root file has to be edited (Automating script will be done in the future). To run script look at example below:
     
     ```bash
        $ python -i rootroot.py
     ```
   - rocCurve.py *** This script is only for earlier NN trained, which have a csv file saved *** \
     Uses a csv file to recreate roc and maxs signif. The csv files are in ~/data/. To run script look at example below:
     ```bash
        $ python rocCurve.py --file data/2020_11_14-rocDataNN-22.13.33.csv
        Score = 0.8901054
        Sign. = 0.88
        nsig. = 56
        nbkg. = 4074
     ```
   - rocs.py \
     Creates a csv file with fpr,tpr,bkgR from h5 file, to then be used by rocplots.py. These csv files are saved in ~/csv/.
     Three phase are avaliable (must be changed in script):
     * phase 1: High Level VARS
     * phase 2: Low Level VARS
     * phase 3: High and Low Level VARS

     ```bash
        $ python rocs.py
     ```
   - rocplots.py \
     Script that creates a modified ROC plot for Low, High, and Low + High VARS (Background rejection vs signal efficiency).
     The three csv files must be changed in script. 
     
      ```bash
           $ python -i rocplots.py
      ```
   - rootTreeFlatten.py *** This scrip Requires ROOT *** \
     Script that flattens branches in a ROOT file and adds new branches. Creates a copy of the ROOT file (new_filename.root) before flattening
     to preserve original file.
     
      ```bash
           $ python rootTreeFlatten.py --file filename.root
      ```
3. Images produced
   - ROC
      ![](https://github.com/JOTELLECHEA/neural_networks/blob/master/Images/Roc.png)
   - modifiedROC
     ![](https://github.com/JOTELLECHEA/neural_networks/blob/master/Images/modifiedRoc.png)


