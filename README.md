# startingblurry

### Contents of this repository

#### Data folders (all will be empty initially):
- csvFiles: Contains important data to run models. Should be empty initially,
  need to run createDictEcoset.py and createDictImgNet.py to fill folder with
  necessary files.
- trials: Contains training logs and best models for each trial.
- labelAccuracy: Contains data on accuracy between basic- and subordinate-level
  categories. To get data, you'll need to run getLabelAccuracyBW.py and
  getLabelAccuracyColored.py AFTER finishing all trials. If you decided to run
  more/less than 10 trials, you'll need to alter the script to accomodate for
  such changes.

#### Code files:
##### General
- bOrS.csv: ImageNet categories labeled (basic or subordinate)
- createDictEcoset.py: important data preprocessing for Ecoset data
- createDictImgNet.py: important data preprocessing for Imagenet data
- getLabelAccuracyBW.py: get accuracy of basic vs subordinate categories using
  grayscale models from experiment 2
- getLabelAccuracyColored.py: get accuracy of basic vs subordinate categories
  using colored models from experiment 2
##### Training for Experiment 1
- BWLinearBlurModel.py: blurs linearly over 50 epochs, trains on grayscale
  Ecoset images
- BWNoBlurModel.py: control model (no blur applied), trains on grayscale Ecoset
  images
- BWNonLinearBlurModel2.py: blurs logarithmically (with a base of 2) over 50
  epochs, trains on grayscale Ecoset data
- ColoredLinearBlurModel.py: blurs linearly over 50 epochs, trains on colored
  Ecoset images
- ColoredNoBlurModel.py: control model (no blur applied), trains on colored
  Ecoset images  
- ColoredNonLinearBlurModel2.py: blurs logarithmically (with a base of 2) over
  50 epochs, trains on colored Ecoset data
##### Training for Experiment 2
- BWImgNetScratch.py: control model (no pretrained model from experiment 1),
  trains on grayscale ImageNet images
- BWLinearBlurModelImgNet.py: uses pretrained models from BWLinearBlurModel.py,
  trains on grayscale ImageNet images
- BWNoBlurModelImgNet.py: uses pretrained models from BWNoBlurModel, trains on
  grayscale ImageNet images
- BWNonLinearBlurModel2ImgNet.py: uses pretrained models from
  BWNonLinearBlurModel2.py, trains on grayscale ImageNet images
- ColoredImgNetScratch.py: control model (no pretrained model from experiment 1),
  trains on colored ImageNet images
- ColoredLinearBlurModelImgNet.py: uses pretrained models from
  ColoredLinearBlurModel.py, trains on colored ImageNet images
- ColoredNoBlurModelImgNet.py: uses pretrained models from ColoredNoBlurModel,
  trains on colored ImageNet images
- ColoredNonLinearBlurModel2ImgNet.py: uses pretrained models from
  ColoredNonLinearBlurModel2.py, trains on colored ImageNet images

### Steps to run the code
<ol>
  <li> Clone this repository
    <ul>
      <li>git clone https://github/com/ojinsi/startingblurry
    </ul>
  <li> Download the Ecoset and ImageNet datasets
     <ul>
       <li> Ecoset: https://codeocean.com/capsule/9570390/tree/v1   
       <li> ImageNet: https://www.image-net.org/download.php
    </ul>
  <li> Prepare the ImageNet database
    <ul>
      <li> On initial download, the validation images will all be in a single folder rather than separated into sub-folders by category. We'll need them in subfolders to continue. To do this, use the script "valprep.sh" provided in this repo (this should be run from the "val" folder in your ImageNet directory).
    </ul>
  <li> Preprocess the data
    <ul>
      <li> Change the path names at the top of createDictEcoset.py and createDictImgNet.py to the folders where you've downloaded Ecoset and ImageNet, respectively.
      <li> Run createDictEcoset.py and createDictImgNet.py. These scripts will do
  important data preprocessing, such as assigning ecoset/imgnet labels to int
  labels and choosing a random subset of images for each epoch.
    </ul>
  <li> Start running your models!
    <ul>
      <li> Use the code provided to run models of each type (details of each code file described above)
      <li> Note that all model training will require PyTorch and access to at least one GPU (4 GPUs recommended).
      <li> By default, each script has "trialNumber" set to 1, which will run the first trial only. Make sure to change the trialNumber variable in your scripts whenever you run a new model, else you'll rewrite your previous trial run!
      <li> Remember, you can only start running
  your Experiment 2 models after you finish running your Experiment 1 models.
      <li> The results for Experiment 1 should be saved under BW/Colored and the results
  for Experiment 2 should be saved under BWImgNet/ColoredImgNet. In experiment 1,
  there should be a total of 60 models (2 color conditions * 3 blur conditions *
  10 trials) while in experiment 2, there should be a total of 80 models (2 color
  conditions * 4 pretrained model conditions * 10 trials).
    </ul>
  <li> Analyze results
      <ul>
        <li> To examine the difference between accuracy between basic-
  and subordinate-level categories in the models trained in Experiment 2, you
  can run getLabelAccuracyBW.py and getLabelAccuracyColored.py. Results can be
  found under the folder "labelAccuracy".
        <li>Use the provided jupyter notebooks (in folder "analysis") to perform statistics and visualize results.
    </ul>

</ol>
