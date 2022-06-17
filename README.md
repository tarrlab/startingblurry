# startingblurry

Description of repo files:
- BWImgNetScratch.py: Experiment 2, control model (no pretrained model from
  experiment 1), trains on grayscale ImageNet images
- BWLinearBlurModel.py: Experiment 1, blurs linearly over 50 epochs, trains on
  grayscale Ecoset images
- BWLinearBlurModelImgNet.py: Experiment 2, uses pretrained models from
  BWLinearBlurModel.py, trains on grayscale ImageNet images
- BWNoBlurModel.py: Experiment 1, control model (no blur applied), trains on
  grayscale Ecoset images
- BWNoBlurModelImgNet.py: Experiment 2, uses pretrained models from
  BWNoBlurModel, trains on grayscale ImageNet images
- BWNonLinearBlurModel2.py: Experiment 1, blurs logarithmically (with a base of
  2) over 50 epochs, trains on grayscale Ecoset data
- BWNonLinearBlurModel2ImgNet.py: Experiment 2, uses pretrained models from
  BWNonLinearBlurModel2.py, trains on grayscale ImageNet images
- ColoredImgNetScratch.py: Experiment 2, control model (no pretrained model from
  experiment 1), trains on colored ImageNet images
- ColoredLinearBlurModel.py: Experiment 1, blurs linearly over 50 epochs, trains on
  colored Ecoset images
- ColoredLinearBlurModelImgNet.py: Experiment 2, uses pretrained models from
  ColoredLinearBlurModel.py, trains on colored ImageNet images
- ColoredNoBlurModel.py: Experiment 1, control model (no blur applied), trains on
  colored Ecoset images
- ColoredNoBlurModelImgNet.py: Experiment 2, uses pretrained models from
  ColoredNoBlurModel, trains on colored ImageNet images
- ColoredNonLinearBlurModel2.py: Experiment 1, blurs logarithmically (with a base of
  2) over 50 epochs, trains on colored Ecoset data
- ColoredNonLinearBlurModel2ImgNet.py: Experiment 2, uses pretrained models from
  ColoredNonLinearBlurModel2.py, trains on colored ImageNet images
- bOrS.csv: ImgNet categories labeled (basic or subordinate)
- createDictEcoset.py: important data preprocessing for ecoset data
- createDictImgNet.py: important data preprocessing for imagenet data

Let's get started!:
Before you start running code, in addition to all files included this github
repo, make sure your tree has the following folders (yes, the innermost folders
should be empty!):

- csvFiles
  - Ecoset
    - train
    - val
    - test  
  - ImgNet
    - train
    - val
- trials
  - BW
    - 1
    - 2
    - ...
    - 10
  - BWImgNet
    - 1
    - 2
    - ...
    - 10
  - Colored
    - 1
    - 2
    - ...
    - 10
  - ColoredImgNet
    - 1
    - 2
    - ...
    - 10

If you are doing more than 10 trials, make sure to create folders for those
trials as well. We suggest doing at least 10 trials for each condition to avoid
noise.

After you have the correct folder structure, we need to preprocess some data!
Run createDictEcoset.py and createDictImgNet.py (make sure to insert the
approrirate path names to your Ecoset/ImageNet data). These scripts will do
important data preprocessing, such as assigning ecoset/imgnet labels to int
labels and choosing a random subset of images to train on for each epoch.

Then, you can start running your models! Remember, you can only start running
your ImageNet/Experiment2 models after you finish running your
Ecoset/Experiment1 trials. Make sure for each new trials, you change the
trialNumber variable in your scripts! 
