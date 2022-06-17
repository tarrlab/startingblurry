# startingblurry

Things to include: 
- need to make a trials folder, with subfolders BW, BWImgNet, Colored, ColoredImgNet, and subsubfolders 1-10 (10 trials) 
- need to make csvFiles folder, with Ecoset and ImgNet subfolders. Each of these subfolders should have a train and val subsubfolder. Ecoset should have a test subsubfolder as well. Train folders need to have imagesByEpoch folder. 
- need to generate a labelDict.csv file for both Ecoset and ImgNet subfolders using _ script
- For all train, val, and test folders, need to generate a imageToLabelDict.csv using _ script
- need to randomly select 50K images per epoch for Ecoset and ImgNet and save them under their respective imagesByEpoch folder using the _ script 
