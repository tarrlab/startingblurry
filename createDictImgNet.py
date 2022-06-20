import os
import csv
from random import sample

### ENTER YOUR IMAGENET PATH HERE #####
imagenet_path = '/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC/'
train_path = os.path.join(imagenet_path, 'train')
val_path = os.path.join(imagenet_path, 'val')

def parse_data(datadir, target_dict = None):
    img_list = []
    ID_list = []
    uniqueID_list = []
    items = os.listdir(datadir);
    folders = [it for it in items if os.path.isdir(os.path.join(datadir, it)) and '.DS_STORE' not in it]
    for direc in folders:      
        curr_path = os.path.join(datadir, direc)
        uniqueID_list.append(direc)
        for filename in os.listdir(curr_path):
          if filename != '.DS_Store':
            img_list.append(datadir+"/"+direc+"/"+filename)
            ID_list.append(direc)

    # construct a dictionary, where key and value correspond to ID and target
    class_n = len(uniqueID_list)
    if not target_dict:
        target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), class_n))
    return img_list, label_list, class_n, target_dict

# TRAIN

img_list, label_list, class_n, target_dict = parse_data(train_path)
total_len = len(img_list)
lstOfInds = range(total_len)

#make csv of original label to new label
filename = "csvFiles/ImgNet/labelDict.csv"
header = ["Orginal Label", "New Label"]
all_labels = target_dict.items()
with open(filename, 'w') as csvfile:
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(header)
  csvwriter.writerows(all_labels)

#make csv of all images and labels
filename = "csvFiles/ImgNet/train/imageToLabelDict.csv"
header = ["Order", "Image", "Label"]
results = []
for i in range(total_len):
    results.append([str(i), img_list[i], label_list[i]])
with open(filename, 'w') as csvfile:
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(header)
  csvwriter.writerows(results)

#make csv for each epoch
header = ["Epoch", "Order", "Image", "Label", "Index"]
#We did a maximum of 300 epochs but chose random images for 1000 epochs in case
#we wanted to play around with more epochs!
for i in range(1000):
  inds_for_epoch = sample(lstOfInds,50000)
  filename = "csvFiles/ImgNet/train/imagesByEpoch/epoch" + str(i) + ".csv"
  final_result = []
  #create csv for epoch i
  for j in range(len(inds_for_epoch)):
    ind = inds_for_epoch[j]
    img = img_list[ind]
    label = label_list[ind]
    epoch = i
    order = j
    final_result.append([epoch, order, img, label, ind])
  #write into file
  with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerows(final_result)

#VAL
img_list, label_list, class_n, target_dict = parse_data(val_path, target_dict)
total_len = len(img_list)

#make csv of all images and labels
filename = "csvFiles/ImgNet/val/imageToLabelDict.csv"
header = ["Image", "Label"]
for i in range(total_len):
    results.append([str(i), img_list[i], label_list[i]])
with open(filename, 'w') as csvfile:
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(header)
  csvwriter.writerows(results)
