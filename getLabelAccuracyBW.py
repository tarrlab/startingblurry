import collections
import torch
import torch.nn as nn
import os
import shutil
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import csv

cuda = torch.cuda.is_available()
print("Let's use", torch.cuda.device_count(), "GPUs!")
num_workers = 4 if cuda else 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=565, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers):
    model = ResNet(block, layers)
    return model

def resnet18():
    """ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34',BasicBlock, [3, 4, 6, 3])

def resnet50():
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3])

class ImageDataset(Dataset):
    def __init__(self, img_lst, label_lst, typeEval):
        self.img_lst = img_lst
        self.label_lst = label_lst
        self.typeEval = typeEval
        if self.typeEval == "train":
            self.current_indicies = range(50000)
        else:
            self.current_indicies = range(len(self.label_lst))

    def __len__(self):
        return len(self.current_indicies)

    def __getitem__(self, index):
        img = Image.open(self.img_lst[self.current_indicies[index]]).convert('L')
        width, height = img.size
        min_dim = min([width, height])
        img = torchvision.transforms.ToTensor()(img)
        cropped_img = transforms.CenterCrop(min_dim)(img)
        img = transforms.Resize(224)(cropped_img)
        label = self.label_lst[self.current_indicies[index]]
        return img, label

def load_data(typeEval):
    filename = 'csvFiles/ImageNet/'+typeEval+'/imageToLabelDict.csv'
    img_lst = []
    label_lst = []
    open_file = open(filename)
    read_file = csv.reader(open_file, delimiter="\t")
    i = 0
    for row in read_file:
        if i == 0:
            i+=1
            continue
        parsed_row = row[0].split(',')
        img_lst.append(parsed_row[1])
        label_lst.append(int(parsed_row[2]))
    open_file.close()
    return img_lst, label_lst


def get_int_label_to_folder(file):
    d = dict()
    open_file = open(file)
    read_file = csv.reader(open_file, delimiter="\t")
    i = 0
    for row in read_file:
        if i == 0:
            i+=1
            continue
        parsed_row = row[0].split(',')
        d[int(parsed_row[1])] = parsed_row[0]
    open_file.close()
    return d

def get_B_or_S(file):
    d = dict()
    open_file = open(file)
    read_file = csv.reader(open_file, delimiter="\t")
    i = 0
    for row in read_file:
        if i == 0:
            i+=1
            continue
        parsed_row = row[0].split(',')
        d[parsed_row[0]] = parsed_row[2]
    open_file.close()
    return d

int_label_to_folder = get_int_label_to_folder("csvFiles/ImageNet/labelDict.csv")

b_or_s = get_B_or_S("bOrS.csv")

assert(len(int_label_to_folder.values()) == 1000)


#Non-Linear

val_img_list, val_label_list = load_data('val')

val_dataset = ImageDataset(val_img_list, val_label_list, "val")

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)

filename = 'labelAccuracy/BW/NonLinearLabelAccuracy.csv'

B = []
S = []
for i in range(1, 11):
    final_d = dict()
    for val in range(1000):
        final_d[int_label_to_folder[val]] = []
    temp_B = []
    temp_S = []
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)
    loaded_state_dict = torch.load('trials/BWImgNet/'+ str(i) +'/BWNonLinearBlurModel2.pt')
    correct_state_dict = collections.OrderedDict([k[7:],v] for k,v in loaded_state_dict.items())
    model.load_state_dict(correct_state_dict)
    model = nn.DataParallel(model)
    model = model.to(device)

    model.eval()

    for batch_num, (x, y) in enumerate(val_dataloader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y = y[0]
        if torch.argmax(outputs).item() == y:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [1]
        else:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [0]

    for label in final_d:
        if b_or_s[label] == 'B':
            temp_B += final_d[label]
        else:
            temp_S += final_d[label]
    B.append(np.mean(temp_B))
    S.append(np.mean(temp_S))


results = [["Level", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "Average"]]
B = ["Basic"] + B + [np.mean(B)]
S = ["Subordinate"] + S + [np.mean(S)]
results.append(B)
results.append(S)
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(results)

#Linear

val_img_list, val_label_list = load_data('val')

val_dataset = ImageDataset(val_img_list, val_label_list, "val")

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)

filename = 'labelAccuracy/BW/LinearLabelAccuracy.csv'

B = []
S = []
for i in range(1, 11):
    final_d = dict()
    for val in range(1000):
        final_d[int_label_to_folder[val]] = []
    temp_B = []
    temp_S = []
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)
    loaded_state_dict = torch.load('trials/BWImgNet/'+ str(i) +'/BWLinearBlurModel.pt')
    '''
    You probably do not need correct_state_dict.

    When we saved our models in ecoset training script, for whatever reason, our
    keys were saved with the string 'module.' addedin front. Thus, when we loaded
    our saved model, our keys did not match.

    To get rid of this problem, we removed the first 7 characters in our saved
    keys (length of 'module.' = 7) and then loaded our modified state dictionary
    into our newly declared resnet model.

    In conclusion, pytorch can be wack! Make sure to not make strict = False
    when calling load_state_dict. If you get an error that the keys don't match,
    print out the keys of your saved model (loaded_state_dict) and the keys of your
    newly declared model (model) and see what differs between the two. If you do not
    see any key errors, you can call pass loaded_state_dict, instead of
    correct_state_dict, into the load_state_dict function call.
    '''
    correct_state_dict = collections.OrderedDict([k[7:],v] for k,v in loaded_state_dict.items())
    model.load_state_dict(correct_state_dict)
    model = nn.DataParallel(model)
    model = model.to(device)

    model.eval()

    for batch_num, (x, y) in enumerate(val_dataloader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y = y[0]
        if torch.argmax(outputs).item() == y:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [1]
        else:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [0]

    for label in final_d:
        if b_or_s[label] == 'B':
            temp_B += final_d[label]
        else:
            temp_S += final_d[label]
    B.append(np.mean(temp_B))
    S.append(np.mean(temp_S))


results = [["Level", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "Average"]]
B = ["Basic"] + B + [np.mean(B)]
S = ["Subordinate"] + S + [np.mean(S)]
results.append(B)
results.append(S)
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(results)

#No Blur

val_img_list, val_label_list = load_data('val')

val_dataset = ImageDataset(val_img_list, val_label_list, "val")

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)

filename = 'labelAccuracy/BW/NoBlurLabelAccuracy.csv'

B = []
S = []
for i in range(1, 11):
    temp_B = []
    temp_S = []
    final_d = dict()
    for val in range(1000):
        final_d[int_label_to_folder[val]] = []
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)
    loaded_state_dict = torch.load('trials/BWImgNet/'+ str(i) +'/BWNoBlurModel.pt')
    '''
    You probably do not need correct_state_dict.

    When we saved our models in ecoset training script, for whatever reason, our
    keys were saved with the string 'module.' addedin front. Thus, when we loaded
    our saved model, our keys did not match.

    To get rid of this problem, we removed the first 7 characters in our saved
    keys (length of 'module.' = 7) and then loaded our modified state dictionary
    into our newly declared resnet model.

    In conclusion, pytorch can be wack! Make sure to not make strict = False
    when calling load_state_dict. If you get an error that the keys don't match,
    print out the keys of your saved model (loaded_state_dict) and the keys of your
    newly declared model (model) and see what differs between the two. If you do not
    see any key errors, you can call pass loaded_state_dict, instead of
    correct_state_dict, into the load_state_dict function call.
    '''
    correct_state_dict = collections.OrderedDict([k[7:],v] for k,v in loaded_state_dict.items())
    model.load_state_dict(correct_state_dict)
    model = nn.DataParallel(model)
    model = model.to(device)

    model.eval()

    for batch_num, (x, y) in enumerate(val_dataloader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y = y[0]
        if torch.argmax(outputs).item() == y:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [1]
        else:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [0]

    for label in final_d:
        if b_or_s[label] == 'B':
            temp_B += final_d[label]
        else:
            temp_S += final_d[label]
    B.append(np.mean(temp_B))
    S.append(np.mean(temp_S))


results = [["Level", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "Average"]]
B = ["Basic"] + B + [np.mean(B)]
S = ["Subordinate"] + S + [np.mean(S)]
results.append([B])
results.append([S])
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(results)

#Control

val_img_list, val_label_list = load_data('val')

val_dataset = ImageDataset(val_img_list, val_label_list, "val")

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory = True, drop_last=False)

filename = 'labelAccuracy/BW/ControlLabelAccuracy.csv'

B = []
S = []
for i in range(1, 11):
    temp_B = []
    temp_S = []
    final_d = dict()
    for val in range(1000):
        final_d[int_label_to_folder[val]] = []
    model = resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)
    loaded_state_dict = torch.load('trials/BWImgNet/'+ str(i) +'/BWModel.pt')
    '''
    You probably do not need correct_state_dict.

    When we saved our models in ecoset training script, for whatever reason, our
    keys were saved with the string 'module.' addedin front. Thus, when we loaded
    our saved model, our keys did not match.

    To get rid of this problem, we removed the first 7 characters in our saved
    keys (length of 'module.' = 7) and then loaded our modified state dictionary
    into our newly declared resnet model.

    In conclusion, pytorch can be wack! Make sure to not make strict = False
    when calling load_state_dict. If you get an error that the keys don't match,
    print out the keys of your saved model (loaded_state_dict) and the keys of your
    newly declared model (model) and see what differs between the two. If you do not
    see any key errors, you can call pass loaded_state_dict, instead of
    correct_state_dict, into the load_state_dict function call.
    '''
    correct_state_dict = collections.OrderedDict([k[7:],v] for k,v in loaded_state_dict.items())
    model.load_state_dict(correct_state_dict)
    model = nn.DataParallel(model)
    model = model.to(device)

    model.eval()

    for batch_num, (x, y) in enumerate(val_dataloader):
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y = y[0]
        if torch.argmax(outputs).item() == y:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [1]
        else:
            final_d[int_label_to_folder[int(y)]] = final_d[int_label_to_folder[int(y)]] + [0]

    for label in final_d:
        if b_or_s[label] == 'B':
            temp_B += final_d[label]
        else:
            temp_S += final_d[label]
    B.append(np.mean(temp_B))
    S.append(np.mean(temp_S))

results = [["Level", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "Average"]]
B = ["Basic"] + B + [np.mean(B)]
S = ["Subordinate"] + S + [np.mean(S)]
results.append(B)
results.append(S)
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(results)
