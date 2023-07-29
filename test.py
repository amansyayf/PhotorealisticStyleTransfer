import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import asyncio 
     

import style_transfer

def image_loading(path, size=None):
  img = Image.open(path)

  if size is not None:
    img = img.resize((size, size))

  transform = transforms.Compose([
      transforms.ToTensor()
  ])

  img = transform(img)
  img = img.unsqueeze(0)
  return img


def im_convert(img):
    img = img.to('cpu').clone().detach()
    img = img.numpy().squeeze(0)
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    return img

class Args:
    def __init__(self):
        self.img_root = 'C:/Users/Asus/devel/images'
        self.content_img = 'door.png'
        self.style_img = 'starry_night.png'
        self.use_gpu = True

args = Args()

content_img = image_loading(args.content_img, size=300)
style_img = image_loading(args.style_img, size = 300)


# st_object = style_transfer.Art_Style_transfer(content_img, style_img)
# target = asyncio.run(st_object.transfer())



# plt.imshow(im_convert(target))
# plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg19(weights='VGG19_Weights.DEFAULT').features.to(device)

def get_features(img, model, layers=None):

    if layers is None:
        layers = {
            '0': 'conv1_1',   # style layer
            # '5': 'conv2_1',   # style layer
            # '10': 'conv3_1',  # style layer
            # '19': 'conv4_1',  # style layer
            # '28': 'conv5_1',  # style layer

            '21': 'conv4_2'   # content layer
        }

    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        print(name)
        if name in layers:
            features[layers[name]] = x

    return features

def image_transform( image):

    transform = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)

    return image.to(device)



content_features = get_features(image_transform(content_img), vgg)