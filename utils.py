import numpy as np
import logging
import config
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from PIL import Image
import PIL
import config
from aiogram import Bot
import style_transfer as st
import asyncio



def img_to_torch(path, size = 300):
    img = Image.open(path)

    if size is not None:
      img = img.resize((size, size))

    transform = transforms.Compose([
      transforms.ToTensor()
    ])

    img = transform(img)
    img = img.unsqueeze(0)

    return img

def tensor_to_img(image, normalize = True):

    image = image.to('cpu').detach().squeeze(0)
    
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    if normalize:
      image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save('result.jpg')


def make_photorealistic_style_transfer(*imgs):
    content_img = img_to_torch(imgs[0])
    style_img = img_to_torch(imgs[1])

    style_transfer = st.Photorealistic_Style_transfer(content_img, style_img)
    output_img = style_transfer.transfer()

    tensor_to_img(output_img)


def make_art_style_transfer(*imgs):
    content_img = img_to_torch(imgs[0])
    style_img = img_to_torch(imgs[1])

    style_transfer = st.Art_Style_transfer(content_img, style_img)
    output_img = style_transfer.transfer()

    tensor_to_img(output_img, normalize=False)

 

    

