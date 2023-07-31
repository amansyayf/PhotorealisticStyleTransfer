from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg19, vgg16
import torchvision.transforms as transforms
import Net


def get_gram_matrix(img):

    b, c, h, w = img.size()
    img = img.view(b*c, h*w)
    gram = torch.mm(img, img.t())
    return gram


def get_features(img, model, layers=None):

    if layers is None:
        layers = {
            '0': 'conv1_1',   # style layer
            '5': 'conv2_1',   # style layer
            '10': 'conv3_1',  # style layer
            '19': 'conv4_1',  # style layer
            '28': 'conv5_1',  # style layer

            '21': 'conv4_2'   # content layer
        }

    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


class Photorealistic_Style_transfer:
    def __init__(self, content_img, style_img, steps=50, style_weight=12, content_weight=400):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.style_img = self.__image_transform(style_img)
        self.content_img = self.__image_transform(content_img)

        self.steps = steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.vgg = vgg19(
            weights='VGG19_Weights.DEFAULT').features.to(self.device)
        self.style_weights = {'conv1_1': 0.1, 'conv2_1': 0.2,
                              'conv3_1': 0.4, 'conv4_1': 0.8, 'conv5_1': 1.6}

    def __image_transform(self, image):

        transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image = transform(image)

        return image.to(self.device)

    def __get_style_net(self):
        style_net = Net.HRNet()
        return style_net.to(self.device)

    def __get_optimizer(self, style_net):
        optimizer = optim.Adam(style_net.parameters(), lr=5e-3)
        return optimizer

    def __get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    def __layer_style_gram_matrixs(self, layer):
        style_features = get_features(self.style_img, self.vgg)
        style_gram_matrixs = {layer: get_gram_matrix(
            style_features[layer]) for layer in style_features}

        return style_gram_matrixs[layer]

    def __get_losses(self, target):
        content_features = get_features(self.content_img, self.vgg)

        target_features = get_features(target, self.vgg)
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0

        for layer in self.style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = get_gram_matrix(target_feature)
            style_gram_matrix = self.__layer_style_gram_matrixs(layer)

            layer_style_loss = self.style_weights[layer] * torch.mean(
                (target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c*h*w)

        return content_loss, style_loss

    def transfer(self):

        style_net = self.__get_style_net()
        optimizer = self.__get_optimizer(style_net)
        scheduler = self.__get_scheduler(optimizer)

        for epoch in range(0, self.steps+1):

            target = style_net(self.content_img).to(self.device)
            target.requires_grad_(True)

            content_loss, style_loss = self.__get_losses(target)

            total_loss = self.content_weight * content_loss + self.style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
        return target


class Art_Style_transfer:
    def __init__(self, content_img, style_img, steps=50, style_weight=1000, content_weight=10):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.style_img = self.image_transform(style_img)
        self.content_img = self.image_transform(content_img)

        # self.content_layers = ['conv_4']
        # self.style_layers = ['conv_2','conv_3', 'conv_4', 'conv_5']

        self.steps = steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.vgg = vgg16(
            weights='VGG16_Weights.DEFAULT').features.to(self.device)
        self.style_weights = {'conv1_1': 0.1, 'conv2_1': 0.2,
                              'conv3_1': 0.4, 'conv4_1': 0.8, 'conv5_1': 1.6}

    def image_transform(self, image):

        transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image = transform(image)

        return image.to(self.device)

    def get_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    def layer_style_gram_matrixs(self, layer):
        style_features = get_features(self.style_img, self.vgg)
        style_gram_matrixs = {layer: get_gram_matrix(
            style_features[layer]) for layer in style_features}

        return style_gram_matrixs[layer]

    def get_losses(self, target):
        content_features = get_features(self.content_img, self.vgg)

        target_features = get_features(target, self.vgg)
        content_loss = torch.mean(
            (target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0

        for layer in self.style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = get_gram_matrix(target_feature)
            style_gram_matrix = self.layer_style_gram_matrixs(layer)

            layer_style_loss = self.style_weights[layer] * torch.mean(
                (target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c*h*w)

        return content_loss, style_loss

    def transfer(self):

        target = self.content_img.clone().to(self.device)
        target.requires_grad_(True)
        optimizer = self.get_optimizer(target)
        scheduler = self.get_scheduler(optimizer)
        for epoch in range(0, self.steps+1):

            def closure():
                optimizer.zero_grad()
                content_loss, style_loss = self.get_losses(target)
                total_loss = self.content_weight * content_loss + self.style_weight * style_loss
                total_loss.backward()

                return self.content_weight * content_loss + self.style_weight * style_loss

            optimizer.step(closure)
            scheduler.step()
        return target
