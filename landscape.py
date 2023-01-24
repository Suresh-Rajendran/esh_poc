from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from activation import *

def build_model(activation_function):
    return nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(2, 64)),
                ("activation1", activation_function),  # use custom activation function
                ("fc2", nn.Linear(64, 32)),
                ("activation2", activation_function),
                ("fc3", nn.Linear(32, 16)),
                ("activation3", activation_function),
                ("fc4", nn.Linear(16, 1)),
                ("activation4", activation_function),
            ]
        )
    )


def convert_to_PIL(img, width, height):
    img_r = img.reshape(height, width)

    pil_img = Image.new("RGB", (height, width), "white")
    pixels = pil_img.load()

    for i in range(0, height):
        for j in range(0, width):
            pixels[j, i] = img[i, j], img[i, j], img[i, j]

    return pil_img


def main():

    wandb.init(project="Esh")
    model_relu = build_model(nn.ReLU())
    model_swish = build_model(nn.SiLU())
    model_mish = build_model(nn.Mish())
    model_esh = build_model(Activation('esh').act_func())

    x = np.linspace(0.0, 10.0, num=400)
    y = np.linspace(0.0, 10.0, num=400)

    grid = [torch.tensor([xi, yi], dtype=torch.float) for xi in x for yi in y]

    np_img_relu = np.array([model_relu(point).detach().numpy() for point in grid]).reshape(
        400, 400
    )
    np_img_swish = np.array([model_swish(point).detach().numpy() for point in grid]).reshape(
        400, 400
    )
    np_img_mish = np.array([model_mish(point).detach().numpy() for point in grid]).reshape(
        400, 400
    )
    np_img_esh = np.array([model_esh(point).detach().numpy() for point in grid]).reshape(
        400, 400
    )

    scaler = MinMaxScaler(feature_range=(0, 255))
    np_img_relu = scaler.fit_transform(np_img_relu)
    np_img_swish = scaler.fit_transform(np_img_swish)
    np_img_mish = scaler.fit_transform(np_img_mish)
    np_img_esh = scaler.fit_transform(np_img_esh)

    wandb.log(
        {
            "Landscapes": [
                wandb.Image(np_img_relu, caption="ReLU"),
                wandb.Image(np_img_swish, caption="Swish"),
                wandb.Image(np_img_mish, caption="Mish"),
                wandb.Image(np_img_esh, caption="Esh"),
            ]
        }
    )

    return


if __name__ == "__main__":
    main()
