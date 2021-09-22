# This script predicts masks and can use three different unets as inputs, found in the sub-folder modules

from models.UNets import *
import torch
from torchvision import transforms
import os
from PIL import Image    
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import time


model = UNet_5layers(n_channels=3, n_classes=5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device=device)
model.load_state_dict(torch.load("models/unet_5layers.pth", map_location=device))

model.eval()

img_path = "C:/Users/stebb/OneDrive/Desktop/skoli/10sem/fs-folder/cone-slam/pipeline/data/imgs_airport"
mask_path = "C:/Users/stebb/OneDrive/Desktop/skoli/10sem/fs-folder/cone-slam/pipeline/data/masks_airport_unet"

start = time.time()

scale = 0.5
index = 0
for i, img in enumerate(os.listdir(img_path)):
    index = i + 1
    pil_img = Image.open(img_path + "/" + img)
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))

    img_tensor = torch.from_numpy(img_trans).type(torch.FloatTensor)

    preprocess_new = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess_new(img_tensor)

    input_batch = img_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch).squeeze()
    output_predictions = output.argmax(0)
    np_arr = output_predictions.cpu().numpy().astype(np.uint8)
    print(img)
    r = Image.fromarray(np_arr)
    w, h = r.size
    newW, newH = int(1/scale * w), int(1/scale * h)
    r = r.resize((newW, newH))
    r.save(os.path.join(mask_path, img ))

end = time.time()
total = end - start
print("TOTAL TIME: ")
print(total/index)
