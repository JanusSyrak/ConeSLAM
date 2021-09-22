from models.deeplab import *
import torch
from torchvision import transforms
import torchvision
import os
import time
from PIL import Image    
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np

out_channels = 5
model = custom_DeepLabv3(out_channels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device=device)
model.load_state_dict(torch.load("models/deeplab.pth", map_location=device))

model.eval()

img_path = "C:/Users/stebb/OneDrive/Desktop/skoli/10sem/fs-folder/cone-slam/pipeline/data/imgs_airport"
mask_path = "C:/Users/stebb/OneDrive/Desktop/skoli/10sem/fs-folder/cone-slam/pipeline/data/deeplab_masks2"

start = time.time()

scale = 0.5
index = 0

for i, img in enumerate(os.listdir(img_path)):
    index = i
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

    preprocess_image = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess_image(img_tensor)

    input_batch = img_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    print(img)
    r.save(os.path.join(mask_path, img ))

end = time.time()
total = end - start
print("TOTAL TIME: ")
print(total/index)