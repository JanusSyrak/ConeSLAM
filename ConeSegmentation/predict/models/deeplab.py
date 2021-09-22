from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101

def custom_DeepLabv3(out_channel):
  model = deeplabv3_resnet101(pretrained=True)
  model.classifier = DeepLabHead(2048, out_channel)
  return model