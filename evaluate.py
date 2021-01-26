import os

from torchvision.transforms import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
model = InceptionV3(weights='imagenet')

img_path = r'C:\Users\USER\Desktop\wo_Ltv.png'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=5)[0])


from pretrained_models_pytorch import pretrainedmodels
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from utils import *

netClassifier = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
netClassifier.eval()
image = Image.open(img_path)
normalize = transforms.Normalize(mean=netClassifier.mean,
                                 std=netClassifier.std)
transform = transforms.Compose([
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space == 'BGR'),
        ToRange255(max(netClassifier.input_range) == 255),
        normalize,
    ])
x = transform(image)
x.unsqueeze_(0)
output = netClassifier(x)
prob = F.softmax(output)
_, top5 = output.topk(5)
print(top5)
print(F.softmax(output).data[0][top5])
