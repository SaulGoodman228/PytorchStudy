from torchvision import models
from PIL import Image
import torchvision.transforms.v2 as tfs_v2

vgg_weights = models.VGG16_Weights.DEFAULT

cats = vgg_weights.meta['categories']
transform_1 = vgg_weights.transforms()

img = Image.open('img/auto_1.jpg').convert('RGB')
img_net = transform_1(img).unsqueeze(0)

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

model.eval()
p = model(img_net).squeeze()
res = p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f'{cats[i]}: {s:.4f}')