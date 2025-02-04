from PIL import Image
from torchvision import models

wg = models.ResNet50_Weights.DEFAULT
cats = wg.meta['categories']
transforms = wg.transforms()

model = models.resnet50(weights=wg)

img = Image.open('img/auto_1.jpg').convert('RGB')
img = transforms(img).unsqueeze(0)

model.eval()

p = model(img).squeeze()
res= p.softmax(dim=0).sort(descending=True)

for s,i in zip(res[0][:5],res[1][:5]):
    print(f'{cats[i]}: {s:.4f}')