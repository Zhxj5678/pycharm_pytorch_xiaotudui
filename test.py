import torch
import torchvision
from PIL import Image

image_path = "./imgs/dog1.jpg"

image = Image.open(image_path)
image = image.convert("RGB")
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)

model = torch.load("./tudui_5.pth")
#print(model)
image = torch.reshape(image, (1, 3, 32, 32))
# ps：在GPU上训练的模型需要image = image.cuda()

model.eval()
with torch.no_grad():# 可以节约一些内存和性能
    output = model(image)


output = model(image)
print(output)