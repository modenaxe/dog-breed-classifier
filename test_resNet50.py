import torch
import scipy
from PIL import Image
from torchvision import models, transforms 

Res50 = models.inception_v3(pretrained=True)

img_path = "./test_images/beethoven.jpg"

preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open(img_path)
in_tensor = preprocess(img)
in_tensor = in_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_in_tensor = in_tensor.to('cuda')
    Res50.to('cuda')

with torch.no_grad():
    output = Res50(in_tensor)

#print(torch.nn.functional.softmax(output[0], dim=0))

a, b = torch.topk(torch.nn.functional.softmax(output[0], dim=0), 1)
print(a)
print(b)