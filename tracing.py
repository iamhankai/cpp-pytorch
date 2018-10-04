import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# evalute time
batch = torch.rand(256, 3, 224, 224)
start = time()
output = traced_script_module(batch)
stop = time()
print(str(stop-start) + "s")

# read image
image = Image.open('dog.png')
default_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
      ])
image = default_transform(image)

# forward
output = traced_script_module(image.unsqueeze(0))
print(output[0, :5])

traced_script_module.save("model.pt")

