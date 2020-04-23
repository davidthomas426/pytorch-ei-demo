import time

import torch, torchvision
import PIL
from torchvision import transforms
from PIL import Image


def get_image(filename):
  im = Image.open(filename)
  # ImageNet pretrained models required input images to have width/height of 224
  # and color channels normalized according to ImageNet distribution.
  im_process = transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
  im = im_process(im) # 3 x 224 x 224
  return im.unsqueeze(0) # Add dimension to become 1 x 3 x 224 x 224


image = get_image('kitten.jpg')

# Deserialize model
model = torch.jit.load('wslresnext101_32x8d_traced.pt', map_location=torch.device('cpu')).eval()

# Use it on the image
with torch.no_grad():
  with torch.jit.optimized_execution(True):
    output = model(image)

# Torchvision implementation doesn't have Softmax as last layer.
# Use Softmax to convert activations to normalized probabilities.
probs = torch.nn.Softmax(dim=1)(output)

# Get top 5 predicted classes
classes = eval(open('imagenet_classes.txt').read())
pred_probs, pred_indices = torch.topk(probs, 5)
pred_probs = pred_probs.squeeze().detach().numpy()
pred_indices = pred_indices.squeeze().detach().numpy()

for i in range(len(pred_indices)):
  curr_class = classes[pred_indices[i]]
  curr_prob = pred_probs[i]
  print('{}: {:.4f}'.format(curr_class, curr_prob))

print()

# Now let's check the performance.
# Perform inference. Make sure to disable autograd and use EI execution context
for i in range(30):
  with torch.no_grad():
    with torch.jit.optimized_execution(True):
      start = time.time()
      output = model(image)
      end = time.time()
      latency = (end - start) * 1000
      print('Latency in ms: {}'.format(latency))

