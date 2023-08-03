import sys
import os
import re
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torchvision import datasets
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as nnf

args = sys.argv

test_transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.RandomHorizontalFlip(),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])       
])

classes = {
  0: 'Annunciazione',
  1: 'Libro d\'Ore miniato',
  2: 'Lastra tombale di Giovanni Cabastida',
  3: 'Madonna del Cardillo',
  4: 'Disputa di San Tommaso',
  5: 'Traslazione della Santa Casa',
  6: 'Madonna col Bambino',
  7: 'L\'immacolata Concezione e Dio Padre in Gloria',
  8: 'Adorazione dei Magi',
  9: 'Sant\'Elena e Costantino e Madonna con Bambino in gloria fra angeli',
  10: 'Taccuini di disegni',
  11: 'Martirio di S. Lucia',
  12: 'Volto di Cristo',
  13: 'Dipinti di Sant\'Orsola',
  14: 'Immacolata e i santi Chiara, Francesco, Antonio, Abate, Barbara e Maria Maddalena',
  15: 'Storia della Genesi'
}

sorted_indexes = {
  0: 0,
  1: 1,
  2: 10,
  3: 11,
  4: 12,
  5: 13,
  6: 14,
  7: 15,
  8: 2,
  9: 3,
  10: 4,
  11: 5,
  12: 6,
  13: 7,
  14: 8,
  15: 9
}

def load_image(img):
    img = test_transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    return img

def get_resnet(num_class=16):
  model = resnet18(weights=ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(512, num_class)
  return model

argv = sys.argv


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_resnet()
model = model.to(device)
model.load_state_dict(torch.load('./code/weights/best_model_params.pt'))
model.eval()



if len(argv) == 2:
  path = argv[1]

  #if path starts with 'data' then it is a path to a file in the dataset
  if os.path.exists(path) and path.startswith('./data'):
      match = re.findall(r'\d+', path)
      y_true = match[-1]
      print("True Class: ", y_true, ' | ', classes[int(y_true)])
  else:
      y_true = "unknown"
  img = Image.open(path)
  img = load_image(img)
  img = img.to(device)
  with torch.no_grad():
    output = model(img)
    probabilities = nnf.softmax(output, dim=1)
    _, predicted = torch.max(probabilities, 1)
    predicted = predicted.item()
    predicted = sorted_indexes[predicted]
    top_p, top_c = probabilities.topk(16, dim = 1)
    top_p = top_p.squeeze().tolist()
    top_c = top_c.squeeze().tolist()
    top_c = [sorted_indexes[c] for c in top_c]
    print("Predicted Class: ", predicted, ' | ', classes[predicted], ' | ', max(top_p))
    #plot the probabilities vertically
    #figsize
    plt.imshow(Image.open(path))
    plt.figure(figsize=(12, 8), tight_layout=True)
    plt.barh(top_c, top_p)
    plt.ylabel('Class')
    plt.yticks(top_c, [classes[c] for c in top_c])
    plt.xlabel('Probability')
    plt.show()

    #show image at path

    

  #print("Predicted Class: ", predicted, ' | ', classes[predicted], ' | ')

  