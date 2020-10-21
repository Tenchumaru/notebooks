# https://actamachina.com/notebooks/2019/03/28/captcha.html
# https://pytorch.org/get-started/locally/

# pip install captcha
# pip install opencv-python
# pip install Pillow
# pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

from itertools import groupby
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from captcha.image import ImageCaptcha

image = ImageCaptcha()

import os
root_dir = r"C:\Users\cidzerda\Documents\GitHub\notebooks\data\captcha10000"
for chars in range(0, 10000):
    file_path = os.path.join(root_dir, f'{chars:>04}.png')
    if not os.path.isfile(file_path):
        image.write(f'{chars:>04}', file_path)

class CaptchaDataset(Dataset):
    """CAPTCHA dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = list(Path(root_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])

        if self.transform:
            image = self.transform(image)

        label_sequence = [int(c) for c in self.image_paths[index].stem]
        return (image, torch.tensor(label_sequence))

    def __len__(self):
        return len(self.image_paths)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

dataset = CaptchaDataset(root_dir=root_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=10000)

for batch_index, (inputs, _) in enumerate(dataloader):
    print(f'Mean: {inputs.mean()}, Variance: {inputs.std()}')
    input_mean, input_std = inputs.mean(), inputs.std()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((input_mean,), (input_std,)),
])

dataset = CaptchaDataset(root_dir=root_dir, transform=transform)

train_dataset, test_dataset, _ = random_split(dataset, [128*64, 28*64, 10000 - (128*64 + 28*64)])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StackedLSTM(nn.Module):
    def __init__(self, input_size=60, output_size=11, hidden_size=512, num_layers=2):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
    def forward(self, inputs, hidden): # inputs.shape = (160, 64, 60), len(hidden) = 2, hidden[i].shape = (2, 64, 512)
        seq_len, batch_size, input_size = inputs.shape # 160, 64, 60
        outputs, hidden = self.lstm(inputs, hidden) # outputs.shape = (160, 64, 512), len(hidden) = 2, hidden[i].shape = (2, 64, 512)
        outputs = self.dropout(outputs)
        outputs = torch.stack([self.fc(outputs[i]) for i in range(width)]) # width = 160, outputs.shape = (160, 64, 11)
        outputs = F.log_softmax(outputs, dim=2)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data 
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
    
net = StackedLSTM().to(device)

criterion = nn.CTCLoss(blank=10)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

BLANK_LABEL = 10

net.train()  # set network to training phase

epochs = 30
batch_size = 64

# for each pass of the training dataset
for epoch in range(epochs):
    train_loss, train_correct, train_total = 0, 0, 0

    h = net.init_hidden(batch_size) # len(h) = 2, h[i].shape = (2, 64, 512), h[i] = 0

    # for each batch of training examples
    for batch_index, (inputs, targets) in enumerate(train_dataloader): # inputs.shape = (64, 1, 60, 160), targets.shape = (64, 4)
        inputs, targets = inputs.to(device), targets.to(device)
        h = tuple([each.data for each in h])

        batch_size, channels, height, width = inputs.shape # 64, 1, 60, 160

        # reshape inputs: NxCxHxW -> WxNx(HxC)
        inputs = (inputs
                  .permute(3, 0, 2, 1)
                  .contiguous()
                  .view((width, batch_size, -1))) # inputs.shape = (160, 64, 60)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs, h = net(inputs, h)  # forward pass # outputs.shape = (160, 64, 11)

        # compare output with ground truth
        input_lengths = torch.IntTensor(batch_size).fill_(width) # input_lengths.shape = (64,)
        target_lengths = torch.IntTensor([len(t) for t in targets]) # target_lengths.shape = (64,)
        loss = criterion(outputs, targets.to(device), input_lengths, target_lengths) # loss.shape = ()

        loss.backward()  # backpropagation
        nn.utils.clip_grad_norm_(net.parameters(), 10)  # clip gradients
        optimizer.step()  # update network weights

        # record statistics
        prob, max_index = torch.max(outputs, dim=2) # prob.shape = (160, 64), max_index.shape = (160, 64)
        train_loss += loss.item()
        train_total += len(targets)

        for i in range(batch_size):
            raw_pred = list(max_index[:, i].cpu().numpy()) # len(raw_pred) = 160
            pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL] # len(pred) = 4 when predicting correctly, otherwise in [0, 160]
            target = list(targets[i].cpu().numpy())
            if pred == target:
                train_correct += 1

        # print statistics every 10 batches
        if (batch_index + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, ' +
                  f'Batch {batch_index + 1}/{len(train_dataloader)}, ' +
                  f'Train Loss: {(train_loss/1):.5f}, ' +
                  f'Train Accuracy: {(train_correct/train_total):.5f}')

            train_loss, train_correct, train_total = 0, 0, 0

h = net.init_hidden(batch_size)  # init hidden state

net.eval()

test_loss = 0
test_correct = 0
test_total = len(test_dataloader.dataset)

with torch.no_grad():  # detach gradients so network runs faster

    # for each batch of testing examples
    for batch_index, (inputs, targets) in enumerate(test_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        h = tuple([each.data for each in h])
        batch_size, channels, height, width = inputs.shape

        # reshape inputs: NxCxHxW -> WxNx(HxC)
        inputs = (inputs
                  .permute(3, 0, 2, 1)
                  .contiguous()
                  .view((width, batch_size, -1)))

        outputs, h = net(inputs, h)  # forward pass

        # record loss
        input_lengths = torch.IntTensor(batch_size).fill_(width)
        target_lengths = torch.IntTensor([len(t) for t in targets])
        loss = criterion(outputs, targets.to(device), input_lengths, target_lengths)
        test_loss += loss.item()

        # compare prediction with ground truth
        prob, max_index = torch.max(outputs, dim=2)

        for i in range(batch_size):
            raw_pred = list(max_index[:, i].cpu().numpy())
            pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
            target = list(targets[i].cpu().numpy())
            if pred == target:
                test_correct += 1

print(f'Test Loss: {(test_loss/len(test_dataloader)):.5f}, ' +
      f'Test Accuracy: {(test_correct/test_total):.5f} ' +
      f'({test_correct}/{test_total})')

data_iterator = iter(test_dataloader)
inputs, targets = data_iterator.next()

i = 1

image = inputs[i,0,:,:]

print(f"Target: {''.join(map(str, targets[i].numpy()))}")
window_title = 'Acta Machina'
cv2.imshow(window_title, image.numpy())
cv2.waitKey()

inputs = inputs.to(device)

batch_size, channels, height, width = inputs.shape
h = net.init_hidden(batch_size)

inputs = (inputs
          .permute(3, 0, 2, 1)
          .contiguous()
          .view((width, batch_size, -1)))

# get prediction
outputs, h = net(inputs, h)  # forward pass
prob, max_index = torch.max(outputs, dim=2)
raw_pred = list(max_index[:, i].cpu().numpy())

# print raw prediction with BLANK_LABEL replaced with "-"
print('Raw Prediction: ' + ''.join([str(c) if c != BLANK_LABEL else '-' for c in raw_pred]))

pred = [str(c) for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
print(f"Prediction: {''.join(pred)}")

line_mask = [(a == BLANK_LABEL) & (b != BLANK_LABEL) for a, b in zip(raw_pred, raw_pred[1:])]
indices = [i for i, x in enumerate(line_mask) if x]

annotated_image = image.clone()
for index in indices:
    annotated_image[:, index] = 0
    annotated_image[:, index+1] = -2

cv2.imshow(window_title, annotated_image.numpy())
cv2.waitKey()
cv2.destroyAllWindows()
