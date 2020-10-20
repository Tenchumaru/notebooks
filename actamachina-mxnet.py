# https://actamachina.com/notebooks/2019/03/28/captcha.html
# https://mxnet.incubator.apache.org

# pip install captcha
# pip install opencv-python
# pip install Pillow
# pip install mxnet-cu102mkl

from itertools import groupby
from pathlib import Path
import pickle

from mxnet.gluon.data import ArrayDataset, DataLoader
import mxnet.gluon.data.vision.transforms as transforms
import random

import cv2
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxnet.ndarray as nd
import mxnet.gluon as gluon

from captcha.image import ImageCaptcha

import os
pickle_file_path = 'data/captcha10000.pickle'
batch_size = 64
if os.path.isfile(pickle_file_path):
    with open(pickle_file_path, 'rb') as fin:
        train_dataset, test_dataset = pickle.load(fin)
else:
    labels = [f'{chars:>04}' for chars in range(10000)]
    random.shuffle(labels)
    g = (ImageCaptcha().generate_image(l) for l in labels)
    g = ((np.array(i.getdata()).astype(np.float32), *i.size) for i in g)
    g = (cv2.cvtColor(a.reshape((w, h, 3)) / 255, cv2.COLOR_RGB2GRAY).transpose((1, 0)).reshape((w, h, 1)) for a, w, h in g)
    images = np.stack(list(g))
    mean, std = images.mean(), images.std()
    images = transforms.Normalize(mean, std)(nd.array(images))
    labels = [nd.array(list(map(int, l))) for l in labels]
    train_size, test_size = 128 * batch_size, 28 * batch_size
    test_limit = train_size + test_size
    train_dataset = ArrayDataset(images[:train_size], labels[:train_size])
    test_dataset = ArrayDataset(images[train_size:test_limit], labels[train_size:test_limit])
    with open(pickle_file_path, 'wb') as fout:
        pickle.dump([train_dataset, test_dataset], fout)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

class StackedLSTM(gluon.HybridBlock):
    def __init__(self, output_size=11, hidden_size=512, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.name_scope():
            self.dropout = nn.Dropout(rate=0.5)
            self.fc = nn.Dense(output_size)
            self.lstm = gluon.rnn.LSTM(hidden_size, num_layers)

    def hybrid_forward(self, F, inputs): # inputs.shape = (160, 64, 60)
        seq_len, batch_size, input_size = inputs.shape # 160, 64, 60
        outputs = self.lstm(inputs) # outputs.shape = (160, 64, 512)
        outputs = self.dropout(outputs)
        outputs = F.stack(*(self.fc(outputs[i]) for i in range(outputs.shape[0]))) # outputs.shape = (160, 64, 11)
        outputs = F.log_softmax(outputs, axis=2)
        return outputs

    def to(self, ctx):
        self.initialize(mx.init.Xavier(), ctx)
        #
        return self

net = StackedLSTM().to(device)

criterion = gluon.loss.CTCLoss(layout='TNC')
optimizer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

BLANK_LABEL = 10

#net.train()

epochs = 30
batch_size = 64

# for each pass of the training dataset
for epoch in range(epochs):
    train_loss, train_correct, train_total = 0, 0, 0

    #h = net.init_hidden(batch_size)

    # for each batch of training examples
    for batch_index, (inputs, targets) in enumerate(train_dataloader): # inputs.shape = (64, 1, 60, 160), targets.shape = (64, 4)
        inputs, targets = inputs.as_in_context(device), targets.as_in_context(device)
        #h = tuple([each.data for each in h])

        batch_size, channels, height, width = inputs.shape # 64, 1, 60, 160

        # reshape inputs: NxCxHxW -> WxNx(HxC)
        inputs = (inputs
                  .transpose((3, 0, 2, 1))
                  #.contiguous()
                  .reshape((width, batch_size, -1))) # inputs.shape = (160, 64, 60)

        with mx.autograd.record():
            outputs = net(inputs)  # forward pass # outputs.shape = (160, 64, 11)

            # compare output with ground truth
            input_lengths = nd.array([width] * batch_size, device) # input_lengths.shape = (64,)
            target_lengths = nd.array([len(t) for t in targets], device) # target_lengths.shape = (64,)
            loss = criterion(outputs, targets, input_lengths, target_lengths) # loss.shape = (64,)

        loss.backward()  # backpropagation
        #nn.utils.clip_grad_norm_(net.parameters(), 10)  # clip gradients
        optimizer.step(batch_size)  # update network weights

        # record statistics
        max_index = outputs.argmax(axis=2) # max_index.shape = (160, 64)
        train_loss += loss.mean()
        train_total += len(targets)

        for i in range(batch_size):
            raw_pred = list(max_index[:, i].asnumpy()) # len(raw_pred) = 160
            pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL] # len(pred) = 4 when predicting correctly, otherwise in [0, 160]
            target = list(targets[i].asnumpy())
            if pred == target:
                train_correct += 1

        # print statistics every 10 batches
        if (batch_index + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, ' +
                  f'Batch {batch_index + 1}/{len(train_dataloader)}, ' +
                  f'Train Loss: {float(train_loss.asnumpy()):.5f}, ' +
                  f'Train Accuracy: {(train_correct/train_total):.5f}')

            train_loss, train_correct, train_total = 0, 0, 0

#h = net.init_hidden(batch_size)  # init hidden state

#net.eval()

test_loss = 0
test_correct = 0
test_total = len(test_dataloader) * batch_size

#with torch.no_grad():  # detach gradients so network runs faster

# for each batch of testing examples
for batch_index, (inputs, targets) in enumerate(test_dataloader):
    inputs, targets = inputs.as_in_context(device), targets.as_in_context(device)
    #h = tuple([each.data for each in h])
    batch_size, channels, height, width = inputs.shape

    # reshape inputs: NxCxHxW -> WxNx(HxC)
    inputs = (inputs
                .transpose((3, 0, 2, 1))
                #.contiguous()
                .reshape((width, batch_size, -1)))
    with mx.autograd.record(False):
        outputs = net(inputs)  # forward pass

        # record loss
        input_lengths = nd.array([width] * batch_size, device)
        target_lengths = nd.array([len(t) for t in targets], device)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
    test_loss += loss.mean()

    # compare prediction with ground truth
    max_index = outputs.argmax(axis=2)

    for i in range(batch_size):
        raw_pred = list(max_index[:, i].asnumpy())
        pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
        target = list(targets[i].asnumpy())
        if pred == target:
            test_correct += 1

print(f'Test Loss: {float(test_loss.asnumpy()/len(test_dataloader)):.5f}, ' +
      f'Test Accuracy: {(test_correct/test_total):.5f} ' +
      f'({test_correct}/{test_total})')

data_iterator = iter(test_dataloader)
inputs, targets = next(data_iterator)

i = 1

image = inputs[i,0,:,:]

print(f"Target: {''.join(map(str, targets[i].asnumpy()))}")
window_title = 'Acta Machina'
cv2.imshow(window_title, image.asnumpy())
cv2.waitKey()

inputs = inputs.as_in_context(device)

batch_size, channels, height, width = inputs.shape
#h = net.init_hidden(batch_size)

inputs = (inputs
          .transpose((3, 0, 2, 1))
          #.contiguous()
          .reshape((width, batch_size, -1)))

# get prediction
outputs = net(inputs)  # forward pass
max_index = outputs.argmax(axis=2)
raw_pred = list(max_index[:, i].asnumpy())

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

cv2.imshow(window_title, annotated_image.asnumpy())
cv2.waitKey()
cv2.destroyAllWindows()
