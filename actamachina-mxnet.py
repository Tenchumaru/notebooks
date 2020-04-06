# https://actamachina.com/notebooks/2019/03/28/captcha.html

# pip install captcha
# pip install opencv-python
# pip install Pillow
# pip install mxnet-cu101mkl

import cv2
import itertools as it
import multiprocessing
import numpy as np
import os
import sys
from pathlib import Path
from utilities import partition

# Load the data.
def fn():
    root_dir = r"C:\Users\cidzerda\Documents\GitHub\notebooks\data\captcha10000"
    for p in Path(root_dir).glob('*'):
        image = cv2.imread(str(p))
        yield image, p.stem
data = list(fn())

def transform_(image, label):
    # Take the color channel with the lowest value.  This changes the shape of
    # the image from (60, 160, 3) to (60, 160).
    g = (image[:, :, i] for i in range(image.shape[2]))
    image = min(g, key=lambda image: image.sum())

    # Convert 8-bit [0, 255] to floating point [0, 1].
    image = image.astype(np.float32) / 255

    return image, label

# Determine the mean and the standard deviation.
def fn(data):
    g = it.starmap(transform_, data)
    data = np.vstack([image for image, _ in g])
    return data.mean(), data.std()
mean, std = fn(data)
print(f'Mean: {mean}, Variance: {std}')

def transform(image, label):
    image, label = transform_(image, label)
    image = (image - mean) / std
    label = list(map(int, label))
    label = np.array(label, dtype=np.float32)
    return image, label

BLANK_LABEL = 10
def decode(predictions):
    for prediction in predictions:
        yield ''.join(str(int(a)) for a, _ in it.groupby(prediction) if a != BLANK_LABEL) # len(rv) = 4 when predicting correctly, otherwise in [0, 160]

import mxnet as mx
class StackedLSTM(mx.gluon.HybridBlock):
    def __init__(self, ctx, *, hidden_size=512, num_layers=2, output_size=11, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.name_scope():
            self.lstm = mx.gluon.rnn.LSTM(hidden_size, num_layers)
            self.dropout = mx.gluon.nn.Dropout(rate=0.5)
            self.fc = mx.gluon.nn.Dense(output_size) # TODO does this need flatten=False?
        self.initialize(mx.init.Xavier(), ctx)

    def hybrid_forward(self, F, inputs):
        outputs = self.lstm(inputs)
        outputs = self.dropout(outputs)
        outputs = mx.ndarray.stack(*(self.fc(outputs[i]) for i in range(outputs.shape[0])))
        #outputs = mx.ndarray.log_softmax(outputs, axis=2) TODO according to CTCLoss documentation, it expects an "unnormalized prediction tensor".
        return outputs

ctc_loss = mx.gluon.loss.CTCLoss(weight=0.2)
reporting_period = 5
def run_epoch(ctx, epoch, network, dataloader, trainer, print_name):
    total_loss = mx.nd.zeros(1, ctx)
    for i, (x, y) in enumerate(dataloader):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        batch_size, height, width = x.shape
        with mx.autograd.record(train_mode=trainer is not None):
            x = x.transpose((2, 0, 1)) # NxHxW -> WxNxH
            output = network(x)
            x = x.transpose((1, 0, 2)) # WxNxP -> NxWxP
            loss_ctc = ctc_loss(output, y) # TODO verify input and output layouts according to documentation.
        if trainer is not None:
            loss_ctc.backward()
            trainer.step(batch_size)
        total_loss += loss_ctc.mean()
    if epoch % reporting_period == 0:
        predictions = output.softmax().topk(axis=2).asnumpy()
        decoded_text = next(decode(predictions))
        print(f'first {print_name} prediction "{decoded_text}"')
        decoded_text = next(decode(y.asnumpy()))
        print(f'{" " * (len(print_name) + 12)}truth "{decoded_text}"')
    epoch_loss = float(total_loss.asscalar()) / len(dataloader)
    return epoch_loss

def main():
    checkpoint_dir = 'data'
    checkpoint_name = f'{os.path.splitext(os.path.basename(sys.argv[0]))[0]}.params'
    training_batch_count = 128
    validation_batch_count = 28
    batch_size = 64
    learning_rate = 0.01

    training_data, validation_data = partition(data, training_batch_count * batch_size)
    validation_data, testing_data = partition(validation_data, validation_batch_count * batch_size)
    training_dataset = zip(*it.starmap(transform, training_data))
    training_dataset = mx.gluon.data.dataset.ArrayDataset(*training_dataset)
    validation_dataset = zip(*it.starmap(transform, validation_data))
    validation_dataset = mx.gluon.data.dataset.ArrayDataset(*validation_dataset)

    training_dataloader = mx.gluon.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = mx.gluon.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

    # Training

    net = StackedLSTM(ctx)
    if len(sys.argv) == 1:
        net.hybridize()
    best_validation_loss = 10e5
    if (os.path.isfile(os.path.join(checkpoint_dir, checkpoint_name))):
        net.load_parameters(os.path.join(checkpoint_dir, checkpoint_name))
        print('Parameters loaded')
        best_validation_loss = run_epoch(ctx, 0, net, validation_dataloader, None, print_name='pretrained')
        print('Initial loss: ', best_validation_loss)
    # TODO trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'momentum': 0.9})
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
    epoch = last_updated_epoch = 0
    while best_validation_loss >= 0.1 and epoch - last_updated_epoch < 50:
        epoch += 1
        training_loss = run_epoch(ctx, epoch, net, training_dataloader, trainer, print_name='training')
        validation_loss = run_epoch(ctx, epoch, net, validation_dataloader, None, print_name='validation')    
        if validation_loss < best_validation_loss:
            print('Saving network, previous best validation loss {:.6f}, current validation loss {:.6f}'.format(best_validation_loss, validation_loss))
            net.save_parameters(os.path.join(checkpoint_dir, checkpoint_name))
            best_validation_loss = validation_loss
            last_updated_epoch = epoch
        if epoch % reporting_period == 0:
            print('Epoch {0}, training loss {1:.6f}, validation loss {2:.6f}'.format(epoch, training_loss, validation_loss))

    # Results

    # Visually inspect the results of the test dataset.
    l = []
    for image, actual_label in testing_data:
        image, _ = transform(image, actual_label)
        image = mx.nd.array(image)
        image = image.as_in_context(ctx)
        image = image.expand_dims(axis=0)
        output = net(image)
        predictions = output.softmax().topk(axis=2).asnumpy()
        decoded_prediction_text = decode(predictions)[0]
        print('[Label]: {}\n[Pred]:  {}'.format(actual_label, decoded_prediction_text))
        l.append(image)
    image = np.vstack([image.asnumpy().squeeze() for image in l])
    cv2.imshow(make_title_for_window(), image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
