# Attention: The code is copied from https://github.com/Kenneth111/TransformerDemo, which is a very concise implementation by Pytorch. 
# The original author has the copyright to the code!!! Thanks a lot to the original author！

import argparse
from numpy import arange, random
from torch import save, load, no_grad, LongTensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from number_loader import NumberLoader
from model import SkylarkTransformer
from torch import LongTensor
from torch.utils.data import Dataset


class NumberLoader(Dataset):
    def __init__(self, x, y, inp_len=3, out_len=3):
        if len(x) != len(y):
            raise ValueError("len(x) != len(y)")
        self.x = [[x[i + j] for j in range(inp_len)] for i in range(len(x) - inp_len + 1)]
        self.y = [[y[i + j] for j in range(out_len)] for i in range(len(y) - out_len + 1)]

    def __getitem__(self, index):
        return LongTensor(self.x[index]), LongTensor([0] + self.y[index])

    def __len__(self):
        return len(self.x)

def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        src, tgt = batch
        src, tgt = src.transpose(1, 0).cuda(), tgt.transpose(1, 0).cuda()
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])
        n = output.shape[-1]
        loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validation(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    with no_grad():
        for i, batch in enumerate(loader):
            src, tgt = batch
            src, tgt = src.transpose(1, 0).cuda(), tgt.transpose(1, 0).cuda()
            output = model(src, tgt[:-1, :])
            n = output.shape[-1]
            loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test(model, max_len=3, test_times=1):
    model = model.cuda()
    model.eval()
    with no_grad():
        for i in range(test_times):
            s = random.randint(1, 4998)
            cpu_src = [(s + j) * 2 for j in range(max_len)]
            src = LongTensor(cpu_src).unsqueeze(1).cuda()
            tgt = [0] + [(s + j) * 2 + 1 for j in range(max_len)]
            pred = [0]
            for j in range(max_len):
                inp = LongTensor(pred).unsqueeze(1).cuda()
                output = model(src, inp)
                out_num = output.argmax(2)[-1].item()
                pred.append(out_num)
            print("input: ", cpu_src)
            print("target: ", tgt)
            print("predict: ", pred)


def main(model_name=None, hidden=64, nlayers=1):
    voc_size = 10000
    inp = arange(2, voc_size, 2)
    tgt = arange(3, voc_size, 2)
    batch_size = 128
    epochs = 30
    dataset = NumberLoader(inp, tgt)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    model = SkylarkTransformer(voc_size, voc_size, hidden=hidden, nlayers=nlayers)
    if model_name is not None:
        model.load_state_dict(load(model_name))
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_loss = 100
    for i in range(epochs):
        epoch_loss = train(model, criterion, optimizer, train_loader)
        epoch_loss_val = validation(model, criterion, val_loader)
        # scheduler.step()
        print("epoch: {} train loss: {}".format(i, epoch_loss))
        print("epoch: {} val loss: {}".format(i, epoch_loss_val))
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            model_name = "model/model_{0:.5f}.pt".format(epoch_loss_val)
            save(model.state_dict(), model_name)
    return model_name


if __name__ == "__main__":
    # A demo to predict odd numbers. Given the input [2, 4, 6], this program generates the output [3, 5, 7]. Given the input [100, 102, 104], this program generates the output [101, 103, 105].

    parser = argparse.ArgumentParser(description='A PyTorch Transformer Language Model for Predicting Odd Numbers')
    parser.add_argument('--test_model', type=str, help='the model file to load')
    parser.add_argument('--train_model', type=str, help='the model file to load')
    args = parser.parse_args()
    hidden = 128
    nlayers = 2
    if args.test_model is None:
        if args.train_model is not None:
            model_name = main(args.train_model, hidden=hidden, nlayers=nlayers)
        else:
            model_name = main(hidden=hidden, nlayers=nlayers)
    else:
        model_name = args.test_model
    model = SkylarkTransformer(10000, 10000, hidden=hidden, nlayers=nlayers)
    model.load_state_dict(load(model_name))
    test(model, test_times=10)