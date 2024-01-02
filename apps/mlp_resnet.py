import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    sequence = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    residual = nn.Residual(sequence)
    return nn.Sequential(residual, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    '''Notice: this code is not correct. What is the reason?'''
    # residual_blocks = [ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)]
    # return nn.Sequential(
    #     nn.Linear(dim, hidden_dim),
    #     nn.ReLU(),
    #     *residual_blocks,
    #     nn.Linear(hidden_dim, num_classes)
    # )

    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if not opt:
        model.eval()
    else:
        model.train()

    loss_func = nn.SoftmaxLoss()
    err_sum = 0
    loss_sum = 0
    num_batches = 0

    for _, batch in enumerate(dataloader):
        X, y = batch
        out = model(X)
        loss = loss_func(out, y)

        if model.training:
            opt.reset_grad()
            loss.backward()
            opt.step()

        err = np.sum(np.argmax(out.numpy(), axis=1) != y.numpy())
        err_sum += err
        loss_sum += loss.numpy()
        num_batches += 1

    return err_sum / len(dataloader.dataset), loss_sum / num_batches
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(data_dir + '/train-images-idx3-ubyte.gz', data_dir + '/train-labels-idx1-ubyte.gz')
    test_set = ndl.data.MNISTDataset(data_dir + '/t10k-images-idx3-ubyte.gz', data_dir + '/t10k-labels-idx1-ubyte.gz')
    train_dataloader = ndl.data.DataLoader(train_set, batch_size, True)
    test_dataloader = ndl.data.DataLoader(test_set, batch_size, False)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err_rate, train_loss = epoch(train_dataloader, model, opt)
    test_err_rate, test_loss = epoch(test_dataloader, model, None)

    return train_err_rate, train_loss, test_err_rate, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
