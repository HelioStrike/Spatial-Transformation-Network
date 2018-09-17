import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import os
from model import *

def main():
    #dirs
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./saved_images", exist_ok=True)

    #Hyperparamas
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 10
    NUM_WORKERS = 4
    SAVE_EVERY = 1
    LOAD_MODEL = True
    data_path = "../data/mnist/"

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Transforms
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #Train-Loader
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=data_path, train=True, download=False, transform=tfms), \
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    #Test-Loader
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=data_path, train=False, download=False, transform=tfms), \
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    #Network instance
    net = Net().to(device)
    if LOAD_MODEL:
        if(os.path.exists("./saved_models/models.pth")):
            net.load_state_dict(torch.load("./saved_models/models.pth"))
            net = net.eval()

    #Criterion and optimizer
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    #Train
    def train(epoch):
        net.train()
        tot_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outs = net(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            if i%500 == True:
                print("Training, Epoch:", epoch, "Iteration:", i, "Loss:", loss.item())

        avg_loss = tot_loss/len(train_loader)
        print("Training, Epoch:", epoch, "Loss:", avg_loss)

        if epoch%SAVE_EVERY == 0:
            torch.save(net.state_dict(), f"./saved_models/model.pth")

    #Test
    def test(epoch):
        net.eval()
        tot_loss = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outs = net(images)
            loss = criterion(outs, labels)
            tot_loss += loss.item()

        avg_loss = tot_loss/len(test_loader)
        print("Validation, Epoch:", epoch, "Loss:", avg_loss)

        save_images = next(iter(test_loader))[0][:5].to(device)
        save_outs = net.stn(save_images)

        save_image(torch.cat((save_images, save_outs), dim=0).detach(), os.path.join("./saved_images/", f"{epoch}.jpg"), nrow=5, normalize=True)

    #For each epoch, train, then test
    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test(epoch)

if __name__=='__main__':
    main()
