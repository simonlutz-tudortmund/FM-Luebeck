import torch
from torch import nn
from torch.autograd import Variable
from tqdm import trange

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_data_iris():
    iris = load_iris()
  
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # print(f"Shape of X (data): {X.shape}")
    # print(f"Shape of y (target): {y.shape} {y.dtype}")
    # print(f"Example of x and y pair: {X[0]} {y[0]}")

    # Scale data to have mean 0 and variance 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    torch.manual_seed(42)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

    # print("Shape of training set X", X_train.shape)
    # print("Shape of test set X", X_test.shape)

    return names, feature_names, X, y, X_scaled, X_train, X_test, y_train, y_test

def predict(X, model):
    model.eval()
    with torch.no_grad():
        X = Variable(torch.from_numpy(X)).float()
        X = X.to(device)
        pred = model(X)
        pred = pred.argmax(1)
        pred = pred.cpu().detach().numpy()
    return pred 

def load_data_mnist():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader = DataLoader(training_data, batch_size=64)
    return train_dataloader
 
def load_data_fashionmnist():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_dataloader = DataLoader(training_data, batch_size=64)
  
    return train_dataloader

def show_plots(names, feature_names, X, y, fixed_input = None, epsilon = None, title = '', fig = None, ax1 = None, ax2 = None):
    if fig == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax1.plot(X_plot[:, 0], X_plot[:, 1], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.axis('equal')
    ax1.legend()

    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax2.plot(X_plot[:, 2], X_plot[:, 3], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])
    ax2.axis('equal')
    ax2.legend()

    if fixed_input is not None and epsilon is not None:
    #add rectangle to plot -> shows infinity norm 
        ax1.add_patch(Rectangle((fixed_input[0] - epsilon, fixed_input[1] - epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='red',
                                facecolor='none',      
                                lw=4))
        ax1.set_aspect("equal", adjustable="datalim")

        ax2.add_patch(Rectangle((fixed_input[2]-epsilon, fixed_input[3]-epsilon), 
                                2*epsilon, 2*epsilon, 
                                edgecolor='red',
                                facecolor='none',      
                                lw=4))
        ax2.set_aspect("equal", adjustable="datalim")

