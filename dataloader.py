from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_data_iris():
    iris = load_iris()
  
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    print(f"Shape of X (data): {X.shape}")
    print(f"Shape of y (target): {y.shape} {y.dtype}")
    print(f"Example of x and y pair: {X[0]} {y[0]}")

    # Scale data to have mean 0 and variance 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    torch.manual_seed(42)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

    print("Shape of training set X", X_train.shape)
    print("Shape of test set X", X_test.shape)

    return names, feature_names, X, y, X_scaled, X_train, X_test, y_train, y_test

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

