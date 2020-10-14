import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending.

    Arguments:
        path: path of directory containing image files 
              that can be assumed to have all the same dimensions
        file_ending: string specifying what image files have to end with,
                     if not->ignore file

    Return:
        images (list): list of images (each image as numpy.ndarray and dtype=float64)
        dimension_x (int): size of images in x direction
        dimension_y (int): size of images in y direction
    """

    images = []

    # read each image in path as numpy.ndarray and append to images
    files = os.listdir(path)
    files.sort()
    for cur in files:
        if not cur.endswith(file_ending):
            continue

        try:
            image = mpl.image.imread(path + cur)
            img_mtx = np.asarray(image, dtype="float64")
            images.append(img_mtx)
        except:
            continue

    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # encoder
        self.fc_enc = nn.Linear(x*y, k)

        # latent view
        self.fc_lv = nn.Linear(k, k)

        # decoder
        self.fc_dec = nn.Linear(k, x*y)

    def forward(self, x):
        
        x = F.tanh(self.fc_enc(x))
        x = F.tanh(self.fc_lv(x))
        x = self.fc_dec(x)
        return x


if __name__ == '__main__':

    # 10, 75, 150
    k = 75
    images, x, y = load_images('./data/train/')

    # setup data matrix
    D = np.zeros((len(images), images[0].size), dtype=np.float32) # n*p
    for i in range(len(images)):
        D[i, :] = images[i].flatten()

    # 1. calculate and subtract mean to center the data in D
    means = D.mean(axis=0).reshape(1, -1) # 1*p
    D -= means # n*p

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(D).to(device)
    num_epochs = 2000
    batch_size = 50
    learning_rate = 0.01

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-05)

    for epoch in range(num_epochs):
        data = Variable(data)
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        MSE_loss = nn.MSELoss()(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data, MSE_loss.data))

    # now we use the nn model to reconstruct test images
    # and measure their reconstruction error

    images_test, x, y = load_images('./data/test/')
    D_test = np.zeros((len(images_test), images_test[0].size), dtype=np.float32)
    for i in range(len(images_test)):
        D_test[i, :] = images_test[i].flatten()

    D_test -= means
    data_test = torch.from_numpy(D_test).to(device)

    errors = []
    for i, test_image in enumerate(images_test):

        # evaluate the model using data_test samples i
        pred = model(data_test[i].view(1, -1)) # 1*p

        # add the mean to the predicted/reconstructed image
        pred_np = pred.cpu().data.numpy()
        pred_np += means

        # and reshape to size (116,98)
        img_reconst = pred_np.reshape(images_test[0].shape)

        error = np.linalg.norm(images_test[i] - img_reconst)
        errors.append(error)
        print("reconstruction error: ", error)

    grid = plt.GridSpec(2, 9)

    plt.subplot(grid[0, 0:3])
    plt.imshow(images_test[-1], cmap='Greys_r')
    plt.xlabel('Original person')

    pred = model(data_test[-1, :]).view(1, -1) # 1*p
    pred_np = pred.cpu().data.numpy()
    pred_np += means
    img_reconst = pred_np.reshape(images_test[-1].shape)
    plt.subplot(grid[0, 3:6])
    plt.imshow(img_reconst, cmap='Greys_r')
    plt.xlabel('Reconstructed image')

    plt.subplot(grid[0, 6:])
    plt.plot(np.arange(len(images_test)), errors)
    plt.xlabel('Errors all images')

    plt.savefig(f"nn_solution-{k}.png")
    plt.show()

    print("Mean error", np.asarray(errors).mean())

# inspired by
# https://www.kaggle.com/jagadeeshkotra/autoencoders-with-pytorch