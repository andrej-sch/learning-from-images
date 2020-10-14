import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nn_img_size = 32
num_classes = 3
learning_rate = 0.0001
num_epochs = 500
batch_size = 4

loss_mode = 'mse'
#loss_mode = 'crossentropy'

loss_train_hist = []

##################################################
## Please implement a two layer neural network  ##
##################################################

def relu(x):
    """ReLU activation function."""
    return torch.clamp(x, min=0.0)

def relu_derivative(output):
    """Derivative of the ReLU activation function."""
    output[output <= 0] = 0
    output[output>0] = 1
    return output

def softmax(z):
    """Softmax function to transform values to probabilities."""
    z -= z.max()
    z = torch.exp(z)
    sum_z = z.sum(1, keepdim=True)
    return z / sum_z 

def loss_mse(activation, y_batch):
    """Mean squared loss function."""
    # use MSE error as loss function 
    # Hint: the computed error needs to get normalized over the number of samples
    loss = (activation - y_batch).pow(2).sum() 
    mse = 1.0 / activation.shape[0] * loss
    return mse

def loss_deriv_mse(activation, y_batch):
    """Derivative of the mean squared loss function."""
    dCda2 = (2 / activation.shape[0]) * (activation - y_batch)
    return dCda2

def loss_crossentropy(activation, y_batch):
    """Cross entropy loss function."""
    batch_size = y_batch.shape[0]
    loss = (-y_batch * activation.log()).sum() / batch_size
    return loss

def loss_deriv_crossentropy(activation, y_batch):
    """Derivative of the mean cross entropy loss function."""
    batch_size = y_batch.shape[0]
    dCda2 = activation
    dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
    dCda2 /= batch_size
    return dCda2

def forward(X_batch, y_batch, W1, W2, b1, b2):
    """forward pass in the neural network """
    # Implement the forward pass
    m1 = torch.mm(X_batch, W1) + b1
    a1 = relu(m1)

    m2 = torch.mm(a1, W2) + b2
    #a2 = relu(m2)
    a2 = m2

    if loss_mode == "mse":
        loss = loss_mse(a2, y_batch)
    elif loss_mode == "crossentropy":
        a2 = softmax(a2)
        loss = loss_crossentropy(a2, y_batch)
    else:
        exit() # error

    # the function should return the loss 
    # and both intermediate activations
    return loss, a2, a1

def backward(a2, a1, X_batch, y_batch, W2):
    """backward pass in the neural network """
    # Implement the backward pass by computing
    # the derivative of the complete function
    # using the chain rule as discussed in the lecture

    # m - number of samples, h - number of hidden neurons
    # n - input dimension, o - output dimension 
    #dCda2 = 2*(a2-y_batch) # m*o
    if loss_mode == "mse":
        dCda2 = loss_deriv_mse(a2, y_batch) # m*o
    elif loss_mode == "crossentropy":
        dCda2 = loss_deriv_crossentropy(a2, y_batch) # m*o
    else:
        exit() # error
    #da2dm2 = relu_derivative(a2) # m*o
    da2dm2 = 1
    da1dm1 = relu_derivative(a1) # m*h
    dm2da1 = W2 # h*o
    dm1dW1 = X_batch # m*n
    dm2dW2 = a1 # m*h
    
    tmp1 = dCda2 * da2dm2 # m*o
    tmp2 = torch.mm(tmp1, dm2da1.t()) # m*h
    tmp3 = tmp2 * da1dm1 # m*h

    dCdW1 = torch.mm(dm1dW1.t(), tmp3) # n*h
    dCdW2 = torch.mm(dm2dW2.t(), tmp1) # h*o
    dCdb1 = tmp3.sum(dim=0, keepdim=True) # 1*h
    dCdb2 = tmp1.sum(dim=0, keepdim=True) # 1*o
    #dCdb1 = torch.matmul(dm1db1, tmp3) # 1*h
    #dCdb2 = torch.matmul(dm2db2, tmp1) # 1*o

    # function should return 4 derivatives 
    # with respect to W1, W2, b1, b2
    return dCdW1, dCdW2, dCdb1, dCdb2

def setup_train():
    """Train function."""
    # load and resize train images in three categories
    # cars = 0, flowers = 1, faces = 2 (true_ids)
    train_images_cars = glob.glob('./images/db/train/cars/*.jpg')
    train_images_flowers = glob.glob('./images/db/train/flowers/*.jpg')
    train_images_faces = glob.glob('./images/db/train/faces/*.jpg')
    train_images = [train_images_cars, train_images_flowers, train_images_faces]
    num_rows = len(train_images_cars)+len(train_images_flowers)+len(train_images_faces)
    X_train = torch.zeros((num_rows, nn_img_size*nn_img_size))
    y_train = torch.zeros((num_rows, num_classes))

    counter = 0
    for (label, file_names) in enumerate(train_images):
        for file_name in file_names:
            img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (nn_img_size, nn_img_size) , interpolation=cv.INTER_AREA)

            print(label, file_name)
            # print(label, " -- ", file_name, img.shape)

            # fill matrices 
            # X_train - each row is an image vector
            # y_train - one-hot encoded, put only a 1 where the label is correct for the row in X_train
            X_train[counter] = torch.from_numpy(img.flatten().astype(np.float32))
            y_train[counter, label] = 1
            
            counter += 1

    # print(y_train)
    return X_train, y_train

def train(X_train, y_train):
    """Train procedure."""
    # for simplicity of this execise you don't need to find useful hyperparameter
    # I've done this for you already and every test image should work for the
    # given very small trainings database and the following parameters.
    h = 1500
    std = 0.001
    # YOUR CODE HERE
    # initialize W1, W2, b1, b2 randomly
    # Note: W1, W2 should be scaled by variable std

    num_rows = X_train.shape[0]
    x_dim = X_train.shape[1]
    y_dim = y_train.shape[1]

    W1 = std * torch.randn(x_dim, h)
    W2 = std * torch.randn(h, y_dim)

    b1 = torch.randn(1, h)
    b2 = torch.randn(1, y_dim)
    
    # run for num_epochs
    for i in range(num_epochs):

        X_batch = None
        y_batch = None

        # use only a batch of batch_size of the training images in each run
        # sample the batch images randomly from the training set
        sample = torch.randint(high=num_rows, size=(batch_size, 1)).view(batch_size)
        X_batch = X_train[sample]
        y_batch = y_train[sample]

        # forward pass for two-layer neural network using ReLU as activation function
        loss, a2, a1 = forward(X_batch, y_batch, W1, W2, b1, b2)
        
        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)

        if i % 10 == 9: # print every 10th
            print(f"iteration {i+1}: loss {loss}")

        # backward pass
        dCdW1, dCdW2, dCdb1, dCdb2 = backward(a2, a1, X_batch, y_batch, W2)
        
        #print("dCdb1.shape:", dCdb1.shape)
        #print("dCdb2.shape:", dCdb2.shape)

        # depending on the derivatives of W1, and W2 regaring the cost/loss
        # we need to adapt the values in the negative direction of the 
        # gradient decreasing towards the minimum
        # we weight the gradient by a learning rate
        W1 -= learning_rate*dCdW1
        W2 -= learning_rate*dCdW2
        b1 -= learning_rate*dCdb1
        b2 -= learning_rate*dCdb2
        
    return W1, W2, b1, b2

#------------------------------------------------------------------------

X_train, y_train = setup_train()
W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and 
# run prediction by computing the forward pass
test_images = []
test_images.append( (cv.imread('./images/db/test/car.jpg', cv.IMREAD_GRAYSCALE), 0) )
test_images.append( (cv.imread('./images/db/test/flower.jpg', cv.IMREAD_GRAYSCALE), 1) )
test_images.append( (cv.imread('./images/db/test/face.jpg', cv.IMREAD_GRAYSCALE), 2) )

for ti in test_images:
    resized_ti = cv.resize(ti[0], (nn_img_size, nn_img_size) , interpolation=cv.INTER_AREA)
    x_test = resized_ti.reshape(1,-1)
    # YOUR CODE HERE 
    # convert test images to pytorch
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.zeros((1, num_classes))
    y_test[0, ti[1]] = 1
    # do forward pass depending mse or softmax
    a2_test = forward(x_test, y_test, W1, W2, b1, b2)[1]
    print("Test output (values / pred_id / true_id):", a2_test, torch.argmax(a2_test), ti[1])

    

# print("------------------------------------")
# print("Test model output Weights:", W1, W2)
# print("Test model output bias:", b1, b2)


plt.title("Training Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1, num_epochs+1), loss_train_hist, label="Train")
plt.ylim((0, 3.))
plt.xticks(np.arange(1, num_epochs+1, 50.0))
plt.legend()

plt.savefig(f"simple_nn_train_{loss_mode}.png")
plt.show()

# inspired by
# https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/