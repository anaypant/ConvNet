import numpy as np
from PIL import Image
import torch
from keras.datasets import mnist


(train_X, train_y), (test_X, test_Y) = mnist.load_data()


def relu(x):
    return np.maximum(0, x)


def convolve_data(data, all_kernels, padding=0):
    # returns the image convolved by all different kernels. This means that we should end up with num_kernels * num_images as an output
    imgs = []
    for x in all_kernels:
        imgs.append(convolve(data, x, padding=padding))
    return imgs


def convolve(image, kernel, padding=0):
    # Get input and kernel dimensions
    h, w = image.shape
    kh, kw = kernel.shape

    # Calculate the output shape
    oh, ow = h - kh + 1, w - kw + 1

    # Apply padding if specified
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
        oh, ow = h + 2 * padding - kh + 1, w + 2 * padding - kw + 1

    # Create an empty output array
    output = np.zeros((oh, ow))

    # Slide the kernel over the input image and compute the dot product at each position
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i : i + kh, j : j + kw] * kernel)

    return output


def max_pool2d(input, kernel_size, stride=1):
    # Get input dimensions
    h, w = input.shape

    # Calculate the output shape
    oh = (h - kernel_size) // stride + 1
    ow = (w - kernel_size) // stride + 1

    # Create an empty output array
    output = np.zeros((oh, ow))

    # Slide the kernel over the input and take the maximum value in each region
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.max(
                input[
                    i * stride : i * stride + kernel_size,
                    j * stride : j * stride + kernel_size,
                ]
            )

    return output


# import all of the images and store them in a list of lists
directory = "indoor/Images/"
image = Image.open(directory + "airport_inside/airport_inside_0001.jpg").convert("L")
image.show()
np_image = np.asarray(image)


# expressing the image as an np array
# np_image = np.asarray(train_X[0])


# Creating a kernel of random values size N*N to convolve against the image
N = 3  # kernel size
num_kernels = 32
kernels = [
    np.array(
        [
            [np.random.uniform(-(1 / np.sqrt(N)), 1 / np.sqrt(N)) for _ in range(N)]
            for _ in range(N)
        ]
    )
    for _ in range(num_kernels)
]

# convolving the image


# Convolutional layer 1:
# Apply a 2D convolution with 32 filters of size 3x3,
# followed by a ReLU activation function.
# This layer extracts 32 different 3x3 features from the input image.

filtered_imgs = relu(convolve_data(np_image, kernels, padding=0))

# Max pooling layer 1:
# Apply max pooling with a pool size of 2x2 to reduce
# the spatial dimensions of the output from the previous layer by a factor of 2.
pooled_imgs = []
for x in filtered_imgs:
    pooled_imgs.append(max_pool2d(x, 2, stride=1))
    Image.fromarray(pooled_imgs[-1]).show()


# display all the filters on a subplot


# Convolutional layer 2:
# Apply a 2D convolution with 64 filters of size 3x3,
# followed by a ReLU activation function.
# This layer extracts 64 different 3x3 features from the output of the previous layer.

N = 3  # kernel size
num_kernels = 64
kernels = [
    np.array(
        [
            [np.random.uniform(-(1 / np.sqrt(N)), 1 / np.sqrt(N)) for _ in range(N)]
            for _ in range(N)
        ]
    )
    for _ in range(num_kernels)
]
# filtered_imgs = relu(convolve_data(pooled_imgs))
print(pooled_imgs[0].shape)
