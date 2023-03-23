from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x, y), (_, _) = mnist.load_data()
x = x.astype(np.float64)
x /= 255.0


class ConvLayer:
    def __init__(self, num_filters, kernel_size=(3, 3), stride=1, activation="relu"):
        self.num_filters = num_filters
        self.stride = stride
        self.kernel = []
        self.activation = activation
        for _ in range(self.num_filters):
            self.kernel.append(np.random.uniform(-0.5, 0.5, kernel_size))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def convolve_single_filter(self, image, kernel, padding=0):
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

        if self.activation == "relu":
            return self.relu(output)
        return output

    def __call__(self, image):
        # assuming image is an np array
        if not isinstance(image, np.ndarray):
            raise TypeError("'image' must be type np.ndarray")

        # Performs the convolution, returning a list of numpy arrays that will be fed into the next batch
        imgs = []
        for x in self.kernel:
            imgs.append(self.convolve_single_filter(image, x, padding=0))
        return imgs


class PoolingLayer:
    def __init__(self, kernel_size=(2, 2), stride=1):
        self.stride = stride
        self.kw, self.kh = kernel_size

    def pool_single_image(self, image):
        # returns a single pooled image based on the kernel size
        pass

    def __call__(self, input):
        # input should be a LIST of all the filters in the np array
        pass


l1 = ConvLayer(32)

image = x[0]
# plt.imshow(x[0])
# plt.show()

# concatenate all the filters into an n/4 * n
N = 8
M = int(l1.num_filters / N)

output = l1(image)

counter = 0
for row in range(N):
    for col in range(M):
        idx = (row * M) + col
        if col == 0:
            mini = output[idx]
        else:
            mini = np.concatenate((mini, output[idx]))
    if row == 0:
        concat = mini
    else:
        concat = np.concatenate((concat, mini), axis=1)


plt.imshow(concat)
plt.show()

print(l1.kernel[0])
