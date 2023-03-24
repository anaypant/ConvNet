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

    def __call__(self, image, padding=0):
        # assuming image is an np array
        #   raise TypeError("'image' must be type np.ndarray")

        # Performs the convolution, returning a list of numpy arrays that will be fed into the next batch
        total_imgs = []
        for img in image:
            imgs = []
            for x in self.kernel:
                imgs.append(self.convolve_single_filter(img, x, padding=padding))

            for z in imgs:
                total_imgs.append(z)
        return total_imgs


class PoolingLayer:
    def __init__(self, kernel_size=(2, 2), stride=2):
        self.stride = stride
        self.kh, self.kw = kernel_size

    def getMaxFromKernel(self, image, row, col):
        # looks at the row and col of an image and determines the max value in the kernel set
        best = -1e10
        for i in range(row, min(row + self.kh - 1, image.shape[0])):
            for j in range(col, min(col + self.kw - 1, image.shape[1])):
                best = max(best, image[i][j])
        return best

    def pool_single_image(self, image):
        # slide kernel over each part of image
        # get max value of kernel
        # add that to valid location
        pooled = np.zeros(
            (image.shape[0] // self.stride, image.shape[1] // self.stride)
        )
        pooledRow, pooledCol = 0, 0
        for row in range(0, image.shape[0], self.stride):
            for col in range(0, image.shape[1], self.stride):
                pooled[pooledRow][pooledCol] = self.getMaxFromKernel(image, row, col)
                pooledCol += 1
            pooledRow += 1
            pooledCol = 0

        return pooled

    def __call__(self, input):
        # input should be a LIST of all the filters in the np array
        outs = []
        for idx in input:
            outs.append(self.pool_single_image(idx))
        return outs


l1 = ConvLayer(32, stride=1)


def graphConvs(convolutions):
    N = 8
    M = 4  # N * M has to equal num_filters
    for row in range(N):
        for col in range(M):
            idx = (row * M) + col
            if col == 0:
                mini = convolutions[idx]
            else:
                mini = np.concatenate((mini, convolutions[idx]))
        if row == 0:
            concat = mini
        else:
            concat = np.concatenate((concat, mini), axis=1)
    plt.imshow(concat)
    plt.show()


def graphPooled(output, pooled, num_samps=4):
    f, axarr = plt.subplots(num_samps, 2)
    for i in range(min(len(pooled), num_samps)):
        axarr[i, 0].imshow(output[i])
        axarr[i, 0].set_title(output[i].shape)
        axarr[i, 1].imshow(pooled[i])
        axarr[i, 1].set_title(pooled[i].shape)
    plt.show()


image = [x[0]][:]

# concatenate all the filters into an n/4 * n
output = l1(image, padding=1)
# print(output)
print(output[0].shape)
graphConvs(output)

pool = PoolingLayer(kernel_size=(2, 2), stride=2)
pooled = pool(output)
print(pooled[0].shape)
graphPooled(output, pooled)


l2 = ConvLayer(2, kernel_size=(3, 3), stride=1)
second_conv_outputs = l2(pooled, padding=1)
print(len(second_conv_outputs))
graphConvs(second_conv_outputs)


second_pool = PoolingLayer()
second_pooled = second_pool(second_conv_outputs)
graphPooled(second_conv_outputs, second_pooled)

final_inputs = []
for x in second_pooled:
    l = x.flatten().tolist()
    for y in l:
        final_inputs.append(y)


print(len(final_inputs))
