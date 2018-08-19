import nets
import cv2
import numpy as np
from helpers import showims


class MnistPredictor(object):
    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self.net = self.load_model()

    def load_model(self):
        # net = nets.build_net_logisticregression()
        # net = nets.build_net_mlp()
        net = nets.build_net_cnn()

        if self.weights_path:
            net.load_weights(self.weights_path)

        return net

    def predict_image(self, im):
        im = cv2.resize(im, (28, 28))  # shrink down to net input size
        im = np.int32(im) # cast to integer type
        im = im / 255.  # normalize
        # showims([im], ['im to be classified']) # visualize image
        X = np.reshape(im, (1, 784))  # flatten into an input vector
        class_probs = self.net.predict(X)[0] # forward pass through the network
        category_index = np.argmax(class_probs) # get the category with highest predicted probability
        return category_index