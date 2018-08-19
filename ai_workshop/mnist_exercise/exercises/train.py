from mnist import MNIST
import numpy as np
from helpers import saveims, showims


class MnistDataLoader(object):
    def __init__(self, dirpath='images'):
        mnistdata = MNIST(dirpath)
        images_train, labels_train = mnistdata.load_training()
        self.num_train = len(images_train)
        self.images_train, self.labels_train = np.array(images_train, dtype=np.float32), np.array(labels_train)
        self.images_train /= 255.
        images_test, labels_test = mnistdata.load_testing()
        self.num_test = len(images_test)
        self.images_test, self.labels_test = np.array(images_test, dtype=np.float32), np.array(labels_test)
        self.images_test /= 255.

    def visualize_input(self, input_vector):
        return np.reshape(input_vector, (28,28))

def visualize_dense_weights(dense_weights, kernelsize):
    '''
    :param dense_weights: ndarray of Keras dense layer weights
    :return: list of 2D maps
    '''
    num_in, num_out = dense_weights.shape
    dense_weights = np.reshape(dense_weights, (kernelsize, kernelsize, num_out))
    dense_weights = np.transpose(dense_weights, (2, 0, 1))
    dense_weights = list(dense_weights)
    return dense_weights

if __name__ == '__main__':
    import nets
    from keras.utils import to_categorical
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from tboard import TrainValTensorBoard

    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    #TODO load data

    #TODO: write some code to visualize a random training input and output

    net = nets.build_net_logisticregression()
    net.summary()

    # checkpointer = ModelCheckpoint('checkpoints/cnn_weights_{epoch:02d}.h5')
    checkpointer = ModelCheckpoint('checkpoints/logisticregression_weights_{epoch:02d}.h5')
    # make a Tensorboard callback to visualize training dynamics
    # tensorboarder = TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=True, write_images=True)
    tensorboarder = TrainValTensorBoard(log_dir='./tb_logs', write_graph=True)

    # dense_1 = net.layers[1]
    # weights, biases = dense_1.get_weights()
    # weights = visualize_dense_weights(weights, 28)
    # showims(weights, list(range(len(weights))))
    # saveims(weights, list(range(len(weights))), 'untrained.png')

    # net.fit(x=X_train, y=y_train,
    #         batch_size=BATCH_SIZE,
    #         validation_data=(X_val, y_val),
    #         epochs=NUM_EPOCHS, callbacks=[checkpointer, tensorboarder])

    # weights, biases = dense_1.get_weights()
    # weights = visualize_dense_weights(weights, 28)
    # showims(weights, list(range(len(weights))))
    # saveims(weights, list(range(len(weights))), 'trained.png')

