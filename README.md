# AI workshop excercises and demos

* Simple interactive Tic Tac Toe program for the console
* MNIST handwritten digit classification with neural networks
* Artistic style transfer with neural networks

##Data

MNIST handwritten digit data

* train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
* train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
* t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
* t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

should be downloaded from http://yann.lecun.com/exdb/mnist/

First navigate to the folder to download data to:
```buildoutcfg
cd /path/to/ai_workshop/ai_workshop/mnist_exercise/exercises/images
```
And download the 4 archives:
```buildoutcfg
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```
And finally unzip them
```buildoutcfg
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

## More cool MNIST visualizations

http://scs.ryerson.ca/~aharley/vis/
