from keras.layers import (Input,
                          Dense,
                          Activation,
                          BatchNormalization,
                          Reshape,
                          Conv2D,
                          MaxPooling2D,
                          GlobalAveragePooling2D,
                          Dropout,
                          Flatten)

from keras.models import Model
from keras.optimizers import Adam


def build_net_logisticregression():
    print('Making a logistic regression model...')
    net_input = Input(shape=(784,))

    x = Dense(10)(net_input)
    x = Activation('softmax')(x)

    model = Model(inputs=net_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Made a logisitc regression model!')
    return model

def build_net_mlp():
    print('Making a MLP...')
    net_input = Input(shape=(784,))

    x = Dense(128)(net_input) # 1 hidden layer of 128 units
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=net_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Made a MLP!')
    return model

def build_net_cnn():
    print('Making a CNN...')
    net_input = Input(shape=(784,))
    x = Reshape((28,28,1))(net_input)

    x = Conv2D(32, 3, padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=net_input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Made a CNN!')
    return model