import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras import Model, Input
from .resnet import *
import sys
#tutorial for deploying tensorflow model. not work for subclass model
# https://towardsdatascience.com/how-to-write-a-custom-keras-model-so-that-it-can-be-deployed-for-serving-7d81ace4a1f8

class VirtualModel():
    def __init__(self, mode='train', filename="cnn_cifar10.h5", norm_mean=False, epochs=100, batch_size=32):
        self.mode = mode #train or load
        # self.filename = filename #not used
        self.norm_mean = norm_mean
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10_data()
        # if self.norm_mean:
        #     self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        # else: # linear 0-1
        #     self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train_labels = self.y_train
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)

        #====================== Model =============================
        self.input_shape = self.x_train.shape[1:]
        self.model = MyResnet(self.num_classes)

        if mode=='train':
            print("the model is not trained here.")
        elif mode=='load':
            self.model.load_weights("./custom_model/model_weights/weights")
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    
class MyResnet(Model):
    def __init__(self, class_count, weight_decay=None):
        super().__init__()
        print("using my resnet.")
        if not isinstance(weight_decay, type(None)):
            regularizer = tf.keras.regularizers.L2(weight_decay)
        else:
            regularizer = None
        self.base_model = ResNet18()
        self.base_model.trainable = True
        for layer in self.base_model.layers:
            for attr in ['kernel_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
        self.out_dense = Dense(class_count, kernel_initializer="zeros")#make sure FUKL won't overflow
        self.config = {
            "class_count":class_count,
            "weight_decay":weight_decay,
        }

    def get_config(self):
        return self.config

    def call(self, x):
        x = tf.cast(x, tf.float32)
        base_feature = self.base_model(x)
        base_feature = tf.reduce_mean(base_feature, axis=[1,2])
        raw_y = self.out_dense(base_feature)
        out_y = tf.nn.softmax(raw_y)
        final_result = tf.argmax(out_y, axis=-1)
        return out_y, final_result, raw_y, base_feature


if __name__ == "__main__":
    model = MyResnet50(1)
    model.build((None, 32,32,3))
    for var in model.non_trainable_variables:
        print(var.name, end=", ")