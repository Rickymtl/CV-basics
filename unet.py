from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
import IPython.display as display
from matplotlib import pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# tensorflow implementation of the unet architecure
def load_train_img():
    train_images = []
    train_mask = []
    for i in range(60):
        train_file = "./cat_data/Train/input/cat." + str(i) + ".jpg"
        train_mask_file = "./cat_data/Train/mask/mask_cat." + str(i) + ".jpg"
        train_images.append(cv2.imread(train_file))
        train_images[i] = cv2.resize(train_images[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        train_mask.append(cv2.imread(train_mask_file))
        train_mask[i] = cv2.resize(train_mask[i], (128, 128), interpolation=cv2.INTER_NEAREST)
        train_mask[i] = cv2.cvtColor(train_mask[i], cv2.COLOR_BGR2GRAY)
        train_images[i], train_mask[i] = train_images[i] / 255.0, train_mask[i] / 255.0

    train_images = np.array(train_images)
    train_mask = np.array(train_mask)
    return train_images, train_mask


def load_test_img():
    test_images = []
    test_mask = []
    for i in range(21):
        test_file = "./cat_data/Test/input/cat." + str(i + 60) + ".jpg"
        test_mask_file = "./cat_data/Test/mask/mask_cat." + str(i + 60) + ".jpg"
        test_images.append(cv2.imread(test_file))
        test_images[i] = cv2.resize(test_images[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        test_mask.append(cv2.imread(test_mask_file))
        test_mask[i] = cv2.resize(test_mask[i], (128, 128), interpolation=cv2.INTER_NEAREST)
        test_mask[i] = cv2.cvtColor(test_mask[i], cv2.COLOR_BGR2GRAY)
        test_images[i], test_mask[i] = test_images[i] / 255.0, test_mask[i] / 255.0

    test_images = np.array(test_images)
    test_mask = np.array(test_mask)

    return test_images, test_mask


class CatData(keras.utils.Sequence):
    def __init__(self, kind, batch_size=5, image_size=128):
        self.kind = kind
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.ids = []
        if self.kind == "train":
            for i in range(60):
                self.ids.append(i)
        else:
            for i in range(21):
                self.ids = [i]

    def __load__(self, idx):
        ## Path
        if self.kind == "train":
            image_path = "./cat_data/Train/input/cat." + str(idx) + ".jpg"
            mask_path = "./cat_data/Train/mask/mask_cat." + str(idx) + ".jpg"
        else:
            image_path = "./cat_data/Test/input/cat." + str(idx + 60) + ".jpg"
            mask_path = "./cat_data/Test/mask/mask_cat." + str(idx + 60) + ".jpg"
        ## Reading Image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        ## Normalizaing
        image = image / 255.0
        mask = mask / 255.0

        final_mask = np.zeros((self.image_size, self.image_size, 1))
        final_mask[:, :, 0] = mask[:, :]

        return image, final_mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []

        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


def conv_twice(input, filters, kernel_size=(3, 3), padding="same", strides=1):
    con1 = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(input)
    con2 = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(con1)
    return con2


def Unet():
    input = keras.layers.Input((128, 128, 3))

    con1 = conv_twice(input, 64)
    down1 = keras.layers.MaxPool2D((2, 2), (2, 2))(con1)

    con2 = conv_twice(down1, 128)
    down2 = keras.layers.MaxPool2D((2, 2), (2, 2))(con2)

    con3 = conv_twice(down2, 256)
    down3 = keras.layers.MaxPool2D((2, 2), (2, 2))(con3)

    con4 = conv_twice(down3, 512)
    down4 = keras.layers.MaxPool2D((2, 2), (2, 2))(con4)

    con5 = conv_twice(down4, 1024)

    up1 = keras.layers.UpSampling2D((2, 2))(con5)
    up1 = tf.concat(values=[up1, con4], axis=-1)
    con6 = conv_twice(up1, 512)

    up2 = keras.layers.UpSampling2D((2, 2))(con6)
    up2 = tf.concat(values=[up2, con3], axis=-1)
    con7 = conv_twice(up2, 256)

    up3 = keras.layers.UpSampling2D((2, 2))(con7)
    up3 = tf.concat(values=[up3, con2], axis=-1)
    con8 = conv_twice(up3, 128)

    up4 = keras.layers.UpSampling2D((2, 2))(con8)
    up4 = tf.concat(values=[up4, con1], axis=-1)
    con9 = conv_twice(up4, 64)

    output = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(con9)
    model = keras.models.Model(input, output)
    return model


def dice_coef(y_true, y_pred):
    smooth = 1.

    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


if __name__ == "__main__":
    # reading training images and masks and resizing everything
    img, mask = load_train_img()
    t_img, t_mask = load_test_img()
    model = Unet()
    # optim = keras.optimizers.Adam(learning_rate=0.01)
    # model.compile(optimizer='adadelta', loss="binary_crossentropy", metrics=["acc"])

    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    model.fit(img, mask, epochs=20,batch_size=3, validation_data=(t_img, t_mask), shuffle=True)
    model.save_weights("weight.h5")
    pred_mask = model.predict(t_img)
    # pred_mask = pred_mask > 0.5

    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(np.reshape(t_mask[0] * 255, (128, 128)), cmap="gray")
    #
    #
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(np.reshape(pred_mask[0] * 255, (128, 128)), cmap="gray")
    # plt.show()





    # train_data = CatData("train")
    # test_data = CatData("test")
    # test, tm = load_test_img()
    #
    # x, y = train_data.__getitem__(0)
    # print(x.shape)
    # print(y.shape)
    # model = Unet()
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    # model.summary()
    # model.fit_generator(train_data, epochs=20, validation_data=test_data, steps_per_epoch=12,
    #                     validation_steps=4, shuffle=True)
    # # t_img, t_mask = test_data.__getitem__(1)
    # pred_mask = model.predict(test)
    # pred_mask = pred_mask>0.5
    # model.save_weights("weight.h5")
    #
    # fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(np.reshape(tm[0] * 255, (128, 128)), cmap="gray")
    #
    #
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(np.reshape(pred_mask[0] * 255, (128, 128)), cmap="gray")
    # plt.show()
    #
    # plt.imshow(np.round(pred_mask[0][:, :, 0]), cmap="gray")
    # plt.show()
    # plt.imshow(t_mask[0], cmap="gray")
    # plt.show()
    #
    plt.figure(figsize=(10, 10))
    for i in range(20):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pred_mask[i][:, :, 0], cmap='gray')
        # plt.subplot(5, 10, 2 * (i + 1))
        # plt.xticks([])
        # plt.yticks([])
        # plt.imshow(t_img[i])
    plt.show()
    #
    # # print("ok")
