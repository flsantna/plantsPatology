from config import global_path, img_class_path, qnt_of_batchs, image_dims,\
    train_without_test
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


# Needed to do a data augmentation to have a better distribution of the data.
class DataCreator(object):
    def __init__(self, paths=global_path + img_class_path[1], augment=False):
        self.path_to_csv = paths
        self.augment = augment
        self.images_path_list = sorted(list(os.listdir(global_path + img_class_path[2])))
        self.batch = qnt_of_batchs
        self.data_set = pd.read_csv(self.path_to_csv)

        if not self.augment:
            self.data_set = self.data_set.sample(frac=1).reset_index(drop=True)
            self.df_names = self.data_set['image']
            self.df_labels = self.data_set['labels']
            self.one_hot = self.df_labels.str.get_dummies(sep=" ")

        else:
            self.aug = pd.DataFrame(self.data_set['Aug'])
            self.df_names = self.data_set['image']
            self.df_labels = self.data_set['labels']
            self.one_hot = self.df_labels.str.get_dummies(sep=" ")
            self.one_hot = pd.concat([self.aug, self.one_hot], axis=1)

        # Dataset train and test ratio to test while training or not.
        data_set_train_rt = 0.15
        if train_without_test:
            data_set_train_rt = 5

        self.dataset_img_train, self.dataset_img_test, self.dataset_class_train, self.dataset_class_test = \
            train_test_split(self.df_names.to_numpy().reshape(self.df_names.shape[0],)
                             , self.one_hot.to_numpy(), test_size=data_set_train_rt, shuffle=False)
        self.dataset_labels = self.one_hot.columns[1:].to_list()
        self.dataset_size = self.df_names.to_numpy().shape[0]
        self.aug_control_train, self.aug_control_test = self.dataset_class_train[..., 0]\
            , self.dataset_class_test[..., 0]
        self.dataset_class_test = self.dataset_class_test[..., 1:]
        self.dataset_class_train = self.dataset_class_train[..., 1:]

    @staticmethod
    def flip_h(img):
        img_fliped = tf.image.flip_left_right(img)
        return img_fliped

    @staticmethod
    def flip_v(img):
        img_fliped = tf.image.flip_up_down(img)
        return img_fliped

    @staticmethod
    def random_crop(img):
        img_crop = tf.image.central_crop(img, central_fraction=0.8)
        return img_crop

    @staticmethod
    def rotate90l(img):
        img_rot = tf.image.rot90(img, k=3)
        return img_rot

    @staticmethod
    def rotate90r(img):
        img_rot = tf.image.rot90(img)
        return img_rot

    @staticmethod
    def rand_rot_flip(img):
        img = tf.expand_dims(img, axis=0)
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])
        image = data_augmentation(img)
        return tf.squeeze(image, axis=0)

    def __img_read(self, indexes, aug=None):
        images = []
        if self.augment:
            for i in range(indexes.shape[0]):
                input_img = tf.io.read_file(global_path + img_class_path[0] + indexes[i])
                tensor_img = tf.io.decode_image(contents=input_img, channels=3, dtype=tf.dtypes.float32)
                if aug[i] == 'None':
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
                elif aug[i] == 'fl_h':
                    tensor_img = self.flip_h(tensor_img)
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
                elif aug[i] == 'fl_v':
                    tensor_img = self.flip_v(tensor_img)
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
                elif aug[i] == 'crop':
                    tensor_img = self.random_crop(tensor_img)
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
                elif aug[i] == 'r90r':
                    tensor_img = self.rotate90r(tensor_img)
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
                elif aug[i] == 'r90l':
                    tensor_img = self.rotate90l(tensor_img)
                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)

                else:
                    for double_aug in list(aug[i].split(sep=",")):
                        if double_aug == 'ran_r_fl':
                            tensor_img = self.rand_rot_flip(tensor_img)
                        elif double_aug == 'ran_sat':
                            tensor_img = tf.image.adjust_saturation(tensor_img, 0.5)

                        elif double_aug == 'ran_bright':
                            tensor_img = tf.image.adjust_brightness(tensor_img, 0.3)

                        elif double_aug == 'ran_contr':
                            tensor_img = tf.image.adjust_contrast(tensor_img, 0.4)

                        elif double_aug == 'ran_hue':
                            tensor_img = tf.image.adjust_hue(tensor_img, 0.5)

                    tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                    images.append(tensor_img)
        else:
            for i in indexes:
                input_img = tf.io.read_file(global_path + img_class_path[0] + i)
                tensor_img = tf.io.decode_image(contents=input_img, channels=3, dtype=tf.dtypes.float32)
                tensor_img = tf.image.resize(tensor_img, [image_dims[0], image_dims[1]])
                images.append(tensor_img)
        return tf.convert_to_tensor(images, dtype=tf.dtypes.float32)

    def get_train_dataset(self, index):
        batch_index = self.dataset_img_train[self.batch * index:self.batch + self.batch * index]
        if self.augment:
            aug = self.aug_control_train[self.batch * index:self.batch + self.batch * index]
            image = self.__img_read(indexes=batch_index, aug=aug)
        else:
            image = self.__img_read(indexes=batch_index)
        return self.dataset_class_train[self.batch * index:self.batch + self.batch * index].astype(np.float32), image

    def get_test_dataset(self, index):
        batch_index = self.dataset_img_test[self.batch * index:self.batch + self.batch * index]
        if self.augment:
            aug = self.aug_control_test[self.batch * index:self.batch + self.batch * index]
            image = self.__img_read(indexes=batch_index, aug=aug)
        else:
            image = self.__img_read(indexes=batch_index)
        return self.dataset_class_test[self.batch * index:self.batch + self.batch * index].astype(np.float32), image

    def test_on_sub(self, index):
        input_img = tf.io.read_file(global_path + img_class_path[2] + self.images_path_list[index])
        image = tf.io.decode_image(contents=input_img, channels=3, dtype=tf.dtypes.float32)
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = self.images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)
