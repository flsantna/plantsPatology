import pandas as pd
import numpy as np
from config import global_path, img_class_path, threshold_img,\
    list_func, aug_factor, list_add_aug_func


class DataAug(object):
    def __init__(self):
        self.path_to_csv = global_path + img_class_path[1]
        self.factor_aug = aug_factor
        self.list_function = list_func
        self.data_set = pd.read_csv(self.path_to_csv)
        self.df_names = self.data_set['image']
        self.df_labels = self.data_set['labels']
        self.data_set.insert(1, "Aug", ['None'] * (self.data_set.index[-1] + 1))

        self.aug_to_qnt = pd.DataFrame(self.df_labels).value_counts().to_list()
        self.labels = []
        for i in range(len(pd.DataFrame(self.df_labels).value_counts().to_list())):
            self.labels.append(pd.DataFrame(self.df_labels).value_counts().index[i][0])
        print(self.data_set['labels'].value_counts())

        self.dict_missing = dict(zip(self.labels, self.aug_to_qnt))
        self.images = np.ndarray([0, 3])
        self.dataset = np.ndarray([0, 3])
        for k, j in self.dict_missing.items():
            imgs = self.data_set.loc[self.data_set['labels'] == k].to_numpy()
            if imgs.shape[0] > 3300:
                imgs = imgs[:3000]
                self.dataset = np.concatenate([self.dataset, imgs], axis=0)
            else:
                self.dataset = np.concatenate([self.dataset, imgs], axis=0)

            imgs_aug = np.ndarray([0, 3])
            if not j > threshold_img:
                for i in range(len(self.list_function)):
                    if int(np.concatenate([imgs_aug, imgs], axis=0).shape[0]) > 3000:
                        break

                    if 5 * j < 3500:
                        imgs[..., 1] = self.list_function[i]
                        imgs_aug = np.concatenate([imgs_aug, imgs], axis=0)

                    elif j > 1700:
                        imgs[:int(j / 5), 1] = self.list_function[i]
                        imgs_save = imgs[:int(j / 5)]
                        imgs_aug = np.concatenate([imgs_aug, imgs_save], axis=0)

                    elif j > 1500:
                        imgs[:int(j / 4), 1] = self.list_function[i]
                        imgs_save = imgs[:int(j / 4)]
                        imgs_aug = np.concatenate([imgs_aug, imgs_save], axis=0)

                    else:
                        imgs[:int(j/3), 1] = self.list_function[i]
                        imgs_save = imgs[:int(j/3)]
                        imgs_aug = np.concatenate([imgs_aug, imgs_save], axis=0)

                if int(np.concatenate([imgs_aug, imgs], axis=0).shape[0]) < threshold_img:
                    imgs[..., 1] = list_add_aug_func[0]
                    imgs_aug = np.concatenate([imgs_aug, imgs], axis=0)
                    for aug_func in list_add_aug_func[1:]:
                        imgs[..., 1] = list_add_aug_func[0] + "," + aug_func
                        imgs_aug = np.concatenate([imgs_aug, imgs], axis=0)

                self.images = np.concatenate([self.images, imgs_aug])
        self.dataset = np.concatenate([self.dataset, self.images], axis=0)
        np.random.shuffle(self.dataset)

    def create_csv(self, path):
        dataset = pd.DataFrame(self.dataset, columns=self.data_set.columns.to_list())
        return dataset.to_csv(path+'data_aug.csv')
