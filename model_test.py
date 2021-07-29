from config import load_model_from_epoch, class_threshold, use_aug_data, aug_csv_dir,\
    total_epochs, qnt_of_batchs, global_path, img_class_path, aug_csv_file_name,save_model_dir
from bin.model import MultiLabel
from bin.data_proc import DataCreator
import csv
import pandas as pd
import tensorflow as tf
import time
import os

if __name__ == '__main__':
    if use_aug_data:
        data = DataCreator(paths=aug_csv_dir + aug_csv_file_name, augment=True)
    else:
        data = DataCreator()
    #Hyperparameters.
    output_dir = "out/"
    test_dir = global_path + img_class_path[2]
    path_load_model = "model/model_complete_with_augEpoch:20/"

    #Hyperparameters.
    epochs = total_epochs
    batch_size = qnt_of_batchs
    step_count = data.dataset_img_train.shape[0] // batch_size
    step_test_data = data.dataset_img_test.shape[0] // batch_size

    image_dims = (300,300,3)
    data_set = pd.read_csv("data_aug/data_aug.csv")
    df_labels = data_set['labels']
    one_hot = df_labels.str.get_dummies(sep=" ")
    dataset_labels = one_hot.columns.to_list()


    # Initialization of model and data creator class.
    model = tf.keras.models.load_model(path_load_model)
    model.load_weights(filepath=save_model_dir + "epoch-{}".format(load_model_from_epoch))

    # Processing to create submission file.
    images_path_list = sorted(list(os.listdir(test_dir)))
    def test_on_sub (index):
        input_img = tf.io.read_file(test_dir+images_path_list[index])
        image = tf.io.decode_image(contents=input_img, channels=3, dtype=tf.dtypes.float32)
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)
    
    values = []
    for i in range(len(images_path_list)):
        name, images = test_on_sub(index=i)
        test_values = model.call(images)
        index_values = [i for i, v in enumerate(test_values[0]) if v > class_threshold]
        classes = dataset_labels
        classes_img = ""
        for i in index_values:
            classes_img = str(classes[i])+" "+classes_img
        values.append([name, classes_img])
    csv_pd = pd.DataFrame(values, columns=['image', 'labels'], index=None)
    #csv_pd.to_csv(output_dir + 'submission.csv',index=False)
    print(csv_pd)

    # Process to test model on its full dataset.
    all_thresholds = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accuracy_per_threshold = []
    values = []
    for i in range(step_test_data):
        dataset_classes, dataset_images = data.get_test_dataset(index=i)
        test_values = model.call(dataset_images)
        values.append([test_values, dataset_classes])

    values_train_data = []
    for i in range(step_test_data):
        dataset_classes, dataset_images = data.get_train_dataset(index=i)
        test_values = model.call(dataset_images),
        values_train_data.append([test_values, dataset_classes])

    for threshold in all_thresholds:
        accuracy = 0
        for j in values:
            for i in range(batch_size):
                if [i for i, v in enumerate(j[0][i]) if v > threshold] == \
                        [i for i, v in enumerate(j[1][i]) if v > threshold]:
                    accuracy += 1
        print('Number of corrected predicted images with threshold in test data {}: {} '.format(threshold, accuracy),
              'de {}'.format(data.dataset_img_test.shape[0]))
        print("Precision :{:.2f}%".format((accuracy / data.dataset_img_test.shape[0]) * 100))
        accuracy_per_threshold.append(["Threshold:{}".format(threshold), accuracy / data.dataset_img_test.shape[0]])

    for threshold in all_thresholds:
        accuracy = 0
        for j in values_train_data:
            for i in range(batch_size):
                if [i for i, v in enumerate(j[0][i]) if v > threshold] == \
                        [i for i, v in enumerate(j[1][i]) if v > threshold]:
                    accuracy += 1
        print('Number of corrected predicted images with threshold in train data {}: {} '.format(threshold, accuracy),
              'de {}'.format(data.dataset_img_test.shape[0]))
        print("Precision :{:.2f}%".format((accuracy / data.dataset_img_test.shape[0]) * 100))
        accuracy_per_threshold.append(["Threshold:{}".format(threshold), accuracy / data.dataset_img_test.shape[0]])

