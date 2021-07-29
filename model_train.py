from config import load_model_from_epoch, save_model, test_model, use_aug_data, aug_csv_dir, \
    total_epochs, qnt_of_batchs, load_model, save_model_dir, save_frequency, aug_csv_file_name, \
    train_without_test, class_threshold, image_dims
from bin.model import MultiLabel
from bin.data_proc import DataCreator
import pandas as pd
import tensorflow as tf
import time

if __name__ == '__main__':
    # Requirement to set up a virtual GPU and limit the amount of used video memory by tensorflow.
    # Particularly effective if is using the algorithm with a virtual desktop, as tensorflow
    # tries to allocates all of gpu memory.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


    def weighted_binary_crossentropy(target, output):
        """
        Weighted binary crossentropy between an output tensor
        and a target tensor. POS_WEIGHT is used as a multiplier
        for the positive targets.

        Combination of the following functions:
        * keras.losses.binary_crossentropy
        * keras.backend.tensorflow_backend.binary_crossentropy
        * tf.nn.weighted_cross_entropy_with_logits
        """
        # Transform back to logits.
        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=tf.dtypes.float32)
        output = tf.clip_by_value(output, epsilon, 1 - epsilon)
        output = tf.math.log(output / (1 - output))
        # compute weighted loss
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                        logits=output,
                                                        pos_weight=5)
        return tf.reduce_mean(loss, axis=-1)


    def trainstep(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            predicted_labels = model.call(batch_images)
            loss_value = weighted_binary_crossentropy(target=batch_labels, output=predicted_labels)
            #print(predicted_labels[:10])
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        loss_metric.update_state(values=loss_value)


    def test_step(batch_images, batch_label):
        predict = model.call(batch_images)
        val_loss_value = weighted_binary_crossentropy(target=batch_label, output=predict)
        val_loss_metric.update_state(values=val_loss_value)


    # Initialization of Data Creator class to use an CSV with augment data if "augment=True" or use
    # the train.csv provided with dataset processed.
    if use_aug_data:
        data = DataCreator(paths=aug_csv_dir + aug_csv_file_name, augment=True)

    else:
        data = DataCreator()
    # Model initialization.
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    # Hyperparameters from config.py file.
    epochs = total_epochs
    batch_size = qnt_of_batchs
    step_count = data.dataset_img_train.shape[0] // batch_size
    step_test_data = data.dataset_img_test.shape[0] // batch_size

    # Initializing Loss metrics.
    val_loss_metric = tf.metrics.Mean()
    loss_metric = tf.metrics.Mean()
    loss_metric.reset_states()

    # Optimizer. Adadelta because was the only one that properly resulted in a functional training model,
    # the traditional Adam resulted in a common value across all images and batches.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000,
                                                                 decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adadelta(rho=0.92, epsilon=0.001)

    # Load the pre-trained weights if chosen at config.py.
    if load_model:
        model.load_weights(filepath=save_model_dir + "epoch-{}".format(load_model_from_epoch))
        model.summary()
        print("Successfully loaded model!")
    else:
        load_weights_from_epoch = -1
        model.summary()

    # Training loop through epochs and steps.
    if not test_model:
        accuracy_per_epoch = []
        for epoch in range(load_model_from_epoch + 1, epochs):
            start_time = time.time()
            for i in range(step_count):
                dataset_classes, dataset_images = data.get_train_dataset(index=i)
                if save_model_dir.split(sep="/")[-2] == 'model_efficientb07':
                    dataset_images = dataset_images * 255 # Returning the image without normalization
                                                                    # as the Efficient B07 model
                                                                    # already have it in a normalization layer.
                trainstep(batch_images=dataset_images, batch_labels=dataset_classes)
                time_per_step = (time.time() - start_time) / (i + 1)
                print("Epoch: {}/{}, Step: {}/{}, {:.2f}s/step, Loss: {:.5f}, "
                      "Estimated time to end epoch:"
                      "{:.0f}h:{:.0f}m".format(epoch,
                                               epochs,
                                               i,
                                               step_count,
                                               time_per_step,
                                               loss_metric.result(),
                                               time.localtime((time_per_step * step_count) + start_time).tm_hour,
                                               time.localtime((time_per_step * step_count) + start_time).tm_min))
            loss_metric.reset_states()
            if not train_without_test:
                for i in range(step_test_data):
                    dataset_classes_test, dataset_images_test = data.get_test_dataset(i)
                    if save_model_dir.split(sep="/")[-2] == 'model_efficientb07':
                        dataset_images_test = dataset_images_test * 255 # Returning the image without normalization
                                                                    # as the Efficient B07 model
                                                                    # already have it in a normalization layer.
                    test_step(batch_images=dataset_images_test, batch_label=dataset_classes_test)
                    print(f"Val-loss = {val_loss_metric.result()}")
            print(f"Validation Loss Epoch {epoch} = {val_loss_metric.result()}")

            if save_model:
                if epoch % save_frequency == 0:
                    model.save_weights(filepath=save_model_dir + "epoch-{}".format(epoch), save_format="tf")
        if save_model:
            model.save_weights(filepath=save_model_dir + "last_epoch-{}".format(total_epochs), save_format="tf")
    else:
        values = []
        for i in range(len(data.images_path_list)):
            name, images = data.test_on_sub(index=i)
            test_values = model.call(images)
            index_values = [i for i, v in enumerate(test_values[0]) if v > class_threshold]
            classes = data.dataset_labels
            classes_img = ""
            for j in index_values:
                classes_img = str(classes[j]) + " " + classes_img
            values.append([name, classes_img])
        csv_pd = pd.DataFrame(values, columns=['image', 'labels'], index=None)
        csv_pd.to_csv(save_model_dir + 'submission.csv', index=False)
        model.save(filepath=save_model_dir + 'model_complete_with_aug' + 'Epoch:{}'
                   .format(load_model_from_epoch), save_format="tf")
"""
accuracy = 0
                accuracy_per_epoch = []
                for j in values:
                    for i in range(batch_size):
                        if [i for i, v in enumerate(j[0][i]) if v > class_threshold] == \
                                [i for i, v in enumerate(j[1][i]) if v > class_threshold]:
                            accuracy += 1
                print('Number of corrected predicted images:', accuracy, 'de {}'.format(data.dataset_img_test.shape[0]))
                print("Precision :{:.2f}%".format((accuracy / data.dataset_img_test.shape[0]) * 100))
                accuracy_per_epoch = [epoch, accuracy / data.dataset_img_test.shape[0]]
                csv_pd = pd.read_csv(save_model_dir + 'precision_per_epoch.csv')
                csv_append = pd.Series(accuracy_per_epoch, index=csv_pd.columns)
                csv_pd = csv_pd.append(csv_append, ignore_index=True)
                csv_pd.to_csv(save_model_dir + 'precision_per_epoch.csv', index=False)
"""