from abc import ABC

from tensorflow.keras.applications import InceptionV3, EfficientNetB7, ResNet50, EfficientNetB4
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalMaxPool2D, Conv2D, InputLayer
from tensorflow import concat
from config import image_dims


class MultiLabel(Model, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_backbone = InceptionV3(include_top=False,
                                             input_shape=[300, 300, 3],
                                             weights='imagenet',
                                             input_tensor=None,
                                             pooling=False)
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding='same'))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding='same'))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding='same'))

        self.model_pred1_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation='sigmoid')

        self.model_pred2_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation='sigmoid')

        self.model_pred3_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation='sigmoid')

        self.model_pred4_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation='sigmoid')

        self.model_pred5_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation='sigmoid')

        self.model_pred6_2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation='sigmoid')

        """
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding='same'))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(1, 1), padding='same'))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=256, kernel_size=(1, 1), padding="same"))
        self.model.add(GlobalMaxPool2D())
        self.model.add(Dense(num_classes, activation='sigmoid'))
        self.model.summary()
        """

    def call(self, predict_input, **kwargs):
        predict_output = self.model(predict_input)
        pred_1 = predict_output[:, :, :, :400]
        pred_2 = predict_output[:, :, :, 400:800]
        pred_3 = predict_output[:, :, :, 800:1200]
        pred_4 = predict_output[:, :, :, 1200:1600]
        pred_5 = predict_output[:, :, :, 1600:2000]
        pred_6 = predict_output[:, :, :, 2000:2400]

        pred_1 = self.model_pred1_2(pred_1)
        pred_1 = self.model_pred1_3(pred_1)
        pred_1 = self.model_pred1_4(pred_1)
        pred_1 = self.model_pred1_5(pred_1)
        pred_1 = self.model_pred1_6(pred_1)

        pred_2 = self.model_pred2_2(pred_2)
        pred_2 = self.model_pred2_3(pred_2)
        pred_2 = self.model_pred2_4(pred_2)
        pred_2 = self.model_pred2_5(pred_2)
        pred_2 = self.model_pred2_6(pred_2)

        pred_3 = self.model_pred3_2(pred_3)
        pred_3 = self.model_pred3_3(pred_3)
        pred_3 = self.model_pred3_4(pred_3)
        pred_3 = self.model_pred3_5(pred_3)
        pred_3 = self.model_pred3_6(pred_3)

        pred_4 = self.model_pred4_2(pred_4)
        pred_4 = self.model_pred4_3(pred_4)
        pred_4 = self.model_pred4_4(pred_4)
        pred_4 = self.model_pred4_5(pred_4)
        pred_4 = self.model_pred4_6(pred_4)

        pred_5 = self.model_pred5_2(pred_5)
        pred_5 = self.model_pred5_3(pred_5)
        pred_5 = self.model_pred5_4(pred_5)
        pred_5 = self.model_pred5_5(pred_5)
        pred_5 = self.model_pred5_6(pred_5)

        pred_6 = self.model_pred6_2(pred_6)
        pred_6 = self.model_pred6_3(pred_6)
        pred_6 = self.model_pred6_4(pred_6)
        pred_6 = self.model_pred6_5(pred_6)
        pred_6 = self.model_pred6_6(pred_6)

        return concat([pred_1, pred_2, pred_3, pred_4, pred_5, pred_6], axis=1)

    def create_model(self):
        return self.model
