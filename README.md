# plant_pathology_kaggle
A solution to the competition "Plant Pathology 2021 - FGVC8" available at https://www.kaggle.com/c/plant-pathology-2021-fgvc8/overview

The dataset is composed with 12 unique combinations of classes for the images, the lowest number of these combinations have 84 images, while the biggest one have more than 4000 images. It's needed an data augmentation to minimize this difference. So, it used a mix of data augmentation techniques to create and produce synthetic images and basically increased the amount of images from 84 to more than 900, while randomly dropping some images on classes with more than 4000 to get closer to 3000 and reduce overfitting.

As well, the classification could be done using the 12 uniques one, so a simple categorical cross entropy loss function activating the last dense layer with a softmax, but it wouldn't be the best choice as even if it have a small chance of one image exists over those 12 categories, the model wouldn't fit that well. So was chosen a multi-label and multi-class classification using sigmoid as activation function and binary cross entropy weighted as loss function.

After some tweaking at the model structure, changing the backbone to EfficientBnet 07 from InceptionV3 which increased the score from 66% to 77%, later, tried a bigger "tail" to the model, creating an transition conv2d layer of 2400 depth to split equally in six to feed six models to classify each one of six class, which increased the score to 81% at epoch 6.

After training over 16 epochs, this is the results:

![score per epoc](https://user-images.githubusercontent.com/66807379/122998397-5ed31380-d383-11eb-9ea7-1dcc214e14c1.png)

Don’t think that more training will get to me better results, as can be seen, the model converge really fast as the dataset with augment its over 20 thousand images, maybe more tweaking on the dataset or use a focal loss with weights for each class to add other step to minimize the overfitting.
