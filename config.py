# Hyperparameters
global_path = "/run/media/flavios/Data/DATASETS/plant patology/plant-pathology-2021-fgvc8/"
img_class_path = ("train_images/", "train.csv", "test_images/")
save_model_dir = "model/model_resnet50/"
qnt_of_batchs = 30
total_epochs = 30
image_dims = [300, 300, 3]

num_classes = 6
class_threshold = 0.7

# loading parameters.
load_model_from_epoch = 0
load_model = False
save_frequency = 1
save_model = True

# Testing parameters
test_model = False

# Data augmentation.
use_aug_data = True
train_without_test = False
aug_csv_dir = 'data_aug/'
aug_csv_file_name = 'data_aug.csv'

# A factor to control the amount of images to aug between single classes images.
aug_factor = 0.5

# List of strings that represents an technique to artificially augment data.
list_func = ['fl_h', 'fl_v', 'crop', 'r90r', 'r90l']
list_add_aug_func = ['ran_r_fl', 'ran_sat', 'ran_bright', 'ran_contr', 'ran_hue']
threshold_img = 3000
