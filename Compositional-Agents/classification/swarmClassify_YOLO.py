import os
import random


def prepare_dataset(images_dir, weights_folder):
    # create a file for the labelled data (labelled_data.data)
    with open(images_dir + '/' + 'labelled_data.data', 'w') as data:
        # By using '\n' we move to the next line
        data.write('classes = ' + str(c) + '\n')

        # Location of the train.txt file
        data.write('train = ' + images_dir + '/' + 'train.txt' + '\n')

        # Location of the test.txt file
        data.write('test = ' + images_dir + '/' + 'test.txt' + '\n')

        # Location of the classes.names file
        data.write('names = ' + images_dir + '/' + 'classes.names' + '\n')

        # Location where to save weights
        data.write('backup = ' + weights_folder)


    f_val = open("test.txt", 'w')
    f_train = open("train.txt", 'w')

    path, dirs, files = next(os.walk(images_dir))
    data_size = len(files)

    ind = 0
    data_test_size = int(0.2 * data_size)  # 20% of files used for testing
    test_array = random.sample(range(data_size), k=data_test_size)

    for f in os.listdir(images_dir):
        if(f.split(".")[1] == "JPG" or "jpg"):
            ind += 1

            if ind in test_array:
                f_val.write(images_dir+'/'+f+'\n')
            else:
                f_train.write(images_dir+'/'+f+'\n')

    print("Dataset prepared. Now ready to train YOLO with your custom images and classes")


# location of dataset
DATA_PATH = "data/swarming"  # '/content/drive/My Drive/research/deep_learning/GDL_code/data/bacteria/'

# run params
SECTION = 'yolo'
RUN_ID = '0000'
DATA_NAME = 'swarming_yolo'
MODEL_FOLDER = 'models/{}/'.format(SECTION)
MODEL_FOLDER += '_'.join([RUN_ID, DATA_NAME])  # where to save the models
print(MODEL_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    os.mkdir(os.path.join(MODEL_FOLDER, 'weights'))

images_dir = './data/' + DATA_NAME

# create a file classes.names from the classes.txt that the YOLO format uses
# counter for classes
c = 0

with open(images_dir + '/' + 'classes.names', 'w') as names, \
     open(images_dir + '/' + 'classes.txt', 'r') as txt:

    # go through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)
        # increment counter
        c += 1

# create files labelled_data.data and train.txt and test.txt for train/test split
prepare_dataset(images_dir, MODEL_FOLDER + '/weights/')
