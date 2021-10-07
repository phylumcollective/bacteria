import os
import random


def prepare_dataset(data_dir, images_dir, weights_folder):
    # create a file for the labelled data (labelled_data.data)
    with open(data_dir + '/' + 'labelled_data.data', 'w') as data:
        # By using '\n' we move to the next line
        data.write('classes = ' + str(c) + '\n')

        # Location of the train.txt file
        data.write('train = ' + data_dir + '/' + 'train.txt' + '\n')

        # Location of the test.txt file
        data.write('test = ' + data_dir + '/' + 'test.txt' + '\n')

        # Location of the classes.names file
        data.write('names = ' + data_dir + '/' + 'classes.names' + '\n')

        # Location where to save weights
        data.write('backup = ' + weights_folder)

    # Defining list to write paths into
    p = []

    # list the files in the images directory
    for f in os.listdir(images_dir):
        # Checking if filename ends with '.jpg' or 'JPG'
        if(f.endswith(('.JPG', '.jpg'))):
            path_to_save_into_txt_files = images_dir + f

            # Append the line into the list
            # We use '\n' to move to the next line
            # when writing lines into txt files
            p.append(path_to_save_into_txt_files + '\n')

    # Slicing first 15% of elements from the list
    # to write into the test.txt file (to use for testing)
    # randomize it first
    random.shuffle(p)
    p_test = p[:int(len(p) * 0.15)]

    # Deleting from initial list first 15% of elements
    p = p[int(len(p) * 0.15):]

    # -- create train.txt and test.txt files --
    # creating file train.txt and writing 85% of lines in it
    with open(data_dir + '/' + 'train.txt', 'w') as train_txt:
        # Going through all elements of the list
        for e in p:
            # Writing current path at the end of the file
            train_txt.write(e)

    # Creating file test.txt and writing 15% of lines in it
    with open(data_dir + '/' + 'test.txt', 'w') as test_txt:
        # Going through all elements of the list
        for e in p_test:
            # Writing current path at the end of the file
            test_txt.write(e)

    # f_val = open(data_dir + "/test.txt", 'w')
    # f_train = open(data_dir + "/train.txt", 'w')
    #
    # path, dirs, files = next(os.walk(images_dir))
    # data_size = int(len(files) / 2)
    #
    # ind = 0
    # data_test_size = int(0.15 * data_size)  # 15% of files used for testing
    # test_array = random.sample(range(data_size), k=data_test_size)
    #
    # for f in os.listdir(images_dir):
    #     if(f.split(".")[1] == "JPG" or "jpg"):
    #         ind += 1
    #
    #         if ind in test_array:
    #             f_val.write(images_dir+'/'+f+'\n')
    #         else:
    #             f_train.write(images_dir+'/'+f+'\n')

    print("Dataset prepared. Now ready to train YOLO with your custom images and classes")


# location of dataset
DATA_PATH = "data/swarming_yolo"

# run params
SECTION = 'yolo'
RUN_ID = '0000'
DATA_NAME = 'swarming_yolo'
MODEL_FOLDER = 'models/{}/'.format(SECTION)
MODEL_FOLDER += '_'.join([RUN_ID, DATA_NAME])  # where to save the models
print(MODEL_FOLDER)

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)
    os.mkdir(os.path.join(MODEL_FOLDER, 'weights'))  # custom weights folder

data_dir = './data/' + DATA_NAME
images_dir = './data/' + DATA_NAME + '/img/'

# create a file classes.names from the classes.txt that the YOLO format uses
# counter for classes
c = 0

with open(data_dir + '/' + 'classes.names', 'w') as names, \
     open(data_dir + '/' + 'classes.txt', 'r') as txt:

    # go through all lines in txt file and writing them into names file
    for line in txt:
        names.write(line)
        # increment counter
        c += 1

# create files labelled_data.data and train.txt and test.txt for train/test split
prepare_dataset(data_dir, images_dir, MODEL_FOLDER + '/weights/')
