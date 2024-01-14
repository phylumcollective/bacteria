"""
This is a kind of dummy script that mimics one state-action pair of the Beauty robot's actions.
It mimics commands from the rl model's controller to select a pipette, collect attract/repellent solution
and drop it on the plate at a particular location (that the controller determines) before returning to
its home position.
"""

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Input, Dense, Dropout, Lambda, Reshape, MaxPooling2D, LSTM, Reshape
# from tensorflow.keras.models importkdl,kkk'sdee
import os
import sys
import serial
import threading
import cv2
import time
from time import sleep
import numpy as np
import gphoto2 as gp
import argparse
import random
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from uarm.wrapper import SwiftAPI


def loadImg(s, read_as_float32=False, gray=False):
    if read_as_float32:
        img = cv2.imread(s).astype(np.float32) / 255
    else:
        img = cv2.imread(s)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def setupCamera(target="Memory card"):
    # open camera connection
    camera = gp.Camera()
    camera.init()

    # get configuration tree
    config = camera.get_config()

    # find the capture target config item
    capture_target = config.get_child_by_name("capturetarget")

    # set value to Memory card (default) or Internal RAM
    # value = capture_target.get_value()
    capture_target.set_value(target)
    # set config
    camera.set_config(config)

    return camera

# configure the Serial port
def serial_connect(port, baudrate=9600, timeout=1):
    ser = serial.Serial(
        # port='/dev/ttyS1',\
        port=port,
        baudrate=baudrate,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=timeout,
    )
    print("Connected to: " + ser.portstr)
    return ser

def main():
    print(__doc__)

    prevTime = 0
    INTERVAL = args.interval  # default = 30 minutes = 1800 seconds

    callback_obj = gp.check_result(gp.use_python_logging())

    # establish connection with digital camera
    camera = setupCamera()

    # === INITIALIZE ARM ===
    swift = SwiftAPI(filters={"hwid": "USB VID:PID=2341:0042"})

    sleep(2)
    print("device info: ")
    print(swift.get_device_info())
    swift.waiting_ready()
    swift.set_mode(0)  # normal mode

    # Get the initial position of the arm.
    position = swift.get_position(wait=True)
    print(position)

    # Set arm to home position
    # HOME = (100, 0, 20)
    HOME = (200, 0, -10)
    swift.set_buzzer(1000, 0.5)
    swift.set_wrist(90)
    print("moving arm to home position...")
    pos_status = swift.set_position(*HOME, speed=100, wait=True)  # Home
    print("pos_status: ", pos_status)
    sleep(1)

    # === INITIALIZE SERIAL COMMUNICATION WITH ARDUINO ===
    syringe_pump_serial = serial_connect("/dev/cu.usbmodem1441201", 19200, timeout=10)
    syringe_pump_serial.reset_output_buffer()

    # serial reader thread
    class SerialReaderThread(threading.Thread):
        def run(self):
            while True:
                # Read output from ser
                output = syringe_pump_serial.readline().decode("ascii")
                print(output)

    serial_reader = SerialReaderThread()
    serial_reader.start()

    syringe_pump_serial.write(b"S\n")  # put the steppers to sleep

    # === COORDINATES & AMOUNTS ===
    # pipette tip location coords (y should be -127 or less)
    tip_coords = (
        (149.1, -161.4, -87.2),
        (149.1, -160.5, -87.2),
        (149.1, -159.6, -87.2),
    )
    tip_idx = 0

    # attractant/repellent locations and amounts (all amounts are in microliters)
    attractants = {
        "peptone": (
            {"concentration": "high", "amount": 20, "location": (240.5, -149.41, 25)},
            {"concentration": "low", "amount": 20, "location": (240.5, -165.41, 25)},
            {"concentration": "high", "amount": 5, "location": (240.5, -149.41, 25)},
            {"concentration": "low", "amount": 5, "location": (240.5, -165.41, 25)},
        ),
        "dextrose": (
            {"concentration": "high", "amount": 20, "location": (240.5, -181.41, 25)},
            {"concentration": "low", "amount": 20, "location": (240.5, -197.41, 25)},
            {"concentration": "high", "amount": 5, "location": (240.5, -181.41, 25)},
            {"concentration": "low", "amount": 5, "location": (240.5, -197.41, 25)},
        ),
        "lb": (
            {"concentration": "high", "amount": 20, "location": (256.5, -149.41, 25)},
            {"concentration": "low", "amount": 20, "location": (256.5, -165.41, 25)},
            {"concentration": "high", "amount": 5, "location": (256.5, -149.41, 25)},
            {"concentration": "low", "amount": 5, "location": (256.5, -165.41, 25)},
        ),
        "soc": (
            {"concentration": "high", "amount": 20, "location": (256.5, -181.41, 25)},
            {"concentration": "low", "amount": 20, "location": (256.5, -197.41, 25)},
            {"concentration": "high", "amount": 5, "location": (256.5, -171.41, 25)},
            {"concentration": "low", "amount": 5, "location": (256.5, -197.41, 25)},
        ),
    }
    repellents = {
        "co-trimoxazole": (
            {"concentration": "high", "amount": 10, "location": (172.5, -129.41, -5)},
            {"concentration": "low", "amount": 10, "location": (172.5, -145.41, -5)},
            {"concentration": "high", "amount": 1, "location": (172.5, -129.41, -5)},
            {"concentration": "low", "amount": 1, "location": (172.5, -145.41, -5)},
        ),
        "chloramphenicol": (
            {"concentration": "high", "amount": 10, "location": (172.5, -161.41, -5)},
            {"concentration": "low", "amount": 10, "location": (172.5, -177.41, -5)},
            {"concentration": "high", "amount": 1, "location": (172.5, -161.41, -5)},
            {"concentration": "low", "amount": 1, "location": (172.5, -177.41, -5)},
        ),
        "ampicillin": (
            {"concentration": "high", "amount": 10, "location": (188.5, -129.41, -5)},
            {"concentration": "low", "amount": 10, "location": (188.5, -145.41, -5)},
            {"concentration": "high", "amount": 1, "location": (188.5, -129.41, -5)},
            {"concentration": "low", "amount": 1, "location": (188.5, -145.41, -5)},
        ),
        "glacial acetic acid": (
            {"concentration": "high", "amount": 10, "location": (188.5, -161.41, -5)},
            {"concentration": "low", "amount": 10, "location": (188.5, -177.41, -5)},
            {"concentration": "high", "amount": 1, "location": (188.5, -161.41, -5)},
            {"concentration": "low", "amount": 1, "location": (188.5, -177.41, -5)},
        ),
    }

    # plate locations
    plate_coords = ((265, 0, -17), (266, 0, -17))

    # trash location
    trash_coords = (270, -175, 50)

    # === LOAD WORLD MODEL ===
    print("loading world model...")

    # variables for potential modification
    img_height = 128 # this and all below variables should be the same for the trained images and input images
    img_width = 128
    num_channels = 1
    input_shape = (img_height, img_width, num_channels)
    z_len = 2048 # the length of the image compression made by the encoder
    a_len = 1 # the length of the action vector

    # model architectures
    # ==========VAE========================

    # load the vae (have to make the architecture again, make sure the code below
    # matches the code in the Data Prepper/VAE Trainer)
    
    # ====== Encoder ======
    # changing this will make the model exponentially larger or smaller
    latent_dim = 2048

    # the model (saved in x)
    input_img = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(64, 3, padding='same', activation='relu')(input_img)
    x = MaxPooling2D((2,2), padding='same')
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2), padding='same')
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2), padding='same')
    x = Dropout(0.2)(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)

    conv_shape = K.int_shape(x) # Shape of conv to be provided to decoder
    # Flatten
    x = Flatten()(x)
    x = Dense(latent_dim*2, activation='relu')(x)

    # Two outputs, for latent mean and log variance (std. dev.)
    # Use these to sample random variables in latent space to which inputs are mapped. 
    z_mu = Dense(latent_dim, name='latent_mu')(x)
    z_sigma = Dense(latent_dim, name='latent_sigma')(x)

    # REPARAMETERIZATION TRICK
    # Define sampling function to sample from the distribution
    # Reparameterize sample based on the process defined by Gunderson and Huang
    # into the shape of: mu + sigma squared x eps
    # This is to allow gradient descent to allow for gradient estimation accurately.
    def sample_z(args):
        z_mu, z_sigma = args
        eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
        return z_mu + K.exp(z_sigma / 2) * eps

    # sample vector from the latent distribution
    # z is the lamda custom layer we are adding for gradient descent calculations
    # using mu and variance (sigma)
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])

    # Z (lambda layer) will be the last layer in the encoder.
    # Define and summarize encoder model.
    encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
    
    # ==== Decoder ====

    # decoder takes the latent vector as input
    decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

    # Need to start with a shape that can be remapped to original image shape as
    # we want our final utput to be same shape original input.
    # So, add dense layer with dimensions that can be reshaped to desired output shape
    x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
    # reshape to the shape of last conv. layer in the encoder, so we can
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    # upscale (conv2D transpose) back to original shape
    # use Conv2DTranspose to reverse the conv layers defined in the encoder
    x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2))(x)
    # Can add more conv2DTranspose layers, if desired. 
    # Using sigmoid activation
    x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

    # Define and summarize decoder model
    decoder = Model(decoder_input, x, name='decoder')
    # apply the decoder to the latent sample 
    z_decoded = decoder(z)

    # ===== Loss Function =====

    class CustomLayer(keras.layers.Layer):
        def vae_loss(self, x, z_decoded):
            x = K.flatten(x)
            z_decoded = K.flatten(z_decoded)

            recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

            kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
            return K.mean(recon_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            z_decoded = inputs[1]
            loss = self.vae.loss(x, z_decoded)
            self.add_loss(loss, inputs=inputs)
            return x

    # apply the custom loss to the input images and the decoded latent distribution sample
    y = CustomLayer()([input_img, z_decoded])
    # y is basically the original image after encoding input img to mu, sigma, z
    # and decoding sampled z values.
    # This will be used as output for vae

    # ===========RNN===========================

    # Layers
    input_to_rnn = Input(shape=(1,z_len))

    x = LSTM(z_len, return_sequences=True)(input_to_rnn)
    x = Dropout(0.2)(x)
    x = Dense(z_len)(x)
    x = Dropout(0.2)(x)

    rnn_output = Dense(2048, activation='sigmoid')(x)


    # ============Controller=========================

    # Layers
    input_to_controller = Input(shape=(1, z_len*2))

    x = Dense(z_len)(input_to_controller)
    x = Dropout(0.2)(x)
    x = Dense(z_len/2)(x)
    x = Dropout(0.2)(x)
    x = Dense(z_len/4)(input_to_controller)
    x = Dropout(0.2)(x)
    x = Dense(z_len/16)(x)
    x = Dropout(0.2)(x)

    ctrl_output = Dense(a_len, activation='sigmoid')(x)

    # model loading
    # load encoder
    vae = Model(input_img, y, name='vae')
    vae.load_weights(os.getcwd() + "/models/vae.h5")
    encoder = Model(vae.input, vae.layers[15].output)
    # load rnn and controller
    rnn = Model(input_to_rnn, rnn_output, name='rnn')
    rnn.load_weights(os.getcwd() + "/models/rnn.h5")
    ctrl = Model(input_to_controller, ctrl_output, name='controller')
    ctrl.load_weights(os.getcwd() + "/models/cntrl.h5")
    #img_path = os.getcwd() + "/images/2021-03-10/2021-03-12-1615-01-30-22.NEF"



    # === CAPTURE IMAGES, DO MODEL PREDICTION & CONTROLLER ACTIONS ===
    # main program loop
    # capture a frame and show it to the rl model
    # (make sure camera is set to 1:1 ratio & check the dimensions of the frames to confirm)
    # then get an action from the model's controller
    # actions consist of location and attract/repellent (including amount and concentration)
    # can also be "null" or "None" action (i.e. do nothing)
    try:
        while True:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            currTime = time.time()

            # === IMAGE CAPTURE ===
            # with a DSLR/Mirrorless camera

            ### -- grab an image, show it to the world model and get an action --
            if currTime - prevTime >= INTERVAL:
            # ==== grab image from camera and save to disk === #
                prevTime = currTime  # reset time-lapse
                print("Capturing image...")
                file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
                print(
                    "Camera file path: {0}/{1}".format(file_path.folder, file_path.name)
                )
                # rename the file with a timestamp
                if file_path.name.lower().endswith(".jpg"):
                    new_filename = "{}.jpg".format(timestr)
                target = os.path.join("./captures", new_filename)
                print("Copying image to", target)
                camera_file = camera.file_get(
                    file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
                )
                camera_file.save(target)
                # load in image and reshape
                img = loadImg(target)
                #img_array = cv2.imread(img_path)

                # convert to grayscale & resize
                img_array = cv2.imread(img)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                img_array = cv2.resize(img_array, (img_height, img_width), interpolation=cv2.INTER_AREA)
                img_array = img_array.reshape(-1, img_height, img_width, 1)

                # show image to world model & get action
                # this will inlcude the plate coords and the attractant/repellent to drop

                ### --- run the image through the models --- ###

                print("showing image to world model and getting next action...")

                font = cv2.FONT_HERSHEY_PLAIN # cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "Capturing image and showing it to world model...", (10, 450), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                
                # predictions by each piece of the model
                z = encoder.predict(img_array) # encode image

                # show predicted image
                decoded_img = decoder.predict(np.array([z[0][0]]))
                decoded_img_reshaped = decoded_img.reshape(img_height, img_width)
                # show image full screen
                cv2.namedWindow("Beauty", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Beauty", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Beauty", decoded_img_reshaped)

                z = z.reshape(-1, 1, 2048) # reshape for rnn
                zprime = rnn.predict(z) # make prediction (of future image/state)
                z_and_zprime = np.reshape(np.concatenate((z[0][0], zprime[0][0])), (1, z_len*2))[None,:,:] # concat for controller
                action = ctrl.predict(z_and_zprime) # controller returns an action

                # World Model Results
                print("z (encoder): ", z)
                print("z' (rnn prediction): ", zprime)
                print("action: ", action)

                cv2.putText(img, "z (encoder): " + str(z), (10, 470), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img, "z' (rnn prediction): " + str(zprime), (10, 490), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img, "action: " + str(action), (10, 510), font, 1, (255, 255, 0), 1, cv2.LINE_AA)


                # To Simulate the Controller Making Predictions: Random Action
                # action = np.randint(0,8)

                sleep(1)
                
                cv2.putText(img, "taking action... ", (10, 530), font, 1, (255, 255, 0), 1, cv2.LINE_AA)


                print("moving arm into place...")
                swift.set_buzzer(1500, 0.25)
                swift.set_buzzer(1500, 0.25)
                # move arm to pipette tip location and pick up pipette tip
                swift.set_position(
                    tip_coords[tip_idx][0],
                    tip_coords[tip_idx][1],
                    z=15.24,
                    speed=20,
                    timeout=30,
                    wait=True,
                )  # current pipette tip location
                swift.set_position(
                    z=tip_coords[tip_idx][2] + 19, speed=20, timeout=30, wait=True
                )  # acquire pipette
                swift.set_position(
                    z=tip_coords[tip_idx][2] + 9, speed=2, wait=True
                )  # acquire pipette... slowly
                swift.set_position(
                    z=tip_coords[tip_idx][2] + 4, speed=2, timeout=30, wait=True
                )  # acquire pipette
                swift.set_position(
                    z=tip_coords[tip_idx][2], speed=1, timeout=30, wait=True
                )  # acquire pipette... got it
                sleep(1)
                swift.set_position(
                    z=tip_coords[tip_idx][2] + 60, speed=2, timeout=30, wait=True,
                )  # go back up
                sleep(0.1)
                swift.set_position(z=35.24, speed=50, timeout=30, wait=True)  # go back up
                sleep(1)

                # increment tip location
                tip_idx += 1

                # move arm to location of attractant/repellent selected by RL controller
                curr_solution_loc = attractants["peptone"][0]["location"]
                print("extracting attractant/repellent solution...")
                swift.set_position(
                    x=curr_solution_loc[0],
                    y=curr_solution_loc[1],
                    z=40,
                    speed=200,
                    timeout=30,
                    wait=True,
                )  # current attractant/repellent
                swift.set_position(z=curr_solution_loc[2] + 19, speed=20, timeout=30, wait=True)
                swift.set_position(z=curr_solution_loc[2] + 9, speed=3, timeout=30, wait=True)
                swift.set_position(
                    z=curr_solution_loc[2] + 4, speed=3, timeout=30, wait=True
                )  # get closer
                swift.set_position(
                    z=curr_solution_loc[2], speed=3, timeout=30, wait=True
                )  # ease in

                # TODO - need to control the syringe pump stepper
                # extract solution
                syringe_pump_serial.write(b"s\n")  # take the steppers out of sleep mode
                sleep(1)
                syringe_pump_serial.write(b"+\n")  # extract
                sleep(3)

                swift.set_position(
                    z=curr_solution_loc[2] + 60, speed=3, timeout=30, wait=True
                )  # go back up
                # sleep(0.1)
                # swift.set_position(z=30, speed=20, timeout=30, wait=True)  # go back up
                sleep(1)

                # move arm to location on plate (that you get from rl controller)
                print("dispensing attractant/repellent solution...")
                swift.set_position(
                    x=plate_coords[0][0],
                    y=plate_coords[0][1],
                    z=5,
                    speed=20,
                    timeout=30,
                    wait=True,
                )  # current plate location
                swift.set_position(z=plate_coords[0][2] + 19, speed=20, timeout=30, wait=True)
                swift.set_position(z=plate_coords[0][2] + 9, speed=3, timeout=30, wait=True)
                swift.set_position(
                    z=plate_coords[0][2] + 4, speed=3, timeout=30, wait=True
                )  # get closer
                swift.set_position(
                    z=plate_coords[0][2], speed=3, timeout=30, wait=True
                )  # get ready to drop

                # TODO - need to control the syringe pump stepper
                # dispense solution
                syringe_pump_serial.write(b"-\n")  # dispense
                sleep(3)

                swift.set_position(
                    z=plate_coords[0][2] + 40, speed=3, timeout=30, wait=True
                )  # go back up
                sleep(0.1)
                swift.set_position(z=25, speed=20, timeout=30, wait=True)  # go back up
                sleep(1)

                # move arm to trash location
                swift.set_position(
                    x=trash_coords[0],
                    y=trash_coords[1],
                    z=trash_coords[2],
                    speed=50,
                    timeout=30,
                    wait=True,
                )  # current trash location
                swift.set_position(z=trash_coords[2] + 9, speed=20, timeout=30, wait=True)
                swift.set_position(z=trash_coords[2], speed=5, timeout=30, wait=True)

                # TODO - connect pipette to servo
                # dispose of pipette
                swift.set_wrist(0, wait=True)
                sleep(1)
                swift.set_wrist(90, wait=True)
                sleep(1)
                swift.set_position(z=50, speed=20, timeout=30, wait=True)  # go back
                sleep(1)

                # Go back to Home position
                print("moving arm back to home position...")
                swift.set_position(*HOME, speed=50, wait=True)  # Home

                print("dispensing soil remediation solution...")
                # if image is considered more beautful to the AI than the previous image
                # then send solution via the soil stepper
                syringe_pump_serial.write(b"L\n")  # SOIL mode on
                sleep(0.1)
                syringe_pump_serial.write(b"+\n")  # dispense
                sleep(3)
                syringe_pump_serial.write(b"l\n")  # SOIL mode off
                sleep(0.1)

                syringe_pump_serial.write(b"S\n")  # put the steppers back to sleep

                # detach the uArm stepper motors
                # print("putting the uArm to sleep...")
                # swift.send_cmd_sync("M2019")

                print("Done. Waiting for next action...")
                cv2.putText(img, "Done. Waiting for next action", (10, 550), font, 1, (255, 255, 0), 1, cv2.LINE_AA)


                # attach the uArm stepper motors
                # print("waking up the uArm...")
                # swift.send_cmd_sync("M17")

        
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        cv2.destroyAllWindows()
    finally:
        # clean up
        camera.exit()
        print("done")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        required=False,
        default=1800,
        help="timelapse interval (default=1800 seconds (30 minutes))",
    )
    args = parser.parse_args()

    print(" ")
    sys.exit(main())
