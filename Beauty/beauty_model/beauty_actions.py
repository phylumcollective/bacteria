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
from time import sleep
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from uarm.wrapper import SwiftAPI


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

def CustomLayer(keras.layers.Layer):
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

def main():
    print(__doc__)

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
    HOME = (100, 0, 20)
    swift.set_buzzer(1000, 0.5)
    swift.set_wrist(90)
    print("moving arm to home position...")
    pos_status = swift.set_position(*HOME, speed=100, wait=True)  # Home
    print("pos_status: ", pos_status)
    sleep(1)

    # === INITIALIZE SERIAL COMMUNICATION WITH ARDUINO ===
    syringe_pump_serial = serial_connect("/dev/cu.usbmodem1441401", 19200, timeout=10)
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
        (51.5, -127.9, -58),
        (51.5, -137, -58),
        (51.5, -146.1, -58),
    )
    tip_idx = 0

    # attractant/repellent locations and amounts (all amounts are in microliters)
    attractants = {
        "peptone": (
            {"concentration": "high", "amount": 20, "location": (140.5, -129.41, -5)},
            {"concentration": "low", "amount": 20, "location": (140.5, -145.41, -5)},
            {"concentration": "high", "amount": 5, "location": (140.5, -129.41, -5)},
            {"concentration": "low", "amount": 5, "location": (140.5, -145.41, -5)},
        ),
        "dextrose": (
            {"concentration": "high", "amount": 20, "location": (140.5, -161.41, -5)},
            {"concentration": "low", "amount": 20, "location": (140.5, -177.41, -5)},
            {"concentration": "high", "amount": 5, "location": (140.5, -161.41, -5)},
            {"concentration": "low", "amount": 5, "location": (140.5, -177.41, -5)},
        ),
        "lb": (
            {"concentration": "high", "amount": 20, "location": (156.5, -129.41, -5)},
            {"concentration": "low", "amount": 20, "location": (156.5, -145.41, -5)},
            {"concentration": "high", "amount": 5, "location": (156.5, -129.41, -5)},
            {"concentration": "low", "amount": 5, "location": (156.5, -145.41, -5)},
        ),
        "soc": (
            {"concentration": "high", "amount": 20, "location": (156.5, -161.41, -5)},
            {"concentration": "low", "amount": 20, "location": (156.5, -177.41, -5)},
            {"concentration": "high", "amount": 5, "location": (156.5, -161.41, -5)},
            {"concentration": "low", "amount": 5, "location": (156.5, -177.41, -5)},
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
    plate_coords = ((260, 0, -12), (265, 0, -12))

    # trash location
    trash_coords = (260, -160, 40)

    # === LOAD WORLD MODEL ===
    print("loading world model...")
    img_height = 128
    img_width = 128
    num_channels = 1
    input_shape = (img_height, img_width, num_channels)
    z_len = 2048
    a_len = 1

    latent_dim = 2048

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

    conv_shape = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(latent_dim*2, activation='relu')(x)

    z_mu = Dense(latent_dim, name='latent_mu')(x)
    z_sigma = Dense(latent_dim, name='latent_sigma')(x)

    def sample_z(args):
        z_mu, z_sigma = args
        eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
        return z_mu + K.exp(z_sigma / 2) * eps

    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])

    encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')

    decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

    x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2))(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2,2))(x)

    x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

    decoder = Model(decoder_input, x, name='decoder')

    z_decoded = decoder(z)
    # === IMAGE CAPTURE ===
    # with Nikon DSLR/Mirrorless camera as streaming device
    cap = cv2.VideoCapture(0)
    print("camera ready...")

    # Put a Python Counter so we grab an image every x minutes
    counter = 30
    while counter == 30:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTERAREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


    y = CustomLayer()([input_img, z_decoded])

    input_to_rnn = Input(shape=(1,z_len))

    x = LSTM(z_len, return_sequences=True)(input_to_rnn)
    x = Dropout(0.2)(x)
    x = Dense(z_len)(x)
    x = Dropout(0.2)(x)

    rnn_output = Dense(2048, activation='sigmoid')(x)

    # Controller
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
    vae = Model(input_img, y, name='vae')
    vae.load_weights(os.getcwd() + "/models/vae.h5")
    encoder = Model(vae.input, vae.layers[15].output)
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
    while True:
        # TODO - grab an image, show it to the world model and get an action
        # grab an image
        print("getting image...")
        #img_array = cv2.imread(img_path)
        img_array = cv2.imread(frame)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_array = cv2.resize(img_array, (img_height, img_width))
        img_array = img_array.reshape(-1, img_height, img_width, 1)
        # show image to world model & get action
        # this will inlcude the plate coords and the attractant/repellent to drop
        print("showing image to world model and getting next action...")
        z = encoder.predict(img_array)
        z = z.reshape(-1, 1, 2048)
        zprime = rnn.predict(z)
        z_and_zprime = np.reshape(np.concatenate((z[0][0], zprime[0][0])), (1, z_len*2))[None,:,:]
        action = ctrl.predict(z_and_zprime)

        #World Mdoel Results
        print("z: ", z)
        print("z': ", zprime)
        print("action: ", action)

        # To Simulate the Controller Making Predictions: Random Action
        action = np.randint(0,8)

        sleep(1)

        print("moving arm into place...")
        swift.set_buzzer(1500, 0.25)
        swift.set_buzzer(1500, 0.25)
        # move arm to pipette tip location and pick up pipette tip
        swift.set_position(
            tip_coords[tip_idx][0],
            tip_coords[tip_idx][1],
            z=35.24,
            speed=200,
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
        swift.set_position(z=35.24, speed=200, timeout=30, wait=True)  # go back up
        sleep(1)

        # increment tip location
        tip_idx += 1

        # move arm to location of attractant/repellent selected by RL controller
        curr_solution_loc = attractants["dextrose"][0]["location"]
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
            z=25,
            speed=200,
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
        swift.set_position(z=50, speed=20, timeout=30, wait=True)  # go back up
        sleep(1)

        # move arm to trash location
        swift.set_position(
            x=trash_coords[0], y=trash_coords[1], z=65, speed=200, timeout=30, wait=True
        )  # current trash location
        swift.set_position(z=trash_coords[2] + 19, speed=20, timeout=30, wait=True)
        swift.set_position(z=trash_coords[2], speed=5, timeout=30, wait=True)

        # TODO - connect pipette to servo
        # dispose of pipette
        swift.set_wrist(0, wait=True)
        sleep(1)
        swift.set_wrist(90, wait=True)
        sleep(1)
        swift.set_position(z=20, speed=20, timeout=30, wait=True)  # go back up
        sleep(1)

        # Go back to Home position
        print("moving arm back to home position...")
        swift.set_position(*HOME, speed=50, wait=True)  # Home

        print("dispensing soil remediatin solution...")
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

        # wait (30 minutes in the real system, 10 secs here for testing)
        print("waiting for next action...")
        sleep(10)

        # attach the uArm stepper motors
        print("waking up the uArm...")
        swift.send_cmd_sync("M17")


if __name__ == "__main__":
    main()
