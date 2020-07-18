import tensorflow as tf
import argparse
from keras.datasets import cifar10
from keras.utils import to_categorical
import random
import numpy as np
import wide_residual_network_fix_v4 as wrn
from  keras.datasets import cifar10
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import LearningRateScheduler
import os, os.path
from keras.applications.vgg16 import preprocess_input
from keras import  backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from numpy import argmax
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model
import simple_cnn as smpcnn
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

NUM_OF_CLASSES = 10
def custom_loss(y_true, y_pred):
    loss  = 0
    try:
        loss_1 = K.categorical_crossentropy(y_true[:, 0:10], y_pred)
        loss_2 = K.categorical_crossentropy(y_true[:, 10:20], y_pred)
        loss = K.minimum(loss_1, loss_2)

        if  loss_1 > loss_2:
            y = y_true[:, 0:10] +0.5*y_true[:, 10:20]
        else:
            y = 0.5 * y_true[:, 0:10] +  y_true[:, 10:20]
    except:
        loss = K.categorical_crossentropy(y_true, y_pred)
        y = y_true

    loss = K.categorical_crossentropy(y, y_pred)
    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='Imubit challange')
    parser.add_argument('--lr', default=1e-3, type=float, dest='learning_rate')
    parser.add_argument('--opt',default='sgd', type=str, dest='optimizer')
    parser.add_argument('--nn', default='vgg16_with_weights', type=str, dest='nueral_net_architecture') #
    parser.add_argument('--bsz',default=32, type=int, dest='batch_size')
    parser.add_argument('--dropout',default=0.25, type=float, dest='dropout')
    parser.add_argument('--epoch',default=10, type=int, dest='epoch')
    parser.add_argument('--steps_per_epoch',default=15, type=int, dest='steps_per_epoch')
    parser.add_argument('--lr_schedule',default=[10,20,60,110,200], type=int,nargs='+', dest='lr_schedule')
    parser.add_argument('--loss', default='categorical_crossentropy', type=str, dest='loss')
    parser.add_argument('--comments', default='', type=str, dest='comments')
    parser.add_argument('--data', default='partial_label', type=str, dest='data') #'partial_label'
    parser.add_argument('--iter', default=3, type=int, dest='num_iteration')

    args = parser.parse_args()
    return args

def generate_dataset(noisy_labels=True):
    ##########################################################
    # generating dataset
    ##########################################################
    NUM_OF_PAIR_IMAGES_TRAIN = 5000
    NUM_OF_IMAGES_VAL = 10000

    (train_data_all, train_gt_all), (val_data, val_gt) = cifar10.load_data()
    random.seed(1)
    train_vid_id_vec = np.random.randint(NUM_OF_PAIR_IMAGES_TRAIN, size=NUM_OF_PAIR_IMAGES_TRAIN)
    val_vid_id_vec = np.random.randint(NUM_OF_IMAGES_VAL, size=NUM_OF_IMAGES_VAL)

    # Train
    train_data = train_data_all[train_vid_id_vec]
    train_gt = train_gt_all[train_vid_id_vec]

    val_data = val_data[val_vid_id_vec]
    val_gt = to_categorical(val_gt[val_vid_id_vec], NUM_OF_CLASSES)

    # normalization and mean substruction
    train_data = train_data.astype('float32')
    val_data = val_data.astype('float32')

    train_img_list =np.zeros([10000, 32,32,3], dtype=np.uint8)
    train_list_gt = np.zeros([10000, 10])

    for i in range(0,NUM_OF_IMAGES_VAL,2):
        state = True
        while state:
            rand_index = np.random.randint(train_data_all.shape[0])
            rand_img = train_data_all[rand_index,:,:,:]
            label_1 = to_categorical(train_gt[i//2], NUM_OF_CLASSES)
            label_2 = to_categorical(train_gt_all[rand_index], NUM_OF_CLASSES)
            if np.max(label_2 + label_1) != 2:
                train_img_list[i,:,:,:]   = train_data[i//2,:,:,:]
                train_img_list[i+1,:,:,:] = rand_img
                if noisy_labels:
                    train_list_gt[i]     = label_2 + label_1
                    train_list_gt[i + 1] = label_2 + label_1
                else:
                    train_list_gt[i]     = label_1
                    train_list_gt[i + 1] = label_2
                state = False

    # train_img_list = preprocess_input(train_img_list,data_format=None)
    # val_data       = preprocess_input(val_data,data_format=None)

    print('prepearing the data')
    return ( train_img_list, train_list_gt, val_data, val_gt)

def step_decay_schedule(initial_lr, decay_factor, step_size):
    def schedule(epoch):
        lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        print('Learning rate for epoch number {} is {}' .format(epoch,lr))
        return lr
    return callbacks.LearningRateScheduler(schedule)

def schedule(epoch_idx):
        schedule = [10,20,30,70,100,130,150]
        if (epoch_idx + 1) < schedule[0]:
            return 0.1
        elif (epoch_idx + 1) < schedule[1]:
            return 0.02  # lr_decay_ratio = 0.2
        elif (epoch_idx + 1) < schedule[2]:
            return 0.004
        return 0.0008

def update_lables(model,trainX,trainY,trainY_orig):
    print('Update labels')
    for i, curr_image in enumerate(trainX):
        curr_image = np.expand_dims(curr_image, axis=0)
        predicition = model.predict(curr_image)
        index = np.where(trainY_orig[i] == 1)

        if predicition[0][index[0][0]] > predicition[0][index[0][1]]:
            trainY[i][index[0][0]] = trainY[i][index[0][0]] * 1.5
            trainY[i][index[0][1]] = trainY[i][index[0][1]] * 0.5

        else:
            trainY[i][index[0][0]] = trainY[i][index[0][0]] * 0.5
            trainY[i][index[0][1]] = trainY[i][index[0][1]] * 1.5

    return trainY

def split(y_true):
    gt  = np.zeros([10000, 20])
    for i,vec in enumerate(y_true):
        print(i)
        index = np.where(vec == 1)
        y_true_1 = to_categorical(index[0][0],NUM_OF_CLASSES)
        y_true_2 = to_categorical(index[0][1],NUM_OF_CLASSES)
        y_true_1 = np.clip(y_true_1 + 0.1,0,1)
        y_true_1 = np.clip(y_true_2 + 0.1,0,1)
        gt[i, 0:10]  = y_true_1
        gt[i,10:20] = y_true_2
    return gt

def main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'imubit_challange_model/imubit_Weights.h5')
    try:
        os.remove(filename)
    except:
        print('no file to delete')

    args = parse_args()
    print(args)
    #########################################
    # Loading original data or Data with partial labeling
    #########################################
    if args.data == 'partial_label':
        train_img_list, train_list_gt, val_data, val_gt = generate_dataset()
    else:
        # loading original cifar10 data
        (train_img_list, train_list_gt), (val_data, val_gt) = cifar10.load_data()
        train_list_gt = to_categorical(train_list_gt)
        val_gt = to_categorical(val_gt)

    #########################################
    #########################################
    #########################################
    train_img_list = train_img_list / 255.
    val_data = val_data / 255.

    name = 'imubit_challange_model'
    if not os.path.exists(name):
        os.makedirs(name)

    trainX = train_img_list
    trainY = train_list_gt
    testX = val_data
    testY = val_gt

    generator = ImageDataGenerator(zca_epsilon=0,
                                   width_shift_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='reflect', )

    generator.fit(trainX, seed=0, augment=True)

    test_generator = ImageDataGenerator(zca_epsilon=0,
                                        horizontal_flip=True,
                                        fill_mode='reflect')

    test_generator.fit(testX, seed=0, augment=True)


    #######################################################
    # Models
    #######################################################
    init_shape = (3, 32, 32) if 0 == 'th' else (32, 32, 3)
    if args.nueral_net_architecture == 'vgg16':
        model = VGG16(weights=None, include_top=True, classes=10, input_shape=(32, 32, 3))
    if args.nueral_net_architecture == 'vgg16_with_weights':
        initial_model = VGG16(weights="imagenet", include_top=False, classes=10, input_shape=(32, 32, 3))
        last = initial_model.output
        x = Flatten()(last)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        preds = Dense(10, activation='softmax')(x)
        model = Model(initial_model.input, preds)
    if args.nueral_net_architecture == 'simple_cnn':
        model = smpcnn.create_simple_cnn()
    if args.nueral_net_architecture == 'vgg19':
        model = VGG19(weights=None, include_top=True, classes=10, input_shape=(32, 32, 3))
    if args.nueral_net_architecture == 'wrn':
        model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=1, dropout=args.dropout)
    model.summary()

    #######################################################
    # optimizer and loss
    #######################################################
    if args.optimizer == 'adam':
        opt = 'adam' #Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    if args.optimizer == 'sgd':
        opt = SGD(lr=args.learning_rate, nesterov=True, decay=0.0005)
    if args.optimizer == 'rmsprop':
        opt = RMSprop(lr=args.learning_rate, decay=1e-6)
    if args.loss == 'categorical_crossentropy':
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
    if args.loss == 'my_loss':
        model.compile(loss=custom_loss, optimizer=opt, metrics=["acc"])
    if args.loss == 'CategoricalHinge':
        model.compile(loss=tf.keras.losses.Hinge(), optimizer=opt, metrics=["acc"])


    #######################################################
    # Callbacks
    #######################################################
    csv_logger = callbacks.CSVLogger(name + '/log.csv', append=True, separator=';')
    save_c = callbacks.ModelCheckpoint(name + '/imubit_Weights.h5', monitor="val_acc", save_best_only=True)
    #lrs = LearningRateScheduler(schedule=schedule)

    lrs  = step_decay_schedule(initial_lr=1e-3, decay_factor=0.8, step_size=10)
    reduce_learning_rate = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, epsilon=0.001,
                                             cooldown=2, min_lr=0.00001)
    #######################################################
    # train
    #######################################################

    if args.loss == 'my_loss':
        history=  model.fit(
            x=trainX,
            y=split(trainY),
            batch_size=args.batch_size,
            epochs=args.epoch,
            callbacks=[csv_logger, save_c, lrs,reduce_learning_rate],
            verbose=1,
            validation_split=0.05 # validation_data=(testX, testY)
        )

    else:
        # if args.num_iteration == 1:
        #     history = model.fit(
        #         x=trainX,
        #         y=trainY,
        #         batch_size=args.batch_size,
        #         epochs=args.epoch,
        #         callbacks=[csv_logger, save_c, lrs],
        #         verbose=1,
        #         validation_split=0.05  # validation_data=(testX, testY)
        #     )
        # else:
        trainY_orig = trainY.copy()
        for i in range(0, args.num_iteration):
            history = model.fit(
                x=trainX,
                y=trainY,
                batch_size=args.batch_size,
                epochs=args.epoch,
                callbacks=[csv_logger, save_c, lrs,reduce_learning_rate],
                verbose=1,
                validation_split=0.05  # validation_data=(testX, testY)
            )
            #######################################################
            # Train intermediate result
            #######################################################
            epoch = history.epoch
            classifier_acc = history.history["acc"]
            classifier_val_acc = history.history["val_acc"]
            plt.plot(epoch[0::], classifier_acc[0::], 'r--', label='train')
            plt.plot(epoch[0::], classifier_val_acc[0::], 'b--', label='val')
            plt.legend(loc='upper right')
            plt.title("Imubit challenge")
            plt.xlabel("Epochs")
            plt.ylabel("accuracy")
            plt.show()

            scores = model.evaluate_generator(test_generator.flow(testX, testY, 1),
                                              (testX.shape[0] / args.batch_size + 1))
            print("_Accuracy = %f" % (100 * scores[1]))

            trainY = update_lables(model, trainX, trainY,trainY_orig)



    #######################################################
    # Train intermediate result
    #######################################################
    epoch = history.epoch
    classifier_acc = history.history["acc"]
    classifier_val_acc = history.history["val_acc"]
    plt.plot(epoch[0::], classifier_acc[0::], 'r--', label='train')
    plt.plot(epoch[0::], classifier_val_acc[0::], 'b--', label='val')
    plt.legend(loc='upper right')
    plt.title("Imubit challenge")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.show()
    #######################################################
    # Final Accuracy measure
    #######################################################
    try:
        scores = model.evaluate_generator(test_generator.flow(testX, testY, 1), (testX.shape[0] / args.batch_size + 1))
        print("_Accuracy = %f" % (100 * scores[1]))
    except:
        counter = 0
        for i, curr_image in enumerate(testX):
            curr_image = np.expand_dims(curr_image, axis=0)
            prediction = argmax(model.predict(curr_image))
            if prediction == argmax(testY[i]):
                counter =counter+1

        print("__Accuracy = %f" % (counter / testX.shape[0]))



if __name__ == "__main__":
    main()
