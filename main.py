#%%

# imports
# data, graph, files
import os
import pathlib
import shutil
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import clear_output
from PIL import Image

# AI
import numpy as np
import tensorflow as tf

from keras import models
from keras import layers

# GPU stats
import GPUtil
import time

from threading import Thread


# global variables
SEED = 42
GPUDELAY = 0.1
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATASET_PATH = 'D:\\training_data\\handwriting\\data\\by_class'
LABELS = []

EPOCH = 5

def main():
    #
    # load data
    #
    data_dir = pathlib.Path(DATASET_PATH)
    # if not data_dir.exists():
    #     tf.keras.utils.get_file(
    #         'by_class.zip',
    #         origin='https://s3.amazonaws.com/nist-srd/SD19/by_class.zip',
    #         extract=True,
    #         cache_dir='.',
    #         cache_subdir='data'
    #     )

    LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))

    # # # rename_and_move_file # ()

    print(LABELS)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        batch_size=32,
        color_mode='grayscale',
        image_size=(128, 128),
        seed=0,
        validation_split=0.3,
        subset='both'
    )

    # train_ds = train_ds.shard(num_shards=2, index=0)
    # val_ds = train_ds.shard(num_shards=2, index=1)

    print(train_ds.element_spec)

    #
    # data preprocessing (labeling, split train/eval/test, caching/shuffling)
    #
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    for exp_data, exp_label in train_ds.take(1):
        # print(exp_data.shape, "\n", exp_label.shape, "\n", exp_label)
        pass

    # verify image and labels match
    # plt.figure(figsize=(12, 12))
    # ROWS = 3
    # COLS = 3
    # N = ROWS * COLS
    # for i in range(N):
    #     plt.subplot(ROWS, COLS, i + 1)
    #     data = exp_data[i]
    #     np_data = np.asarray(data)
    #     plt.imshow(np_data)
    #     plt.title(LABELS[exp_label[i]])
    # plt.show()

    # train_ds = train_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000)

    val_ds = val_ds
    test_ds = test_ds

    input_shape = exp_data.shape[1:]
    # print("\nINPUT SHAPE: ", input_shape)
    num_labels = len(LABELS)

    #
    # instantiate model (CNN, relu)
    #
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(64, 64),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    #
    # training model (Adam optimizer?)
    #
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCH,
        callbacks=[training_plot]
    )

    #
    # save model
    #
    # model.save('handwriting_model', save_format='tf')
    model.save('handwriting_model_normalized')

    model.evaluate(test_ds, return_dict=True)

    #
    # terminate processes
    #
    gpumonitor.stop()


#           #
# functions #
#           #
def rename_and_move_file():
    MAIN_PATH = 'C:\\Users\\User\\Desktop\\Python\\mnlt\\handwriting\\data\\by_class'
    #SUBDIRS = ['hsf_0', 'hsf_1', 'hsf_2', 'hsf_3', 'hsf_4', 'hsf_5', 'hsf_6', 'hsf_7', 'train_4a']

    # access 4a, 4b, ...
    ALL_LABEL_DIRS = os.listdir(MAIN_PATH)
    for label_folder in ALL_LABEL_DIRS:
        SUB_DIR_PATH = os.path.join(MAIN_PATH, label_folder)
        SUBDIRS = os.listdir(SUB_DIR_PATH)

        # access hsf_0, hsf_1, ...
        for sub in SUBDIRS:
            SRC_SUB_PATH = os.path.join(MAIN_PATH, label_folder, sub)
            DST_SUB_PATH = os.path.join(MAIN_PATH, label_folder)

            ALL_SUB_FILES = os.listdir(SRC_SUB_PATH)

            # access each image
            for file in ALL_SUB_FILES:
                SRC_PATH = os.path.join(SRC_SUB_PATH, file)

                NEW_FILE_NAME = label_folder + "_" + file
                NEW_FILE_PATH = os.path.join(SRC_SUB_PATH, NEW_FILE_NAME)
                DST_PATH = os.path.join(DST_SUB_PATH, NEW_FILE_NAME)

                if os.path.isfile(DST_PATH) or os.path.isfile(NEW_FILE_PATH):
                    print("FILE ALREADY EXISTS")
                else:
                    os.rename(SRC_PATH, NEW_FILE_PATH)
                    shutil.move(NEW_FILE_PATH, DST_PATH)
                    print(DST_PATH)


def confusion():
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
            'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x','y', 'z']
    data_dir = pathlib.Path(DATASET_PATH)

    LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        batch_size=32,
        color_mode='grayscale',
        image_size=(128, 128),
        seed=0,
        validation_split=0.3,
        subset='both'
    )

    model = tf.keras.models.load_model('handwriting_model_omega')
    # confusion matrix
    y_pred = model.predict(val_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(val_ds.map(lambda data, label: label)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    plt.autoscale(tight=True)
    sns.heatmap(confusion_mtx,
                xticklabels=values,
                yticklabels=values,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')

    plt.show()


# export model
class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model
        self.classes = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '41', '42', '43', '44',
            '45', '46', '47', '48', '49', '4a', '4b', '4c', '4d', '4e', '4f', '50', '51', '52',
            '53', '54', '55', '56', '57', '58', '59', '5a', '61', '62', '63', '64', '65', '66',
            '67', '68', '69', '6a', '6b', '6c', '6d', '6e', '6f', '70', '71', '72', '73', '74',
            '75', '76', '77', '78', '79', '7a']
        self.values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
            'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x','y', 'z']

        self.__call__.get_concrete_function(
            data=tf.TensorSpec(shape=[1, 128, 128, 1], dtype=tf.int64)
        )

    @tf.function
    def __call__(self, data):
        if isinstance(data, tf.Tensor):
            result = self.model(data, training=False)
        elif isinstance(data, str):
            data = tf.io.read_file(str(data))
            data = tf.image.decode_image(data)
            data = tf.image.rgb_to_grayscale(data)
            data = data[tf.newaxis, ...]
            result = self.model(data, training=False)
        else:
            raise ValueError("Unsurported data type")
        
        return result


# graph, plot
class TrainingPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        # append latest log
        self.logs.append(logs)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # at least 2 data points
        if len(self.loss) > 1 and len(self.acc) > 1:

            # plot graph
            clear_output(wait=True)
            N = np.arange(0, len(self.loss))

            plt.style.use("seaborn")

            plt.figure(figsize=(16, 8))
            plt.subplot(1, 3, 1)
            plt.plot(N, self.loss, self.val_loss)
            plt.legend(['loss', 'val_loss'])
            plt.ylim([0, max(plt.ylim())])
            plt.xlabel('Epoch')
            plt.ylabel('Loss [CrossEntrophy]')

            plt.subplot(1, 3, 2)
            plt.plot(N, 100*np.array(self.acc), 100*np.array(self.val_acc))
            plt.legend(['accuracy', 'val_accuracy'])
            plt.ylim([0, 100])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy [%]')

            plt.subplot(1, 3, 3)
            plt.plot(gpumonitor.timestamp, gpumonitor.gpustats)
            plt.legend(['GPU Utilization'])
            plt.ylim([0, max(plt.ylim())])
            plt.xlabel(f'Time [{GPUDELAY}s]')
            plt.ylabel('GPU Utilization')

            plt.show()
        

# pickle
def save_data(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_data(path):
    with open(path, 'rb') as enc:
        return pickle.load(enc)


# GPU stats
class GPUMonitor(Thread):
    def __init__(self, delay):
        super(GPUMonitor, self).__init__()
        self.delay = delay
        self.stopped = False
        self.timestamp = []
        self.gpustats = []
        self.start()

    def run(self):
        while not self.stopped:
            t, s = len(self.timestamp)*self.delay + self.delay, GPUtil.getGPUs()[0].memoryUsed
            self.timestamp.append(t)
            self.gpustats.append(s)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

# main
if __name__ == '__main__':
    gpumonitor = GPUMonitor(GPUDELAY)
    training_plot = TrainingPlot()
    main()
    # confusion()

# %%
