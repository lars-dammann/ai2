from os.path import isfile, join
import tensorflow as tf

assert tf.__version__.startswith('2')
tf.compat.v1.disable_v2_behavior()

tf.compat.v1.enable_eager_execution()

from os import listdir
import os
from tensorflow.keras import optimizers
import numpy as np
import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import io
from keras import backend as K
import tensorflow_addons as tfa
import json
from functools import partial
import psutil
import keras
from keras.callbacks import Callback

checkpoint_path = None#"model_RevHuber_init_Adam_lr0.001_RevHub_1_107_addep35"

#ToDo change folder
data_folder = "/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/data_overlap_aligned_01/" #"/home/haicu/anna.lisitsyna/data_overlap_aligned_01/"
log_folder = "/Users/anna.lisitsyna/Documents/Corrosion estimation/runs/" #"/home/haicu/anna.lisitsyna/runs/"
patch_size = (256, 256)
batch_size = 4
delta_par = 1
scale_coeff = 100
resize_coeff = 1

print("Tensorflow version: {}".format(tf.__version__),flush=True)
print("Keras version: {}".format(keras.__version__),flush=True)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def get_volume(hmap, pixsize=(7.406, 7.406, 0.1)):
    h, w = hmap.shape[:2]
    mask = create_circular_mask(h, w)
    masked_img = hmap.copy()
    masked_img[~mask] = 0
    all_hmap = (np.clip(masked_img, None, 0))
    all_hmap = all_hmap.flatten()
    vol = abs(sum(all_hmap)) * pixsize[0] * resize_coeff * pixsize[1] * resize_coeff * pixsize[
        2] * resize_coeff / 10 ** 9
    return vol * 13100


# def relative_mean_error(target, output):
#    #output_flattened = output.numpy().flatten()
#    target_flattened = target.numpy().flatten()
#    mean_target = abs(target_flattened).mean()
#    rel_mae = tf.losses.Mean* 100
#    return rel_mae

rho = 0.05


class loss_with_KLD(tf.losses.Loss):

    def __init__(self, rho):
        super(loss_with_KLD, self).__init__()
        self.rho = rho
        self.kl = tf.losses.KLDivergence()
        self.mse = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        mse = self.mse(y_true, y_pred)
        kl = self.kl(self.rho, y_pred)
        return mse + kl


class relative_MAE(tf.losses.Loss):

    def __init__(self):
        super(relative_MAE, self).__init__()
        self.mae = tf.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        mae_temp = K.abs(y_true - y_pred) / (K.abs(y_true) + 0.0001)
        mae_rel = tf.reduce_mean(mae_temp) * 100
        return mae_rel


def RevHub_loss(labels, predictions, c = 0.1):
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")

    # Make sure shape do match
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # Get absolute error for each pixel in batch
    abs_error = tf.abs(tf.subtract(predictions, labels), name='abs_error')
    c = c * tf.reduce_max(abs_error)
    RevHub_loss = tf.where(abs_error <= c,
                          abs_error,
                          (tf.square(abs_error) + tf.square(c)) / (2 * c))

    loss = tf.reduce_mean(RevHub_loss)

    return loss


def volume_loss_diff(target, output):
    output = return_back(output.numpy(), par=1, mean_val=-0.0059, std_val=0.0112)
    target = return_back(target.numpy(), par=1, mean_val=-0.0059, std_val=0.0112)
    vol_tar = get_volume(target)
    vol_out = get_volume(output)
    vol_loss_diff = tf.convert_to_tensor((abs(vol_tar - vol_out) / vol_tar) * 100)
    return vol_loss_diff


def _parse_tfr_element(element):
    parse_dic = {'prof': tf.io.FixedLenFeature([], tf.string), 'heat': tf.io.FixedLenFeature([], tf.string)}
    example_message = tf.io.parse_single_example(element, parse_dic)
    # get byte string
    prof = tf.io.parse_tensor(example_message['prof'], out_type=tf.float64)
    heat = tf.io.parse_tensor(example_message['heat'], out_type=tf.float64)  # restore 2D array from byte string

    del parse_dic
    del example_message

    return prof, heat


from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model3(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x = layers.Conv2D(32, 3, strides=1, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    x = layers.Conv2D(16, 3, padding='same', kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    outputs_list = []
    filters_list = [128, 64, 32]

    for intermediate_index, filters in enumerate(filters_list):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        upsampled = layers.UpSampling2D(2 ** (len(filters_list) - intermediate_index))(x)
        outputs_list.append(
            layers.Conv2D(num_classes, 1, padding='same', name=f'intermediate_output{intermediate_index}', kernel_initializer=initializer, bias_initializer=bias_initializer)(upsampled))

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.Conv2D(16, 1, activation='relu', padding="same", kernel_initializer=initializer, bias_initializer=bias_initializer)(x)
    # Add a per-pixel classification layer
    outputs_list.append(layers.Conv2D(num_classes, 1, padding="same", name='output', activation='relu', kernel_initializer=initializer, bias_initializer=bias_initializer)(x)) #ToDo add activation function

    # Define the model
    model = tf.keras.Model(inputs, outputs_list)
    return model


def return_back(X, mean_val=0, std_val=1, min_val=-0.03, max_val=0.01, par=0):
    if par == 0:
        X_std = X * (max_val - min_val) + min_val
    if par == 1:
        X_std = X * std_val + mean_val
    return X_std

class MemoryUsageCallback(Callback):
    '''Monitor memory usage on epoch begin and end.'''

    def on_epoch_begin(self,epoch,logs=None):
        print('**Epoch {}**'.format(epoch))
        print('Memory usage on epoch begin: {} GB'.format(psutil.Process(os.getpid()).memory_info().rss/(1024*1024*1024)))
        tf.summary.scalar('memory epoch end', data=psutil.Process(os.getpid()).memory_info().rss/(1024*1024*1024), step=epoch)

    def on_epoch_end(self,epoch,logs=None):
        print('Memory usage on epoch end:   {} GB'.format(psutil.Process(os.getpid()).memory_info().rss/(1024*1024*1024)))

    def on_train_batch_begin(self, batch, logs=None):
        if batch%100==0:
            print('Memory usage on beginning of batch {} :   {} GB'.format(batch,
                psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)))

res = []
for seed_v in [1]:#range(1): #change to 50
    seed_value= seed_v

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    #import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions:
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    #from keras import backend as K
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)



    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()

    # Build model
    inp_size=patch_size

    initializer = tf.keras.initializers.GlorotUniform(seed=seed_value)
    bias_initializer = tf.keras.initializers.Zeros()
    model = get_model3(inp_size, 1)

    if checkpoint_path is not None:
        # Load model:
        start_num = int(checkpoint_path.split('ep')[-1])+107
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={'RevHub_loss':RevHub_loss})
        print("The starting point is epoch ", start_num)    
    else:
        start_num=0
    model.summary()

    opt = 'Adam'
    lr = 0.001
    loss = 'MAE' #"RevHub


    losses = {
        "intermediate_output0": tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
        "intermediate_output1": tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
        "intermediate_output2": tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
        "output": tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) #tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    }

    lossWeights = {"intermediate_output0": 1.0,
                   "intermediate_output1": 1.0,
                   "intermediate_output2": 1.0,
                   "output": 1.0}
    metrics = [tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare()] #[tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare(), volume_loss_diff, relative_mean_error]

    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=metrics, run_eagerly=True)

    logdir = log_folder+r"RevHuber_overlap_{0}_lr{1}_{2}_{3}_init".format(
        opt, lr, loss, seed_value) + datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    tensorboard_callback = tf.compat.v1.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_grads =True)
    tfrecords_pattern_path = data_folder+"dataset_train_*.tfrecords"
    files = tf.io.matching_files(tfrecords_pattern_path)
    print("Train files are: ", files)
    shards = tf.data.Dataset.from_tensor_slices(files)
    #ToDo change back

    # TODO make a function creating a datagen completely, do not duplicated code
    # FIXME removed interlevae!
    # train_data = shards.interleave(partial(tf.data.TFRecordDataset,compression_type='GZIP'))
    #train_data = train_data.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = shards.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_data_size = train_data.reduce(0, lambda x, _: x+1).numpy()
    train_data_num_batches = (train_data_size + batch_size - 1) // batch_size

    train_data = train_data.batch(batch_size)
    train_data = train_data.repeat()
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_data = train_data.as_numpy_iterator()


    tfrecords_pattern_path = data_folder+"dataset_val_*.tfrecords"
    files = tf.io.matching_files(tfrecords_pattern_path)
    #files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)
    # FIXME removed interlevae!
    #val_data = shards.interleave(partial(tf.data.TFRecordDataset,compression_type='GZIP'))
    #val_data = val_data.shuffle(buffer_size=10)
    ## val_data = val_data.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_data = shards.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_data_size = val_data.reduce(0, lambda x, _: x+1).numpy()
    val_data_num_batches = (val_data_size + batch_size - 1) // batch_size

    val_data = val_data.batch(batch_size)
    val_data = val_data.repeat()
    val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_data = val_data.as_numpy_iterator()

    # tfrecords_pattern_path = data_folder+"dataset_test_*.tfrecords"
    # files = tf.io.matching_files(tfrecords_pattern_path)
    # files = tf.random.shuffle(files)
    # shards = tf.data.Dataset.from_tensor_slices(files)
    # test_data_temp = shards.interleave(partial(tf.data.TFRecordDataset,compression_type='GZIP'))
    # test_data_temp = test_data_temp.shuffle(buffer_size=10)
    # test_data_temp = test_data_temp.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # test_data_temp_size = test_data_temp.reduce(0, lambda x, _: x+1).numpy()
    # test_data_temp_num_batches = (test_data_temp_size + batch_size - 1) // batch_size
    #
    # test_data_temp = test_data_temp.batch(batch_size)
    # test_data_temp = test_data_temp.repeat()
    # test_data_temp = test_data_temp.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_data_temp = test_data_temp.as_numpy_iterator()

    mcp_save = ModelCheckpoint(
        "model_adjRevHuber_overlap_orig_{0}_lr{1}_{2}_{3}_{4}_addep".format(
            opt, lr, loss, seed_value, start_num) + "{epoch}")

    mem_check = MemoryUsageCallback()

    history = model.fit(train_data, workers=0, validation_data=val_data, epochs=35,
                        verbose=1,
                        steps_per_epoch=train_data_num_batches,
                        validation_steps = val_data_num_batches,
                        callbacks=[tensorboard_callback, mcp_save, mem_check])#, imsave_callback])


    res_temp = {'seed':seed_value, 'r2_tr':history.history['output_r_square'],
                'r2_val':history.history['val_output_r_square'],
                'loss_tr':history.history['output_loss'],
                'loss_val':history.history['val_output_loss']}

    with open('res_adjRevHuber_overlap_from0_seed_'+str(seed_value), 'w', encoding='utf-8') as file:
        json.dump(res_temp, file)

    res.append(res_temp)



with open('seeds_res_overlap_adjRevHub_from0', 'w', encoding='utf-8') as file:
    for dic in res:
        data = json.dumps(dic)
        file.write(data)
        file.write("\n")

