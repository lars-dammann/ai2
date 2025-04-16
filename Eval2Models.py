from os.path import isfile, join
import tensorflow as tf

assert tf.__version__.startswith('2')
from os import listdir
import numpy as np
import segmentation_models as sm
import cv2

sm.set_framework('tf.keras')
sm.framework()
from PIL import Image
import tensorflow as tf
from scipy import ndimage
import pandas as pd

import skimage
from skimage.measure import label
import copy
patch_size = (256, 256)
from keras import backend as K

#model1_path = "model_adjRevHuber_init_Adam_lr0.001_RevHub_1_ep27"
#model2_path = "model_init_full_Adam_lr0.001_MAE_4_ep32"
model1_path = "model_RevHuber_nonoverlap_orig_Adam_lr0.001_RevHub_1_35_addep34"#"model_adjRevHuber_init_Adam_lr0.001_RevHub_1_ep35"
model2_path = "model_RevHuber_init_Adam_lr0.001_RevHub_1_ep37"

#model_path = "/Users/anna.lisitsyna/Downloads/scripts/Current scripts/model_init_full_Adam_lr0.001_MAE_1_ep36"



from tensorflow.keras import layers

def RevHub_loss(labels, predictions, c = 0.1):
    if labels is None:
        raise ValueError("labels must not be None.")
    if predictions is None:
        raise ValueError("predictions must not be None.")

    # Make sure shape do match
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # Get absolute error for each pixel in batch
    abs_error = tf.abs(tf.subtract(predictions, labels), name='abs_error')
    RevHub_loss = tf.where(abs_error <= c,
                          abs_error,
                          (tf.square(abs_error) + tf.square(c)) / (2 * c))

    loss = tf.reduce_sum(RevHub_loss)

    return loss

def get_model3(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###
    # Entry block
    x = layers.Conv2D(32, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    outputs_list = []
    filters_list = [128, 64, 32]

    for intermediate_index, filters in enumerate(filters_list):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        upsampled = layers.UpSampling2D(2 ** (len(filters_list) - intermediate_index))(x)
        outputs_list.append(
            layers.Conv2D(num_classes, 1, padding='same', name=f'intermediate_output{intermediate_index}')(upsampled))

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.Conv2D(16, 1, activation='relu', padding="same")(x)
    # Add a per-pixel classification layer
    outputs_list.append(layers.Conv2D(num_classes, 1, padding="same", name='output', activation='relu')(x)) #ToDo add activation function

    # Define the model
    model = tf.keras.Model(inputs, outputs_list)
    return model


def return_back(X, mean_val=0, std_val=1, min_val=-0.424, max_val=0.089, par=0):
    if par==0:
        X_std = X*(max_val - min_val)+min_val
    if par==1:
        X_std = X*std_val + mean_val
    return X_std

resize_coeff = 1

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

    def __init__(self, **kwargs):
        super(relative_MAE, self).__init__()
        self.mae = tf.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

    def call(self, y_true, y_pred):
        mae_temp = K.abs(y_true-y_pred)/(K.abs(y_true)+0.0001)
        mae_rel = tf.reduce_mean(mae_temp)*100
        return mae_rel

def get_volume(hmap, pixsize=(7.406, 7.406, 0.1)):
    h, w = hmap.shape[:2]
    mask = create_circular_mask(h, w)
    masked_img = hmap.copy()
    masked_img[~mask] = 0
    all_hmap = (np.clip(masked_img, None, 0))
    all_hmap = all_hmap.flatten()
    vol = abs(sum(all_hmap)) * pixsize[0] * resize_coeff * pixsize[1] * resize_coeff * pixsize[2] * resize_coeff / 10 ** 9
    return vol*13100

def relative_mean_error(target, output):
    #rel_mae = (K.abs(output - target) / K.mean(K.abs(target)) )* 100
    output_flattened = output.numpy().flatten()
    target_flattened = target.numpy().flatten()
    rel_mae = abs(output_flattened-target_flattened).mean()/abs(target_flattened).mean()
    return rel_mae

def volume_loss_diff(target, output):
    output = return_back(output.numpy())
    target = return_back(target.numpy())
    vol_tar = get_volume(target)
    vol_out = get_volume(output)
    vol_loss_diff = tf.convert_to_tensor((abs(vol_tar-vol_out)/vol_tar)*100)
    return vol_loss_diff

# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

# Build model
inp_size=patch_size

#model = get_model3(inp_size, 1)
#model.summary()
import tensorflow_addons as tfa

opt = 'Adam'
lr = 0.01
loss = 'Huber'

losses = {
    "intermediate_output0": relative_MAE(),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
    "intermediate_output1": relative_MAE(),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
    "intermediate_output2": relative_MAE(),#tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE, delta=delta_par),
    "output": relative_MAE() #tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
}

lossWeights = {"intermediate_output0": 1.0,
               "intermediate_output1": 1.0,
               "intermediate_output2": 1.0,
               "output": 1.0}

def new_rsq(target, output):
    rsq = tfa.metrics.r_square.RSquare()
    return rsq(return_back(target.numpy(), par=1, mean_val=-0.0059, std_val=0.0112), return_back(output.numpy(), par=1, mean_val=-0.0059, std_val=0.0112))


metrics = [tf.keras.metrics.MeanAbsoluteError(), tfa.metrics.r_square.RSquare(), new_rsq]

#model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=metrics, run_eagerly=True)
model1 = tf.keras.models.load_model(model1_path, custom_objects={'new_rsq':new_rsq, 'relative_MAE': relative_MAE(), 'RevHub_loss':RevHub_loss})
model2 = tf.keras.models.load_model(model2_path, custom_objects={'new_rsq':new_rsq, 'relative_MAE': relative_MAE(), 'RevHub_loss':RevHub_loss})

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mask(img, c, rad=670):
    mean_val = np.array([el for el in img.flatten() if el >= 0 and el < 0.1]).mean()
    if len(img.shape)==3:
        mean_val = np.array([el for el in img.reshape(-1, img.shape[-1]) ]).mean(0)
    mask = create_circular_mask(img.shape[0], img.shape[1], center=c, radius=rad)
    new_img = copy.deepcopy(img)
    new_img[~mask] = mean_val
    new_img = new_img[c[1]-rad:c[1]+rad, c[0]-rad:c[0]+rad]
    pad = ((98, 98), (98, 98))
    if len(img.shape)==3 and img.shape[2]==3:
        #pad = ((96, 96), (96, 96), (0, 0))
        #new_img = np.pad(new_img, mode='constant', pad_width=pad, constant_values=mean_val)
        r_, g_, b_ = new_img[:, :, 0], new_img[:, :, 1], new_img[:, :, 2]
        rb = np.pad(array=r_, pad_width=pad, mode='constant', constant_values=mean_val[0])
        gb = np.pad(array=g_, pad_width=pad, mode='constant', constant_values=mean_val[1])
        bb = np.pad(array=b_, pad_width=pad, mode='constant', constant_values=mean_val[2])
        new_img = np.dstack(tup=(rb, gb, bb))
    elif len(img.shape)==3 and img.shape[2]==1:
        new_img = np.pad(new_img[:, :, 0], mode='constant', pad_width=pad, constant_values=mean_val)
        new_img = np.expand_dims(new_img, axis=2)
    else:
        new_img = np.pad(new_img, mode='constant', pad_width=pad, constant_values=mean_val)
    return new_img

def interp(data, method=None, filename=None, resize_coeff=None, pad=((0, 0), (0, 0))):
    try:
        mask = np.isnan(data)
        if method is None:
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            if resize_coeff is not None:
                data = np.pad(data, mode='constant', pad_width=pad, constant_values=0)
                data = ndimage.zoom(data, 1.0 / resize_coeff)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!The problem is {e}")
        print(f"The type of data is {type(data)}, and data itself is {data} in file {filename}")
    return np.expand_dims(data, axis=2) # data


def resize_profilometer(filepath, resize_coeff, greysc=False):
    pic_profilometer_temp = Image.fromarray(cv2.imread(filepath)[:1664, :1792])
    width_temp, height_temp = pic_profilometer_temp.size
    pic_profilometer_temp= pic_profilometer_temp.resize(
        (int(width_temp / resize_coeff), int(height_temp / resize_coeff)))
    pic_profilometer_temp = np.array(pic_profilometer_temp)
    return pic_profilometer_temp / 255


def norm_01(X, mean_val=0, std_val=1, min_val=-0.03, max_val=0.01, par=0):
    if par==0:
        X_std = (X - min_val) / (max_val - min_val)
    if par==1:
        X_std = (X - mean_val) / std_val
    return X_std

def _parse_tfr_element(element):
  parse_dic = {'prof': tf.io.FixedLenFeature([], tf.string),  'heat': tf.io.FixedLenFeature([], tf.string) }
  example_message = tf.io.parse_single_example(element, parse_dic)
  prof = example_message['prof']
  heat = example_message['heat']
  # get byte string
  prof = tf.io.parse_tensor(prof, out_type=tf.float64)
  heat = tf.io.parse_tensor(heat, out_type=tf.float64)# restore 2D array from byte string
  return prof, heat

def return_back(X, mean_val=0, std_val=1, min_val=-0.03, max_val=0.01, par=0):
    if par==0:
        X_std = X*(max_val - min_val)+min_val
    if par==1:
        X_std = X*std_val + mean_val
    return X_std

def patches(img, patch_size=patch_size, step=128):
    res_patches = []
    x_tops = []
    y_tops = []
    if step is None:
        for i in range(1, img.shape[0] // patch_size[0] * img.shape[1] // patch_size[1]-1):
            x_num = i // (img.shape[0] // patch_size[0])
            y_num = i % (img.shape[0] // patch_size[0])
            print("For the patch {0} num_x is {1} and num_y is {2}".format(i, x_num, y_num))
            patch_temp = img[
                         patch_size[1] * y_num:patch_size[1] * y_num + patch_size[1],
                         patch_size[0] * x_num:patch_size[0] * x_num + patch_size[0]]
            if patch_temp.shape[0] == patch_size[0] and patch_temp.shape[1] == patch_size[1]:
                res_patches.append(patch_temp)
    else:
        x_top = 0
        y_top = 0
        i =0
        while y_top <= img.shape[1] - patch_size[1]:
            while x_top<= img.shape[0]-patch_size[0]:
                print("For the patch {0} num_x is {1} and num_y is {2}".format(i, x_top, y_top))
                patch_temp = img[
                             y_top:y_top + patch_size[1],
                             x_top:x_top + patch_size[0]]
                if patch_temp.shape[0] == patch_size[0] and patch_temp.shape[1] == patch_size[1]:
                    res_patches.append(patch_temp)
                x_tops.append(x_top)
                y_tops.append(y_top)
                i=i+1
                x_top = x_top+step
            x_top = 0
            y_top = y_top + step
    print("I am done getting patches")
    return res_patches#, x_tops, y_tops

def patches_overlap_reconstruct(preds, orig_heat, patch_size=(256, 256), step=128):
    nums_matrix = np.full((orig_heat.shape[0], orig_heat.shape[1]), 0)
    vals_matrix = np.full((orig_heat.shape[0], orig_heat.shape[1]), 0.0)
    for i in range(len(preds)):
        x_top = step*(i % ((orig_heat.shape[1]-patch_size[1]) // step+1))
        y_top = step*(i // ((orig_heat.shape[0]-patch_size[0])// step+1))
        nums_matrix[y_top:y_top + patch_size[1],
                             x_top:x_top + patch_size[0]] += 1
        vals_matrix[y_top:y_top + patch_size[1], x_top:x_top + patch_size[0]] += preds[i][:, :, 0]
    vals_matrix = vals_matrix/np.clip(nums_matrix, a_min=1, a_max=len(preds))
    return vals_matrix

import matplotlib.pyplot as plt
c1=  (1010, 855)
c2 = (1010, 855)
patch_size = (256, 256)

prof_filename = '/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/20220719/profilometer/before/4-6-6/2.png'
heat_filename = '/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/20220719/rawdata/after/4-6-6/2.csv'

orig_img = mask(resize_profilometer(prof_filename,resize_coeff=1), c=c1)
orig_heat = mask(interp(pd.read_csv(filepath_or_buffer=heat_filename,skiprows=22,engine='python').to_numpy()[:1664, :1792], filename=heat_filename,resize_coeff=1), c=c2)
p_heat = np.array(patches(orig_heat))
p_prof = np.array(patches(orig_img))

norm_heat = norm_01(orig_heat)

pred1 = model1.predict(p_prof)[3]#return_back(model.predict(p_prof)[3])
pred2 = model2.predict(p_prof)[3]#return_back(model.predict(p_prof)[3])
pred1_img = patches_overlap_reconstruct(pred1, orig_heat) ##ToDO reconstruct image from patches
pred2_img = patches_overlap_reconstruct(pred2, orig_heat)
#pred_img = return_back(pred_img)
#vol_loss1 = get_volume(pred1_img)
#vol_loss2 = get_volume(pred2_img)
def plot_evaluation(prof, heat, pred1, pred2,  filename_temp):
    vol_loss1 = get_volume(return_back(pred1))
    vol_loss2 = get_volume(return_back(pred2))
    vol_loss_gt = get_volume(return_back(heat))
    fig, axes = plt.subplots(nrows=1, ncols=3)#4)
    fig.suptitle(
        "Prediction for {0} \nVolume loss (predicted): {1} \nVolume loss (on GT): {2}".format(filename_temp,
                                                                                                      vol_loss1,
                                                                                                      vol_loss_gt))
    #fig.suptitle("Prediction for {0} \nVolume loss (predicted): {1} and {3} \nVolume loss (on GT): {2}".format(filename_temp, vol_loss1, vol_loss_gt, vol_loss2))
    #fig.suptitle(filename_temp)
    axes[0].set_title("Original input", fontdict={'fontsize': 10,
                                                  'fontweight': 1,
                                                  'verticalalignment': 'baseline',
                                                  'horizontalalignment': 'center'}, loc='center')
    axes[0].imshow(prof)
    im = axes[1].imshow(heat, vmin=0, vmax=1, cmap='plasma')
    axes[1].set_title('Normalized heatmap (GT)', fontdict={'fontsize': 10,
                                                         'fontweight': 1,
                                                         'verticalalignment': 'baseline',
                                                         'horizontalalignment': 'center'}, loc='center')
    axes[2].imshow(pred1, vmin=0, vmax=1, cmap='plasma')
    axes[2].set_title('Predicted heatmap (Pred1)\n for Reverse Huber\nVolume loss (predicted): {:.4f}'.format(vol_loss1), fontdict={'fontsize': 10,
                                                            'fontweight': 1,
                                                            'verticalalignment': 'baseline',
                                                            'horizontalalignment': 'center'}, loc='center')

    #ToDo put it back
    #axes[3].imshow(pred2, vmin=0, vmax=1, cmap='plasma')
    #axes[3].set_title('Predicted heatmap (Pred2)\n for RevHuber with overlap\nVolume loss (predicted): {:.4f}'.format(vol_loss2),
    #                  fontdict={'fontsize': 10,
    #                            'fontweight': 1,
    #                            'verticalalignment': 'baseline',
    #                            'horizontalalignment': 'center'}, loc='center')


    #axes[3].set_title('Diff (Pred-GT)', fontdict={'fontsize': 10,
    #                                              'fontweight': 1,
    #                                              'verticalalignment': 'baseline',
    #                                              'horizontalalignment': 'center'}, loc='center')
    #axes[3].imshow(pred_img - orig_heat, vmin=-0.25, vmax=0.25,
    #               cmap='gray')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)


#fig, axes = plt.subplots(nrows=1, ncols=3)
#fig.suptitle("Prediction for {0} \nVolume loss (predicted): {1}".format(prof_filename, vol_loss))
#axes[0].set_title("Original input", fontdict={'fontsize': 10,
#                                              'fontweight': 1,
#                                              'verticalalignment': 'baseline',
#                                              'horizontalalignment': 'center'}, loc='center')
#axes[0].imshow(orig_img)
#im = axes[1].imshow(norm_heat, vmin=-0.25, vmax=0.25, cmap='plasma')
#axes[1].set_title('Original heatmap (GT)', fontdict={'fontsize': 10,
 #                                                    'fontweight': 1,
 #                                                    'verticalalignment': 'baseline',
 #                                                    'horizontalalignment': 'center'}, loc='center')
#axes[2].imshow(pred_img, vmin=-0.25, vmax=0.25, cmap='plasma')
#axes[2].set_title('Predicted heatmap (Pred)', fontdict={'fontsize': 10,
 #                                                       'fontweight': 1,
 #                                                       'verticalalignment': 'baseline',
 #                                                       'horizontalalignment': 'center'}, loc='center')
#axes[3].set_title('Diff (Pred-GT)', fontdict={'fontsize': 10,
#                                              'fontweight': 1,
#                                              'verticalalignment': 'baseline',
#                                              'horizontalalignment': 'center'}, loc='center')
#axes[3].imshow(pred_img - orig_heat, vmin=-0.25, vmax=0.25,
#               cmap='gray')
#fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

batch_size=4

tfrecords_pattern_path = "/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/data_aligned_01/dataset_train_*.tfrecords"
files = tf.io.matching_files(tfrecords_pattern_path)
shards = tf.data.Dataset.from_tensor_slices(files)
train_data = shards.interleave(tf.data.TFRecordDataset)
train_data = train_data.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_data_size = train_data.reduce(0, lambda x, _: x+1).numpy()
train_data_num_batches = (train_data_size + batch_size - 1) // batch_size

train_data = train_data.batch(batch_size)
train_data = train_data.repeat()
train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_data = train_data.as_numpy_iterator()


tfrecords_pattern_path = "/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/data_aligned_01/dataset_val_*.tfrecords"
files = tf.io.matching_files(tfrecords_pattern_path)
#files = tf.random.shuffle(files)
shards = tf.data.Dataset.from_tensor_slices(files)
val_data = shards.interleave(tf.data.TFRecordDataset)
#val_data = val_data.shuffle(buffer_size=10)
val_data = val_data.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_data_size = val_data.reduce(0, lambda x, _: x+1).numpy()
val_data_num_batches = (val_data_size + batch_size - 1) // batch_size

val_data = val_data.batch(batch_size)
val_data = val_data.repeat()
val_data = val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_data = val_data.as_numpy_iterator()

tfrecords_pattern_path = "/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/data_aligned_01/dataset_test_*.tfrecords"
files = tf.io.matching_files(tfrecords_pattern_path)
files = tf.random.shuffle(files)
shards = tf.data.Dataset.from_tensor_slices(files)
test_data_temp = shards.interleave(tf.data.TFRecordDataset)
test_data_temp = test_data_temp.shuffle(buffer_size=10)
test_data_temp = test_data_temp.map(map_func=_parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_data_temp_size = test_data_temp.reduce(0, lambda x, _: x+1).numpy()
test_data_temp_num_batches = (test_data_temp_size + batch_size - 1) // batch_size

test_data_temp = test_data_temp.batch(batch_size)
test_data_temp = test_data_temp.repeat()
test_data_temp = test_data_temp.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_data_temp = test_data_temp.as_numpy_iterator()

print("Done with image")



