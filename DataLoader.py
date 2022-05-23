
# %%
import os
import random

import numpy as np
import tensorflow
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from Unet import jacard_coef

random.seed(0)
np.random.seed(0)
import segmentation_models as sm
from osgeo import gdal

# %% 
# Declare static variables

# Pixel-Level class distribution (total sum equals 1.0)
class_distr = np.array([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
 0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052]).astype('float32')

# Band means
bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

cwd = os.getcwd()
dataset_path = os.path.join(cwd, "MARIDA")
print(dataset_path)
scaler = MinMaxScaler()
patch_size = 256

#%% 
# Pixel-level Semantic Segmentation Data Loader  

ROIs = np.genfromtxt(os.path.join(dataset_path, 'splits', 'all_X.txt'), dtype='str')
image_dataset = []  # Loaded Images
mask_dataset = []  # Loaded Output masks
impute_nan = np.tile(bands_mean, (patch_size , patch_size, 1))
for roi in tqdm(ROIs, desc='Load data set to memory'):
    # Construct file and folder name from roi
    roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])  # Get Folder Name
    roi_name = '_'.join(['S2'] + roi.split('_'))  # Get File Name
    roi_file = os.path.join(dataset_path, 'patches', roi_folder, roi_name + '.tif')  # Get File path
    roi_file_cl = os.path.join(dataset_path, 'patches', roi_folder, roi_name + '_cl.tif')

    # get masks
    mask = gdal.Open(roi_file_cl)
    mask = np.copy(mask.ReadAsArray().astype(np.int64))
    mask[mask >= 7] = 7  # ALL Water to Marine Water Class

    mask_dataset.append(mask)

    # get image
    image = gdal.Open(roi_file)
    image = np.copy(image.ReadAsArray())

    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')
    # code for transforming image to remove 0 data
    # mask_cop = np.moveaxis(mask_cop, [0, 1, 2], [2, 0, 1]).astype('int32')
    # new_image = image - mask_cop
    # new_image[new_image < 0] = 0
    image_dataset.append(image)
   # image_dataset.append(new_image)
   
# Load Classification Mask
# print(path)

# %% 
# code to get categorical class masks and split data

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# 1: 'Marine Debris',
# 2: 'Dense Sargassum',
# 3: 'Sparse Sargassum',
# 4: 'Natural Organic Material',
# 5: 'Ship',
# 6: 'Clouds',
# 7: 'Marine Water',
# 8: 'Sediment-Laden Water',
# 9: 'Foam',
# 10: 'Turbid Water',
# 11: 'Shallow Water',
# 12: 'Waves',
# 13: 'Cloud Shadows',
# 14: 'Wakes',
# 15: 'Mixed Water'

n_classes = len(np.unique(mask_dataset))
labels = np.expand_dims(mask_dataset, axis=3)

labels_cat = tensorflow.keras.utils.to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)

# model dims
IMG_CHANNELS = X_train.shape[3]
IMG_WIDTH  = X_train.shape[2]
IMG_HEIGHT = X_train.shape[1]

# %% 
# Model losses

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #
jaccard_loss = sm.losses.JaccardLoss(class_indexes=[1,2,3,4,5,6,7], class_weights=[7, 4, 4, 4, 4, 1, 1])
categorical_ce_loss = sm.losses.CategoricalCELoss(class_indexes=[1,2,3,4,5,6,7], class_weights=[7, 4, 4, 4, 4, 1, 1])
categorical_ce_jaccard_loss = categorical_ce_loss + jaccard_loss

# metrics 
from Unet import jacard_coef

metrics=[jacard_coef, 'accuracy']

# %% 
# get model weights

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train.flatten(order='C'))
y=y_train.flatten(order='C').tolist()
samples=[]
for i in range(7):
    samples.append(y.count(i))

class_weights={}
max_sample=np.max(samples)
print (max_sample)
for i in range (len(samples)):
    class_weights[i]=max_sample/samples[i]
for key, value in class_weights.items():
    print ( key, ' : ', value)


# %%
# train model

from Unet import multi_unet_model


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
model = get_model()
model.compile(optimizer='adam', loss=categorical_ce_loss, metrics=[sm.metrics.IOUScore(class_indexes=[1,2,3,4,5,6,7]), "accuracy"])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint('models/modelCE{epoch:08d}.hdf5', save_freq="epoch")

history1 = model.fit(X_train, y_train, callbacks=[checkpoint],
                    batch_size = 16,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False)

model.save('models/2_standard_unet_10epochs_22May2022weights.hdf5')

#%% 
# Load model

from tensorflow.keras.models import load_model

model = load_model('models/modelCE00000005.hdf5', custom_objects={'categorical_crossentropy_plus_jaccard_loss': categorical_ce_jaccard_loss, 'iou_score':sm.metrics.IOUScore(class_indexes=[1,2,3,4,5,6,7])})

#%% 

# make predictions 
# create n_masks x 256 x 256 predicted mask from the 256 x 256 x n_classes mask
y_test_argmax=np.argmax(y_test, axis=3)

import random

# plots single band of test image.



# # select random image
# test_img_number = random.randint(0, len(X_test))
# test_img = X_test[test_img_number]

#select first image
test_img = gdal.Open("/home/henry/Documents/PyCharm/Plastic_Detect/MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif")
test_img = np.copy(test_img.ReadAsArray())
test_img = np.moveaxis(test_img, [0, 1, 2], [2, 0, 1]).astype('float32')

plt.imshow(test_img[:,:,2])
plt.show()

# ground_truth=y_test_argmax[test_img_number]

# plot mask of test image
ground_truth = gdal.Open("/home/henry/Documents/PyCharm/Plastic_Detect/MARIDA/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0_cl.tif")
ground_truth = np.copy(ground_truth.ReadAsArray().astype(np.int64))
plt.imshow(ground_truth[:,:])
plt.show()

#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = model.predict(test_img_input)

# logits = prediction.reshape((-1, 1))
# target = ground_truth.reshape(-1)
# mask = prediction != -1
# logits = prediction[mask]
# target = ground_truth[mask]

predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
plt.show()
# %%
