import os

#import cv2
import tensorflow
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import numpy as np

from Unet import jacard_coef

random.seed(0)
np.random.seed(0)
from osgeo import gdal
import segmentation_models as sm
# Pixel-Level class distribution (total sum equals 1.0)
# class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
#  0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################
dataset_path = r"C:\Users\3henr\Documents\GEE\MARIDA"
print(dataset_path)
scaler = MinMaxScaler()
patch_size = 256


ROIs = np.genfromtxt(os.path.join(dataset_path, 'splits', 'all_X.txt'), dtype='str')
image_dataset = []  # Loaded Images
mask_dataset = []  # Loaded Output masks

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
    # mask[mask == 15] = 7  # Mixed Water to Marine Water Class
    # mask[mask == 14] = 7  # Wakes to Marine Water Class
    # mask[mask == 13] = 7  # Cloud Shadows to Marine Water Class
    # mask[mask == 12] = 7  # Waves to Marine Water Class

    # reduce every value by 1
    # mask = np.copy(mask - 1)
    mask_dataset.append(mask)

    # get image
    image = gdal.Open(roi_file)
    image = np.copy(image.ReadAsArray())

    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]).astype('float32')
    image_dataset.append(image)
    # FOR CROPPING AND RESIZING - duplicate previous commented code if needed

    #Extract patches from each image - if image bigger than 256x256
    # print("Now patchifying image:", dataset_path + "/" + roi_name)
    # patches_img = patchify(image, (11, patch_size, patch_size),
    #                        step=patch_size)  # Step=256 for 256 patches means no overlap
    #
    # for i in range(patches_img.shape[0]):
    #     for j in range(patches_img.shape[1]):
    #         single_patch_img = patches_img[i, j, :, :]
    #
    #         # Use minmaxscaler instead of just dividing by 255.
    #         single_patch_img = scaler.fit_transform(
    #             single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
    #             single_patch_img.shape)
    #
    #         # single_patch_img = (single_patch_img.astype('float32')) / 255.
    #         single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
    #         image_dataset.append(single_patch_img)

# Load Classification Mask
# print(path)
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
print("done")

#weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #

IMG_CHANNELS = X_train.shape[3]
IMG_WIDTH  = X_train.shape[2]
IMG_HEIGHT = X_train.shape[1]



from Unet import multi_unet_model, jacard_coef

metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
#model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()


history1 = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    shuffle=False)

model.save('models/2_standard_unet_1epochs_16May2022.hdf5')

from keras.models import load_model
model = load_model('models/2_standard_unet_1epochs_16May2022.hdf5', custom_objects={'dice_loss_plus_1focal_loss': total_loss, 'jacard_coef':jacard_coef})

y_test_argmax=np.argmax(y_test, axis=3)
import random
test_img_number =  0 #random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img)
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
plt.show()