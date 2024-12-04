import os
import cv2
import numpy as np
import segmentation_models as sm
import random

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import to_categorical
from unet_model import get_unet_model, jaccard_coef


scaler = MinMaxScaler()
dataset_dir = 'segmentation_dataset/'
patch_size = 256

image_set = []
for path, subdirs, files in os.walk(dataset_dir):
    dir_name = path.split(os.path.sep)[-1]
    if dir_name == 'images':
        images = os.listdir(path)
        for i, img_name in enumerate(images):
            if img_name.endswith(".jpg"):
                img = cv2.imread(path + "/" + img_name, 1)
                img = Image.fromarray(img)
                img = img.crop((0, 0, (img.shape[1] // patch_size) * patch_size,
                                (img.shape[0] // patch_size) * patch_size))
                img = np.array(img)
                patches_img = patchify(img, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, :, :]
                        single_patch_img = scaler.fit_transform(
                            single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0]
                        image_set.append(single_patch_img)

mask_set = []
for path, subdirs, files in os.walk(dataset_dir):
    dir_name = path.split(os.path.sep)[-1]
    if dir_name == 'masks':
        masks = os.listdir(path)
        for i, msk_name in enumerate(masks):
            if msk_name.endswith(".png"):
                msk = cv2.imread(path + "/" + msk_name, 1)
                msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
                msk = Image.fromarray(msk)
                msk = msk.crop((0, 0, (msk.shape[1] // patch_size) * patch_size,
                                (msk.shape[0] // patch_size) * patch_size))
                msk = np.array(msk)
                patches_msk = patchify(msk, (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_msk.shape[0]):
                    for j in range(patches_msk.shape[1]):
                        one_patch_mask = patches_msk[i, j, :, :]
                        one_patch_mask = one_patch_mask[0]
                        mask_set.append(one_patch_mask)

image_set = np.array(image_set)
mask_set = np.array(mask_set)

image_number = random.randint(0, len(image_set))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_set[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_set[image_number], (patch_size, patch_size, 3)))
plt.show()

#######################################
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i + 2], 16) for i in (0, 2, 4)))

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i + 2], 16) for i in (0, 2, 4)))

Road = '#6EC1E4'.lstrip('#')
Road = np.array(tuple(int(Road[i:i + 2], 16) for i in (0, 2, 4)))

Vegetation = 'FEDD3A'.lstrip('#')
Vegetation = np.array(tuple(int(Vegetation[i:i + 2], 16) for i in (0, 2, 4)))

Water = 'E2A929'.lstrip('#')
Water = np.array(tuple(int(Water[i:i + 2], 16) for i in (0, 2, 4)))

Unlabeled = '#9B9B9B'.lstrip('#')
Unlabeled = np.array(tuple(int(Unlabeled[i:i + 2], 16) for i in (0, 2, 4)))

label = one_patch_mask


def convert_rgb_to_2d_label(label):
    segm_label = np.zeros(label.shape, dtype=np.uint8)
    segm_label[np.all(label == Building, axis=-1)] = 0
    segm_label[np.all(label == Land, axis=-1)] = 1
    segm_label[np.all(label == Road, axis=-1)] = 2
    segm_label[np.all(label == Vegetation, axis=-1)] = 3
    segm_label[np.all(label == Water, axis=-1)] = 4
    segm_label[np.all(label == Unlabeled, axis=-1)] = 5
    segm_label = segm_label[:, :, 0]
    return segm_label


labels = []
for i in range(mask_set.shape[0]):
    label = convert_rgb_to_2d_label(mask_set[i])
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Labels from the dataset: ", np.unique(labels))

image_number = random.randint(0, len(image_set))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_set[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:, :, 0])
plt.show()

#######################################
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_set, labels_cat, test_size=0.20, random_state=42)


#######################################
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)


metrics = ['accuracy', jaccard_coef]
model = get_unet_model(n_classes=n_classes, image_height=X_train.shape[1],
                       image_width=X_train.shape[2], image_channels=X_train.shape[3])
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
model.summary()

our_history = model.fit(X_train, y_train,
                        batch_size=16,
                        verbose=1,
                        epochs=100,
                        validation_data=(X_test, y_test),
                        shuffle=False)

hstr = our_history
loss = hstr.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, hstr.history['val_loss'], 'r', label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(epochs, hstr.history['jacard_coef'], 'y', label='Training IoU')
plt.plot(epochs, hstr.history['val_jacard_coef'], 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

model.save('models/satellite_unet.hdf5')


#######################################
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#######################################
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Test image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Test label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on the test image')
plt.imshow(predicted_img)
plt.show()
