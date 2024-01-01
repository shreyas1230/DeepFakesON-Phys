
import numpy as np
import os
import cv2
# # from imageio import imread
# # from skimage.transform import resize
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
# import pandas as pd
# import h5py
# import glob
# import sys
# import scipy.io
# import time 
from IPython import embed
from tqdm import tqdm

root_path = '../../data/Celeb-DF-v2'

def load_test_motion(carpeta):
    # Deep Frames
    X_test = []
    images_names = []
    image_path = []
    y_test = []
    for i in carpeta:
        paths = i[1].split('/')
        paths.insert(1, 'DeepFrames')
        image_path.append(os.path.join(*([root_path] + paths)).split('.mp4')[0] + '.npy')
    # embed()
    # raise Exception
    # print(carpeta)
    print('Read test images')

    for idx, img_path in enumerate(image_path):
        if not os.path.exists(img_path):
            # print(f"{img_path} doesn't exist!")
            continue
        arr = np.load(img_path) # h x w x c x t
        # embed()
        # raise Exception
        assert np.min(arr) >= 0, f"Min value is {np.min(arr)}, less than 0!"
        assert np.max(arr) <= 255, f"Max value is {np.max(arr)}, greater than 255!"
        for i in range(arr.shape[-1]):
            img = cv2.resize(arr[:,:,:,i], (36,36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            y_test.append(int(carpeta[idx][0]))
            # embed()
            # raise Exception
            images_names.append(carpeta[idx][1] + f";frame{i}")
    # embed()
    # raise Exception

    # for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
    #     carpeta= os.path.join(image_path, f)
    #     print(carpeta)
    #     for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
    #         imagenes = os.path.join(carpeta, imagen)
    #         img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
    #         img = img.transpose((-1,0,1))
    #         X_test.append(img)
    #         images_names.append(imagenes)
    return X_test, y_test, images_names


def load_test_attention(carpeta):
    # Raw Frames
    X_test = []
    y_test = []
    images_names = []
    image_path = []
    for i in carpeta:
        paths = i[1].split('/')
        paths.insert(1, 'RawFrames')
        image_path.append(os.path.join(*([root_path] + paths)).split('.mp4')[0] + '.npy')
    print('Read test images')

    for idx, img_path in enumerate(image_path):
        if not os.path.exists(img_path):
            # print(f"{img_path} doesn't exist!")
            continue
        arr = np.load(img_path) # h x w x c x t
        # embed()
        # raise Exception
        assert np.min(arr) >= 0, f"Min value is {np.min(arr)}, less than 0!"
        assert np.max(arr) <= 255, f"Max value is {np.max(arr)}, greater than 255!"
        for i in range(arr.shape[-1]):
            img = cv2.resize(arr[:,:,:,i], (36,36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            y_test.append(int(carpeta[idx][0]))
            images_names.append(carpeta[idx][1] + f";frame{i}")


    # for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
    #     carpeta= os.path.join(image_path, f)
    #     print(carpeta)
    #     for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
    #         imagenes = os.path.join(carpeta, imagen)
    #         img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
    #         img = img.transpose((-1,0,1))
    #         X_test.append(img)
    #         images_names.append(imagenes)
    return X_test, y_test, images_names

# np.set_printoptions(threshold=np.inf)
# data = []
batch_size = 512
model = load_model('../pretrained models/DeepFakesON-Phys_CelebDF_V2.h5')

print(model.summary())
# embed()
# input("Press Enter to continue...")

# image_path = '../../data/Celeb-DF-v2'

with open(os.path.join(root_path, "List_of_testing_videos.txt"), "r") as f:
    testing_vid_lst = f.readlines()

testing_vid_lst = [i.split(' ') for i in testing_vid_lst]
for i in range(len(testing_vid_lst)):
    testing_vid_lst[i][1] = testing_vid_lst[i][1].strip()


# carpeta_deep= os.path.join(image_path, "DeepFrames")
# carpeta_raw= os.path.join(image_path, "RawFrames")

test_data, test_labels, images_names = load_test_motion(testing_vid_lst)
test_data2, test_labels2, images_names2 = load_test_attention(testing_vid_lst)

assert len(images_names) == len(images_names2), f"Length of two images_names not same; images_names={len(images_names)}; images_names2={len(images_names2)}"
for i in range(len(images_names)):
    assert images_names[i] == images_names2[i]
    assert test_labels[i] == test_labels2[i]

test_data = np.array(test_data, copy=False, dtype=np.float32)
test_data2 = np.array(test_data2, copy=False, dtype=np.float32)
test_labels = np.array(test_labels)

predictions = []

for i in tqdm(range(0, len(test_data), batch_size)):
    # print(f"{i}/{len(test_data)}")
    ex = i+batch_size if i+batch_size < len(test_data) else len(test_data)
    predictions.extend(model([test_data[i:ex], test_data2[i:ex]]))
predictions = np.array([i.numpy()[0] for i in predictions])
preds = (predictions > 0.5).astype(int)
# predictions = model.predict([test_data[:2], test_data2[:2]], batch_size=batch_size, verbose=1)

# embed()
# raise Exception

bufsize = 1
nombre_fichero_scores = '../../data/deepfake_scores.txt'
fichero_scores = open(nombre_fichero_scores,'w',buffering=bufsize)
fichero_scores.write("img;score\n")
for i in tqdm(range(len(predictions))):
    fichero_scores.write("%s" % images_names[i]) #fichero
    # if float(predictions[i])<0:
        # predictions[i]='0'
    # elif float(predictions[i])>1:
        # predictions[i]='1'
    fichero_scores.write(";%s\n" % predictions[i]) #scores predichas
    fichero_scores.write(";%s\n" % preds[i]) #scores predichas
    fichero_scores.write(";%s\n" % test_labels[i]) #scores predichas


print("RESULTS")
print("---------------------")
print("Accuracy:", np.sum(test_labels == preds) / len(test_labels))
print("Min prediction value:", np.min(predictions), "Max prediction value:", np.max(predictions))