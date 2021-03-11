import glob

import numpy
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

import numpy as np

model = ResNet152V2(weights='imagenet', include_top=False)

list_features_extracted = []

file_list = glob.glob("images/*.jpg")
file_list = sorted(file_list)
for file in file_list:

    print(file)
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = numpy.array(model.predict(x), dtype='float16')
    list_features_extracted .append(features.flatten())


y_pred = KMeans(n_clusters=5, random_state=10).fit_predict(list_features_extracted)

print(y_pred)