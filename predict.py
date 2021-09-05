from tensorflow.keras.models import load_model
import os
from wordtoletter import getLetters
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
model = load_model("Model.h5")


def getPrediction(imagepath):
    ans = ""
    getLetters(imagepath)
    dir = 'subimage'
    for f in os.listdir(dir):
        img = image.load_img("subimage/" + f, target_size=(28, 28))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = img_tensor[:, :, :, :1]
        img_tensor /= 255
        plt.imshow(img_tensor.reshape(28, 28))
        prediction = model.predict(img_tensor)
        alphabets = "abcdefghijklmnopqrstuvwxyz"
        list1 = []
        [list1.append(i) for i in range(26)]
        list2 = []
        [list2.append(i) for i in alphabets]
        dic = dict(zip(list1, list2))
        ans += dic[np.argmax(prediction)]
        os.remove(os.path.join(dir, f))
    print(ans)


getPrediction("two.png")
