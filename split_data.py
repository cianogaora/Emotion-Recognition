import pandas as pd
import numpy as np
import cv2
import math

path = 'fer2013.csv'
data = pd.read_csv(path)

labels = data['emotion']
images = data['pixels']
image_list = []

for image in images:
    image = image.split()
    image = list(map(int, image))
    image = np.array(image)
    image = np.reshape(image, (math.sqrt(image.shape[0]), math.sqrt(image.shape[0])))
    image_list.append(image)
    
print(image_list[0])

cv2.imshow('sample', image_list[0].astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
