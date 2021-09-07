import numpy as np
import cv2
import os

path = '/media/2T/cc/RSICD/RSICD_images'
file_list = os.listdir(path)
img_list = []
for item in file_list:
    img = cv2.imread(path+'/'+item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r,g,b = cv2.split(img)
    r = r/255
    g = g/255
    b = b/255
    img = np.vstack([r.flatten('a'), g.flatten('a'), b.flatten('a')])
    img_list.append(img)

data = np.hstack(img_list)
m = np.mean(data, axis = 1)
std = np.std(data, axis = 1)
print(m)
print(std)