import PIL.Image as Image
import os
import numpy as np
import cv2
# root = '/mnt/data-hdd/wgk/mmsegmentation/data/trans/images/111/'
# images = os.listdir(root)
# for image in images:
#     img = Image.open(root + image)
#     h, w = img.size[0], img.size[1]
#     img1 = Image.open('/mnt/data-hdd/wgk/Transfer-Learning-Library/data/robotic_intubation/source/transimage/' + image)
#     img1 = img1.resize((h, w))
#     img1.save('/mnt/data-hdd/wgk/mmsegmentation/data/trans/images/train/' + image)


import cv2
import os
import numpy as np
img_root = '/mnt/data-hdd/wgk/mmsegmentation/data/robotic_intubation/annotations/test/'
imgs = os.listdir(img_root)
number = 1
for img in imgs:
    img = img_root + img
    image = cv2.imread(img)
    print(number)
    gray_img = np.ones([image.shape[0], image.shape[1]])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][1] == 2:
                gray_img[i][j] = 2
            elif image[i][j][1] == 3:
                gray_img[i][j] = 3
            elif image[i][j][1] == 4:
                gray_img[i][j] = 4


    cv2.imwrite(img.replace('test', '111'), gray_img)
    number = number + 1
    