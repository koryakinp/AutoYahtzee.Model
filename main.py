import uuid
import cv2
import os
import glob

from utils import build_empty_kernels
from utils import build_dice_kernels
from utils import prepare_image_data
from utils import get_dice_images
from utils import predict

empty_kernels = build_empty_kernels()
dice_kernels, dice_sides = build_dice_kernels()
images = glob.glob(os.path.join('data','full', '*.jpg'))

total = len(images)
current = 1

for image in images:
    
    print(image)
    
    image = prepare_image_data(image)
    dices = get_dice_images(image, empty_kernels)
    labels = predict(dices, dice_sides, dice_kernels)
    
    for i in range(len(dices)):
        dices[i][dices[i] == -1] = 0
        dices[i][dices[i] == 1] = 255
        label = str(labels[i])
        filename = str(uuid.uuid4()) + '.jpg'
        path = os.path.join('data', label, filename)
        cv2.imwrite(path, dices[i])
        
    print(str(current) + "/" + str(total))
    current += 1