import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw(dataset, names):
    for i, d in enumerate(dataset):
        img, boxes = d
        img = img.numpy().transpose(1, 2, 0)
        img = 255 * img  # Now scale by 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for box in boxes['boxes']:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, p1, p2, color=(0, 255, 0), thickness=1)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'out/dataset/{i}.png', img)