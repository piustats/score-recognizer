import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import PIL
from matplotlib import pyplot as plt

from dataset.transforms import ExIfTransposeImg
from ocr.unproject.unproject_text import unproject
from visualizer.objects import filter, show, draw_boxes
import pytesseract



def find_score(img, pred, player):
    if pred[player] != None:
        box = pred[player][0][2]
        crop = img.crop(box)
        return crop
    else:
        return None


def preprocess_score_crop(img):
    img = pil2cv(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(inputImage, 200, 255, cv2.THRESH_BINARY_INV)
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1001, -100)
    filtered = 255 - filtered
    return filtered
    # cv2.imwrite('foo.png', filtered, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# cv2.imwrite(output, filtered, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def get_prediction(img, threshold, model, names, device):
    pred = model([img])
    pred_class_ids = [names[i]['id'] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [(int(i[0]), int(i[1]), int(i[2]), int(i[3])) for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    return filter(pred_class_ids, pred_boxes, pred_score, threshold)


def read_img(img_path, device):
    img = Image.open(img_path)
    transform_exif = T.Compose([
        ExIfTransposeImg()
    ])
    transform_tensor = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])
    img_pil = transform_exif(img)
    img_tensor = transform_tensor(img_pil)
    img_tensor = img_tensor.to(device)
    return img_pil, img_tensor


def ocr(img, dpi):
    config = f'--dpi {dpi} --tessdata-dir ../ocr/Karnivore-digits-model-rotate-skew/tesstrain/data --psm 4 -l Karnivore -c load_system_dawg=0 -c load_freq_dawg=0'
    ocred = pytesseract.image_to_string(img, config=config)
    return ocred.rstrip('\n')


def get_grade_divisions(img):
    h, w = img.shape
    div_count = 8
    l = h / 14.82
    s = (h - 8 * l) / 7.0
    dpi = int((l * l) / 8.33)

    print(f'Estimated dpi: {dpi}')
    print(f'Estimated line height: {l:.2f}')
    print(f'Estimated spacing height: {s:.2f}')

    div_headers = ['perfect', 'great', 'good', 'bad', 'miss', 'maxCombo', 'totalScore', 'calories']
    d = {}
    for i in range(div_count):
        if i == 0:
            begin = 0
            end = int(l + s // 2)
        elif i == div_count - 1:
            begin = int(i * l + (i - 1) * s + s // 2)
            end = h
        else:
            begin = int(i * l + (i - 1) * s + s // 2)
            end = int((i + 1) * l + i * s + s // 2)

        div = img[begin:end, :]
        div = cv2.cvtColor(div, cv2.COLOR_GRAY2RGB)
        header = div_headers[i]
        d[header] = div
    return d, dpi


def recognize(img_path, threshold, model, names, device):
    print(f'Reading image "{img_path}" from disk')
    img_pil, img_tensor = read_img(img_path, device)
    show(img_pil)

    print(f'Predicting bboxes')
    pred = get_prediction(img_tensor, threshold, model, names, device)
    show(draw_boxes(pil2cv(img_pil), names, pred))

    print(f'Cropping')
    crop = find_score(img_pil, pred, 'p2')
    show(crop)

    print(f'Filtering')
    filtered_crop = preprocess_score_crop(crop)
    show(filtered_crop)

    print(f'Trying to unproject')
    crop_unprojected = unproject(filtered_crop)
    show(crop_unprojected)

    print(f'Getting grade divisions')
    d, dpi = get_grade_divisions(crop_unprojected)
    for l, i in d.items():
        show(i)

    print(f'OCRing')
    p = {l: ocr(i, dpi) for l, i in d.items()}

    print(p)


def pil2cv(img):
    img = np.asarray(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
