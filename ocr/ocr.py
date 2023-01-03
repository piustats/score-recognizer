import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from dataset.transforms import ExIfTransposeImg
from ocr.unproject.unproject_text import unproject
from visualizer.objects import filter, show, draw_boxes
import pytesseract
from re import match as re_match


def crop_image(img, box):
    return img.crop(box)


def preprocess_score_crop(img, align_position, upper_threshold=1001):
    color = pil2cv(img)
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # v = (img_hsv[:, :, 2] + 127) % 255
    h, w = img.shape
    # ret, thresh = cv2.threshold(inputImage, 200, 255, cv2.THRESH_BINARY_INV)
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, upper_threshold, -100)
    filtered = 255 - filtered

    hw = w // 3
    ph = int(h * 0.7)
    if align_position == 'right':
        hw = w // 3
        filtered[:ph, :hw] = 255
    elif align_position == 'left':
        hw = 2 * (w // 3)
        filtered[:ph, hw:] = 255

    mask = 255 - filtered
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    im_thresh_color = cv2.bitwise_and(color, mask)
    _, _, bands = im_thresh_color.shape
    median = []
    std = []
    for i in range(bands):
        band = im_thresh_color[:, :, i]
        band = band[band != 0]
        band = band.reshape(-1)
        band_median = np.median(band)
        band_std = int(np.std(band))
        median.append(band_median)
        std.append(band_std)
    lower_range = np.array([median - 2 * std for median, std in zip(median, std)])
    upper_range = np.array([median + 2 * std for median, std in zip(median, std)])
    mask = cv2.inRange(color, lower_range, upper_range)

    if align_position == 'right':
        mask[:ph, :hw] = 0
    elif align_position == 'left':
        mask[:ph, hw:] = 0

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    im_thresh_color = cv2.bitwise_and(color, mask)

    result = cv2.cvtColor(im_thresh_color, cv2.COLOR_BGR2GRAY)
    result[result != 0] = 255
    result = 255 - result

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result


def get_object_prediction(img, threshold, model, names, device):
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


def ocr(tesseract_model, img, dpi):
    config = f'--dpi {dpi} --tessdata-dir {tesseract_model} --psm 4 -l Karnivore' \
             f' -c tessedit_char_whitelist=0123456789. -c load_system_dawg=0 -c load_freq_dawg=0 -c load_number_dawg=0' \
             f'-c load_unambig_dawg=0 -c load_bigram_dawg=0 -c load_fixed_length_dawgs=0 -c load_punc_dawg=0' \
             f'-c segment_penalty_garbage=0 -c segment_penalty_dict_nonword=0 -c segment_penalty_dict_frequent_word=0' \
             f'-c segment_penalty_dict_case_ok=0 -c segment_penalty_dict_case_bad=0'
    ocred = pytesseract.image_to_string(img, config=config)
    return ocred.rstrip('\n')


def divide_ocr_and_assess(tesseract_model, img):
    print(f'Getting grade divisions')
    d, dpi = get_grade_divisions(img)
    # for l, i in d.items():
    #     show(i)
    print(f'OCRing')
    p = {l: ocr(tesseract_model, i, dpi) for l, i in d.items()}
    grade = assess_score_ocred(p)
    return p, grade


def assess_score_ocred(dic):
    assessment_grade = 0
    # if totalScore is possibly wrong, score 0
    totalScore = dic['totalScore']
    if not totalScore.isnumeric() and totalScore[-2:] != '00':
        print(f'totalScore "{totalScore}" is malformed')
        return assessment_grade

    for l, score in dic.items():
        if l != 'calories':
            if len(score) > 2 and score.isnumeric():
                assessment_grade += 1
            else:
                print(f'{l} "{score}" is malformed')

    if is_number(dic['calories']):
        assessment_grade += 1
    else:
        print(f'calories "{dic["calories"]}" is malformed')

    return assessment_grade / len(dic)


def is_number(s):
    """ Returns True if string is a number. """
    if re_match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True


def estimate_score_dimensions(img):
    h, w = img.shape
    l = h / 14.82
    s = (h - 8 * l) / 7.0
    dpi = int((l * l) / 8.33)
    return l, s, dpi


def get_grade_divisions(img):
    h, w = img.shape
    div_count = 8
    l, s, dpi = estimate_score_dimensions(img)

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


def recognize_score(tesseract_model, img_pil, box, align_position):
    print(f'Cropping')
    crop = crop_image(img_pil, box)
    show(crop)

    print(f'Filtering')
    filtered_crop = preprocess_score_crop(crop, align_position)
    show(filtered_crop)

    _, _, dpi = estimate_score_dimensions(filtered_crop)
    h, w = filtered_crop.shape
    if dpi < 100:
        filtered_crop = cv2.resize(filtered_crop, (int(w * 2.25), int(h * 2.25)), interpolation=cv2.INTER_NEAREST)
        show(filtered_crop)
    elif dpi < 200:
        filtered_crop = cv2.resize(filtered_crop, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_NEAREST)
        show(filtered_crop)

    # 1. try to ocr without projection
    p, s = divide_ocr_and_assess(tesseract_model, filtered_crop)
    print(f'OCR without projection score: {s}')
    print(f'Prediction: {p}')

    # 2. try to ocr with projection
    print(f'Trying to unproject')
    crop_unprojected = unproject(filtered_crop)
    show(crop_unprojected)

    p_proj, s_proj = divide_ocr_and_assess(tesseract_model, crop_unprojected)

    print(f'OCR with projection score: {s_proj}')
    print(f'Prediction: {p_proj}')
    return (s, p) if s > s_proj else (s_proj, p_proj)


def search_best_ocr_score(tesseract_model, img_pil, box, align_position):
    # filters = [1001 + i * 50 for i in range(3)]
    # filters.extend([851 + i * 50 for i in range(3)])
    paddings = [0.025 * i for i in range(1, 6)]
    ocr_predictions = []
    # for filter in filters:
    for padding in paddings:
        print(f'Trying padding={padding:.2f}')
        bh = box[3] - box[1]
        bw = box[2] - box[0]
        p1x = box[0] - bw * padding
        p1y = box[1] - bh * padding
        p2x = box[2] + bw * padding
        p2y = box[3] + bh * padding
        newBox = (int(p1x), int(p1y), int(p2x), int(p2y))
        # show(draw_boxes(pil2cv(img_pil), names, pred))
        s, p = recognize_score(tesseract_model, img_pil, newBox, align_position)

        ocr_predictions.append((s, p))
        if s == 1.0:
            break
    # if s == 1.0:
    #     break
    sort = sorted(ocr_predictions, key=lambda x: x[0], reverse=True)
    print(f'best prediction: {sort[0]}')
    return sort[0]


def recognize(img_path, threshold, cnn_model, tesseract_model, names, device):
    ocred = {}
    print(f'Reading image "{img_path}" from disk')
    img_pil, img_tensor = read_img(img_path, device)
    show(img_pil)

    print(f'Predicting bboxes')
    pred = get_object_prediction(img_tensor, threshold, cnn_model, names, device)
    show(draw_boxes(pil2cv(img_pil), names, pred))

    if pred['p1']:
        boxp1 = pred['p1'][0][2]
        p1score = search_best_ocr_score(tesseract_model,img_pil, boxp1, 'left')
        score = clean_up_score(p1score[1])
        ocred['p1Score'] = score

    if pred['p2']:
        boxp2 = pred['p2'][0][2]
        p2score = search_best_ocr_score(tesseract_model, img_pil, boxp2, 'right')
        score = clean_up_score(p2score[1])
        ocred['p2Score'] = score

    print(f'OCR completed:')
    print(ocred)

    return ocred


def clean_up_score(score):
    out = {}
    total_score = score['totalScore']
    if total_score.isnumeric() and total_score[-2:] != '00':
        out['totalScore'] = int(total_score)

    for l, s in score.items():
        if l != 'calories':
            if len(s) > 2 and s.isnumeric():
                out[l] = int(s)

    if is_number(score['calories']):
        out['calories'] = float(score['calories'])

    return out


def pil2cv(img):
    img = np.asarray(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
