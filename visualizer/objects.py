import cv2
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
import PIL
from dataset.transforms import ExIfTransposeImg


def get_prediction(img_path, threshold, model, names):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    img = Image.open(img_path)
    transform = T.Compose([
        ExIfTransposeImg(),
        T.ToTensor()
    ])
    img = transform(img)
    pred = model([img])
    pred_class_ids = [names[i]['id'] for i in list(pred[0]['labels'].numpy())]
    pred_class = [names[i]['name'] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    return filter(pred_class_ids, pred_boxes, pred_score, threshold)

    # pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    # pred_boxes = pred_boxes[:pred_t + 1]
    # pred_class = pred_class[:pred_t + 1]
    # return pred_boxes, pred_class


def get_n_best(pred_class_ids, pred_boxes, pred_score, label_list, threshold, n):
    preds = [(i, prob, box) for i, prob, box in zip(pred_class_ids, pred_score, pred_boxes) if
             i in label_list and prob > threshold]
    if len(preds) == 0:
        return None
    else:
        best_pred = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
        return best_pred


def filter(pred_class_ids, pred_boxes, pred_score, threshold):
    best_grade = get_n_best(pred_class_ids, pred_boxes, pred_score, [i for i in range(16)], 0.0, 2)
    best_level = get_n_best(pred_class_ids, pred_boxes, pred_score, [16], threshold, 2)
    best_p1 = get_n_best(pred_class_ids, pred_boxes, pred_score, [17], threshold, 1)
    best_p2 = get_n_best(pred_class_ids, pred_boxes, pred_score, [18], threshold, 1)
    best_title = get_n_best(pred_class_ids, pred_boxes, pred_score, [19], threshold, 1)

    return {'grades': best_grade,
            'levels': best_level,
            'p1': best_p1,
            'p2': best_p2,
            'title': best_title}


def object_detection_api(img_path, model, names, threshold=0.5, rect_th=3, text_size=3, text_th=3, output_image=None):
    pred = get_prediction(img_path, threshold, model, names)
    # Get predictions
    img = cv2.imread(img_path)
    # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to RGB
    for object, list in pred.items():
        if list != None:
            for params in list:
                i, prob, boxes = params
                pred_cls = names[i]['name']

                cv2.rectangle(img, boxes[0], boxes[1], color=(0, 255, 0), thickness=rect_th)
                # Draw Rectangle with the coordinates
                cv2.putText(img, f'{pred_cls} ({prob:.2f})', boxes[0], cv2.FONT_HERSHEY_SIMPLEX, text_size,
                            (0, 255, 0), thickness=text_th)
    # Write the prediction class
    if output_image != None:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image, img)
    else:
        plt.figure(figsize=(20, 30))
        # display the output image
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()
