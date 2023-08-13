import math
import numpy as np
import cv2 as cv
import os

from lib.general import xywh2xyxy, xywha2xyxyxyxy


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, :4] = xywh2xyxy(boxes[:, :4])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((x1 - pad_x // 2) / unpad_w) * orig_w
    y1 = ((y1 - pad_y // 2) / unpad_h) * orig_h
    x2 = ((x2 - pad_x // 2) / unpad_w) * orig_w
    y2 = ((y2 - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 0] = (x1 + x2) / 2
    boxes[:, 1] = (y1 + y2) / 2
    boxes[:, 2] = (x2 - x1)
    boxes[:, 3] = (y2 - y1)
    return boxes


def get_color(c, x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def plot_boxes(img_path, boxes, class_names, img_size, output_folder, color=None):
    img = np.array(cv.imread(img_path))
    nc = len(class_names)

    if len(boxes):
        boxes = rescale_boxes(boxes, img_size, img.shape[:2])
        polys = xywha2xyxyxyxy(boxes[:, :5])
        polys = np.array(polys, dtype=np.intp)

        preds = np.array(boxes[:, 5:])

        for poly, pred in zip(polys, preds):
            cls_id = int(pred[1])
            offset = cls_id * 93 % nc
            red = get_color(2, offset, nc)
            green = get_color(1, offset, nc)
            blue = get_color(0, offset, nc)
            if color:
                rgb = color
            else:
                rgb = (red, green, blue)

            cv.drawContours(img, [poly], 0, rgb, 2)
            img = cv.putText(img, class_names[cls_id] + ":" + str(round(pred[0], 2)),
                            tuple(poly[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 1)

    output_path = os.path.join(output_folder, os.path.split(img_path)[-1])
    cv.imwrite(output_path, img)
