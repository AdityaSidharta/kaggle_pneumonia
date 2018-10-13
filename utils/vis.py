import os

import imgaug as ia
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pydicom
import skimage

from utils.envs import train_dcm_path, test_dcm_path
from utils.transform import get_series_attributes, to_imgaugbb

img_h = 1024
img_w = 1024


def draw_imgaug(pred_idx, pred_df, target_df, train=True):
    pred_col = pred_df.iloc[pred_idx]
    pred_patientid, pred_x_min, pred_y_min, pred_width, pred_height, pred_label = get_series_attributes(
        pred_col
    )
    target_col = target_df.loc[target_df.patientId == pred_patientid, :].squeeze()
    tgt_patientid, tgt_x_min, tgt_y_min, tgt_width, tgt_height, tgt_label = get_series_attributes(
        target_col
    )

    dcm_filename = pred_patientid + ".dcm"
    if train:
        dcm_path = os.path.join(train_dcm_path, dcm_filename)
    else:
        dcm_path = os.path.join(test_dcm_path, dcm_filename)

    img_array = pydicom.read_file(dcm_path).pixel_array
    img_array = skimage.color.gray2rgb(img_array)

    if tgt_label:
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = to_imgaugbb(
            tgt_x_min, tgt_y_min, tgt_width, tgt_height
        )
        tgt_bbs = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=tgt_x1, y1=tgt_y1, x2=tgt_x2, y2=tgt_y2)],
            shape=img_array.shape,
        )
        img_array = tgt_bbs.draw_on_image(img_array, thickness=2, color=[0, 0, 255])

    if pred_label:
        pred_x1, pred_y1, pred_x2, pred_y2 = to_imgaugbb(
            pred_x_min, pred_y_min, pred_width, pred_height
        )
        pred_bbs = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=pred_x1, y1=pred_y1, x2=pred_x2, y2=pred_y2)],
            shape=img_array.shape,
        )
        img_array = pred_bbs.draw_on_image(img_array, thickness=2, color=[255, 0, 0])

    if tgt_label and pred_label:
        bb1 = ia.BoundingBox(x1=tgt_x1, y1=tgt_y1, x2=tgt_x2, y2=tgt_y2)
        bb2 = ia.BoundingBox(x1=pred_x1, y1=pred_y1, x2=pred_x2, y2=pred_y2)
        print("IoU Score : {}".format(bb1.iou(bb2)))

    plt.imshow(img_array)


def draw(pred_idx, pred_df, target_df, train=False):
    pred_col = pred_df.iloc[pred_idx]
    pred_patientid, pred_x_min, pred_y_min, pred_width, pred_height, pred_label = get_series_attributes(
        pred_col
    )
    target_col = target_df.loc[target_df.patientId == pred_patientid, :].squeeze()
    tgt_patientid, tgt_x_min, tgt_y_min, tgt_width, tgt_height, tgt_label = get_series_attributes(
        target_col
    )

    dcm_filename = pred_patientid + ".dcm"
    if train:
        dcm_path = os.path.join(train_dcm_path, dcm_filename)
    else:
        dcm_path = os.path.join(test_dcm_path, dcm_filename)
    img_array = pydicom.read_file(dcm_path).pixel_array

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.imshow(img_array)
    if pred_label:
        pred_rect = patches.Rectangle(
            (pred_x_min, pred_y_min),
            pred_width,
            pred_height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        axes.add_patch(pred_rect)

    if tgt_label:
        target_rect = patches.Rectangle(
            (tgt_x_min, tgt_y_min),
            tgt_width,
            tgt_height,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        axes.add_patch(target_rect)
    plt.show()
