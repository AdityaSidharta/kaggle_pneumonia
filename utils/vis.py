import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from utils.transform import get_series_attributes, denormalize_bb, to_imgaugbb
import imgaug as ia
from imgaug import augmenters as iaa

img_h = 1024
img_w = 1024


def draw_imgaug(pred_idx, pred_df, target_df):
    pred_col = pred_df.iloc[pred_idx]
    pred_patientid, pred_x_min, pred_y_min, pred_width, pred_height, pred_label = get_series_attributes(
        pred_col
    )
    pred_x_min, pred_y_min, pred_width, pred_height = denormalize_bb(
        img_h, img_w, pred_x_min, pred_y_min, pred_width, pred_height
    )
    target_col = target_df.loc[target_df.patientId == pred_patientid, :].squeeze()
    tgt_patientid, tgt_x_min, tgt_y_min, tgt_width, tgt_height, tgt_label = get_series_attributes(
        target_col
    )

    dcm_filename = pred_patientid + ".dcm"
    img_array = pydicom.read_file(dcm_filename).pixel_array

    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = to_imgaugbb(
        tgt_x_min, tgt_y_min, tgt_width, tgt_height
    )
    pred_x1, pred_y1, pred_x2, pred_y2 = to_imgaugbb(
        pred_x_min, pred_y_min, pred_width, pred_height
    )

    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=tgt_x1, y1=tgt_y1, x2=tgt_x2, y2=tgt_y2),
        ia.BoundingBox(x1=pred_x1, y1=pred_y1, x2=pred_x2, y2=pred_x2)
    ], shape=(img_h, img_w))

    return bbs.draw_on_image(img_array, thickness=2)


def draw(pred_idx, pred_df, target_df):
    pred_col = pred_df.iloc[pred_idx]
    pred_patientid, pred_x_min, pred_y_min, pred_width, pred_height, pred_label = get_series_attributes(
        pred_col
    )
    target_col = target_df.loc[target_df.patientId == pred_patientid, :].squeeze()
    tgt_patientid, tgt_x_min, tgt_y_min, tgt_width, tgt_height, tgt_label = get_series_attributes(
        target_col
    )

    dcm_filename = pred_patientid + ".dcm"
    img_array = pydicom.read_file(dcm_filename).pixel_array

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
