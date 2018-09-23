import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from utils.transform import get_series_attributes, denormalize_bb

img_h = 1024
img_w = 1024


def draw(pred_idx, pred_df, target_df):
    pred_col = pred_df.iloc[pred_idx]
    pred_patientId, pred_x_min, pred_y_min, pred_height, pred_width, pred_label = get_series_attributes(
        pred_col
    )
    pred_x_min, pred_y_min, pred_height, pred_width = denormalize_bb(
        img_h, img_w, pred_x_min, pred_y_min, pred_width, pred_height
    )
    target_col = target_df.loc[target_df.patientId == pred_patientId, :].squeeze()
    tgt_patientId, tgt_x_min, tgt_y_min, tgt_height, tgt_width, tgt_label = get_series_attributes(
        target_col
    )

    dcm_filename = pred_patientId + ".dcm"
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
