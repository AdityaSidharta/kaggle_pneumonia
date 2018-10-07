import pandas as pd
import pydicom
from PIL import Image

from utils.envs import *


def save_image_from_dc(dcm_path):
    dcm = pydicom.read_file(dcm_path)
    filename = dcm_path.split("/")[-1].split(".")[0] + ".png"
    array = dcm.pixel_array
    image = Image.fromarray(array)
    filepath = os.path.join(train_images_path, filename)
    image.save(filepath)


def get_dcm_list(path):
    return [
        os.path.join(path, x) for x in os.listdir(path) if x.split(".")[-1] == "dcm"
    ]


def create_single_label_df(train_repo):
    train_label_df = pd.read_csv(train_repo)
    train_label_df["area"] = train_label_df.width * train_label_df.height
    single_label_df = (
        train_label_df.sort_values("area", ascending=False)
        .drop_duplicates("patientId")
        .sort_index()
        .reset_index(drop=True)
        .copy()
    )
    return single_label_df
