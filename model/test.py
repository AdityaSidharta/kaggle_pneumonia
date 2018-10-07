import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
from utils.common import get_batch_info
from utils.transform import denormalize_bb, get_series_attributes
import pandas as pd


def predict_model(model, test_dataloader):
    n_obs, batch_size, batch_size_per_epoch = get_batch_info(test_dataloader)
    target_list, x_min_list, y_min_list, width_list, height_list = ([], [], [], [], [])
    model = model.eval()
    t = tqdm(enumerate(test_dataloader), total=batch_size_per_epoch)
    for idx, data in t:
        img = data
        prediction = F.sigmoid(model(img))
        prediction_array = prediction.data.cpu().numpy()
        target, x_min, y_min, width, height = [prediction_array[:, i] for i in range(5)]
        img_h, img_w = 1024., 1024.
        x_min, y_min, width, height = denormalize_bb(
            img_w, img_h, x_min, y_min, width, height
        )
        target_list.extend(target.reshape(-1).tolist())
        x_min_list.extend(x_min.reshape(-1).tolist())
        y_min_list.extend(y_min.reshape(-1).tolist())
        width_list.extend(width.reshape(-1).tolist())
        height_list.extend(height.reshape(-1).tolist())
    return pd.DataFrame(
        {
            "patientId": test_dataloader.dataset.patientId,
            "target": target_list,
            "x_min": x_min_list,
            "y_min": y_min_list,
            "width": width_list,
            "height": height_list,
        }
    )


def kaggle_submission(pred_df):
    result = []
    for idx, row in pred_df.iterrows():
        patientid, target, x_min, y_min, width, height = get_series_attributes(row)
        result.append(
            {
                "patientId": patientid,
                "PredictionString": "{} {} {} {} {}".format(
                    target, x_min, y_min, width, height
                ),
            }
        )
    return pd.DataFrame(result)
