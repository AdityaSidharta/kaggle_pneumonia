import torch


def img2tensor(img_array, device):
    img_array = img_array.transpose((2, 0, 1))
    return torch.from_numpy(img_array).float().to(device)


def normalize_bb(img_w, img_h, x_min, y_min, width, height):
    x_min = float(x_min) / float(img_w)
    y_min = float(y_min) / float(img_h)
    width = float(width) / float(img_w)
    height = float(height) / float(img_h)
    return x_min, y_min, width, height


def denormalize_bb(img_w, img_h, x_min, y_min, width, height):
    x_min = x_min * float(img_w)
    y_min = y_min * float(img_h)
    width = width * float(img_w)
    height = height * float(img_h)
    return x_min, y_min, width, height


def to_imgaugbb(x_min, y_min, width, height):
    x1 = x_min
    y1 = y_min
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def get_series_attributes(series):
    return (
        series.patientId,
        series.x,
        series.y,
        series.width,
        series.height,
        series.Target,
    )
