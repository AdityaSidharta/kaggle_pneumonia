import imgaug as ia
import imgaug.augmenters as iaa
import skimage
from utils.transform import to_imgaugbb

random_seed = 0

# TODO still not good
def modify_image_train(img, bb):
    img_height, img_width = img.shape[0], img.shape[1]
    x_min, y_min, width, height = bb
    x1, y1, x2, y2 = to_imgaugbb(x_min, y_min, width, height)
    bb_array = ia.BoundingBoxesOnImage(
        [ia.BoundingBox(x1, y1, x2, y2)], shape=(img_height, img_width)
    )

    rgb_img = skimage.color.gray2rgb(img)
    imagenet_height, imagenet_width = 224, 224
    seq = iaa.Sequential(
        [iaa.Scale({"height": imagenet_height, "weight": imagenet_width})]
    )
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([rgb_img])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bb_array])[0]
    return image_aug, bbs_aug
