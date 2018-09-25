import os

project_path = os.getenv("PROJECT_PATH")

data_source_path = os.path.join(project_path, "data_source")
test_images_path = os.path.join(data_source_path, "test_images")
train_images_path = os.path.join(data_source_path, "train_images")
test_dcm_path = os.path.join(data_source_path, "test_dcm")
train_dcm_path = os.path.join(data_source_path, "train_dcm")
class_info_repo = os.path.join(data_source_path, "stage_1_detailed_class_info.csv")
sample_submission_repo = os.path.join(data_source_path, "stage_1_sample_submission.csv")
train_label_repo = os.path.join(data_source_path, "stage_1_train_labels.csv")
