import os

project_path = os.getenv("PROJECT_PATH")

data_source_path = os.path.join(project_path, "data_source")
output_path = os.path.join(project_path, "output")

logger_path = os.path.join(output_path, "logger")
model_cp_path = os.path.join(output_path, "model_checkpoint")
result_path = os.path.join(output_path, "result")

single_label_repo = os.path.join(data_source_path, "single_label.csv")
test_images_path = os.path.join(data_source_path, "test_images")
train_images_path = os.path.join(data_source_path, "train_images")
test_dcm_path = os.path.join(data_source_path, "test_dcm")
train_dcm_path = os.path.join(data_source_path, "train_dcm")
bb_repo = os.path.join(data_source_path, "bb.csv")
label_repo = os.path.join(data_source_path, "label.csv")
class_info_repo = os.path.join(data_source_path, "stage_1_detailed_class_info.csv")
sample_submission_repo = os.path.join(data_source_path, "stage_1_sample_submission.csv")
train_label_repo = os.path.join(data_source_path, "stage_1_train_labels.csv")
logger_repo = os.path.join(logger_path, "logger.log")


label_trainpred_repo = os.path.join(result_path, "label_trainpred.csv")
bb_trainpred_repo = os.path.join(result_path, "bb_trainpred.csv")
label_predict_repo = os.path.join(result_path, "label_predict.csv")
bb_predict_repo = os.path.join(result_path, "bb_predict.csv")
kaggle_repo = os.path.join(result_path, "kaggle.csv")
