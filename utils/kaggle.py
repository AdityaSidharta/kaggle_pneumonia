def to_predictionstring(x):
    prob, x, y, width, height = x
    return "{} {} {} {} {}".format(prob, x, y, width, height)


def create_predict_df(label_predict_df, bb_predict_df):
    predict_df = label_predict_df.merge(bb_predict_df, how="left", on="name")
    predict_df["patientId"] = predict_df["name"]
    predict_df["x"] = predict_df["label"].apply(lambda x: x.split(" ")[0])
    predict_df["y"] = predict_df["label"].apply(lambda x: x.split(" ")[1])
    predict_df["width"] = predict_df["label"].apply(lambda x: x.split(" ")[2])
    predict_df["height"] = predict_df["label"].apply(lambda x: x.split(" ")[3])
    predict_df["Target"] = predict_df["prob"].apply(lambda x: 1. if x >= 0.5 else 0.)
    predict_df = predict_df[
        ["patientId", "Target", "prob", "x", "y", "width", "height"]
    ]
    return predict_df


def create_kaggle_df(predict_df):
    predict_df["PredictionString"] = predict_df[
        ["prob", "x", "y", "width", "height"]
    ].apply(to_predictionstring, axis=1)
    kaggle_df = predict_df[["patientId", "PredictionString"]]
    return kaggle_df
