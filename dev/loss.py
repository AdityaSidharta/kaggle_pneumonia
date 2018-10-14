from utils.logger import debug_pred_target


def calc_loss(model, criterion, data):
    img, target = data
    prediction = model(img)
    debug_pred_target(prediction, target)
    loss, loss_label, loss_bb, = criterion(prediction, target)
    return loss, loss_label, loss_bb


def record_loss(lossr_list, loss_list, train=True):
    loss, loss_label, loss_bb = loss_list
    total_lossr, label_lossr, bb_lossr = lossr_list
    if train:
        smooth_loss = total_lossr.record_train_loss(loss, True)
        smooth_label_loss = label_lossr.record_train_loss(loss_label, True)
        smooth_bb_loss = label_lossr.record_train_loss(loss_bb, True)
        return smooth_loss, smooth_label_loss, smooth_bb_loss
    else:
        total_lossr.record_epoch_val_loss(loss)
        label_lossr.record_epoch_val_loss(loss_label)
        bb_lossr.record_epoch_val_loss(loss_bb)
