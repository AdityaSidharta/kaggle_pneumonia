from tqdm import tqdm


def train_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def fit_model(
    pneunet_model, n_epoch, dev_dataloader, val_dataloader, optimizer, criterion
):
    pneunet_model.train()
    for idx_epoch in tqdm(range(n_epoch), total=n_epoch):
        t = tqdm(enumerate(dev_dataloader), total=322)
        for idx, data in t:
            img, target = data
            pred = pneunet_model(img)
            loss, loss_label, loss_bb = criterion(pred, target)
            train_step(optimizer, loss)
            t.set_description("loss : {}".format(loss))
        print(loss_label, loss_bb)
