import argparse
import numpy as np
import pandas
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
import os


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(data_path):
        data_converters = {
            "LE_ACCOUNT_NAME": str,
            "LE_ACCOUNT_NO": int,
            "time_since_start": float,
            "time_since_last_event": float,
            "type_event": int,
        }
        data = pd.read_excel(data_path, converters=data_converters)
    
        sequence_id = data["LE_ACCOUNT_NO"].unique()
        all_data = []
        for id in sequence_id:
            sequence = []
            for row in data[data["LE_ACCOUNT_NO"] == id].index:
                event = {"account_num": data.loc[row]["LE_ACCOUNT_NO"],
                         "time_since_start": data.loc[row]["time_since_start"],
                         "time_since_last_event": data.loc[row]["time_since_last_event"],
                         "type_event": data.loc[row]["type_event"]}
                sequence.append(event)
            all_data.append(sequence)
        num_types = data["type_event"].unique().__len__()
        return all_data, int(num_types)

    print('[Info] Loading data...')
    train_data, num_types = load_data('./data_train.xlsx')
    test_data, _ = load_data('./data_test.xlsx')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=True)  # false
    return trainloader, testloader, num_types


def watcher_pd(prediction, num, event_time, time_gap, event_type, watcher):
    with torch.no_grad():
        # watch predict _ winston
        watch_type = prediction[0][:, :-1, :]
        watch_pred_type = torch.max(watch_type, dim=-1)[1]  # 最后一个维度最前面差一个 -1 (4, 110)

        watch_time = prediction[1].squeeze_(-1)[:, :-1]
        time_non_pad_mask = Utils.get_non_pad_mask(event_type).squeeze_(-1)[:, 1:]
        watch_pred_time = watch_time * time_non_pad_mask  # 最后一个维度最前面差一个 -1 (4, 110)

        # to do
        # num, event_time, time_gap, event_type (4, 111)
        reshape = num.shape[0] * num.shape[1]  # 444
        num_reshape = num.reshape(reshape)
        event_time_reshape = event_time.reshape(reshape)
        time_gap_reshape = time_gap.reshape(reshape)
        truth = event_type - 1
        event_type_reshape = truth.reshape(reshape)

        # watch_pred_type, watch_pred_time (4, 110) -> 加一个数
        add_0 = torch.tensor([-10000]).cuda()  # .cuda()
        watch_pred_type_reshape = torch.ones([num.shape[0], num.shape[1]])  # (4, 111)
        watch_pred_time_reshape = torch.ones([num.shape[0], num.shape[1]])  # (4, 111)
        for i in range(watch_pred_type.shape[0]):
            watch_pred_type_reshape[i] = torch.cat([add_0, watch_pred_type[i]])
        for i in range(watch_pred_time.shape[0]):
            watch_pred_time_reshape[i] = torch.cat([add_0, watch_pred_time[i]])
        watch_pred_type_reshape = watch_pred_type_reshape.reshape(reshape)
        watch_pred_time_reshape = watch_pred_time_reshape.reshape(reshape)

        # 合并这些量成dataframe
        watcher_batch = pd.DataFrame({"LE_ACCOUNT_NO": num_reshape.cpu(),
                                      "type_event": event_type_reshape.cpu(),
                                      "pred_type": watch_pred_type_reshape.cpu(),
                                      "time_since_start": event_time_reshape.cpu(),
                                      "time_since_last_event": time_gap_reshape.cpu(),
                                      "pred_time": watch_pred_time_reshape.cpu()})
        watcher_batch_not_pad = watcher_batch[watcher_batch["LE_ACCOUNT_NO"] != 0]
        watcher = watcher._append(watcher_batch_not_pad, ignore_index=True)
    return watcher

def train_epoch(model, training_data, optimizer, pred_loss_func, opt, epoch):
    """ Epoch operation in training phase. """
    writer = pd.ExcelWriter('train_result.xlsx')
    watcher = pd.DataFrame({"LE_ACCOUNT_NO": [], "type_event": [],
                            "pred_type": [], "time_since_start": [],
                            "time_since_last_event": [], "pred_time": []})

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):

        """ prepare data """
        num, event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)  # into cla
        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time, event_type)

        if epoch == opt.epoch:
            watcher = watcher_pd(prediction, num, event_time, time_gap, event_type, watcher)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss  # event_loss=1066 pred_loss=403 se=136522
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    if epoch == opt.epoch:
        watcher.to_excel(writer, sheet_name='train_result', index=None)
        writer._save()

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt, epoch):
    """ Epoch operation in evaluation phase. """

    writer = pd.ExcelWriter('valid_result.xlsx')
    watcher = pd.DataFrame({"LE_ACCOUNT_NO": [], "type_event": [],
                            "pred_type": [], "time_since_start": [],
                            "time_since_last_event": [], "pred_time": []})

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            num, event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            pred_loss, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time, event_type)

            if epoch == opt.epoch:
                watcher = watcher_pd(prediction, num, event_time, time_gap, event_type, watcher)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    if epoch == opt.epoch:
        watcher.to_excel(writer, sheet_name='valid_result', index=None)
        writer._save()

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    train_event_losses = []
    train_pred_losses = []
    train_rmse = []
    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt, epoch)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        train_event_losses += [train_event]
        train_pred_losses += [train_type]
        train_rmse += [train_time]

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt, epoch)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]

        print('  - [Info] Maximum valid ll: {event: 8.5f}, '
              'Maximum valid accuracy: {pred: 8.5f}, Minimum valid RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll_v: 8.5f}, {acc_v: 8.5f}, {rmse_v: 8.5f}, {max_ll_v: 8.5f}, {max_acc_v: 8.5f}, {min_rmse_v: 8.5f}, {ll_t: 8.5f}, {acc_t: 8.5f}, {rmse_t: 8.5f}\n'
                    .format(epoch=epoch, ll_v=valid_event, acc_v=valid_type, rmse_v=valid_time, max_ll_v=max(valid_event_losses), max_acc_v=max(valid_pred_losses), min_rmse_v=min(valid_rmse), ll_t=train_event, acc_t=train_type, rmse_t=train_time))

        scheduler.step()


def main(cmd_args=''):
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_time_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args(cmd_args.split())

    # default device is CUDA
    opt.device = torch.device('cuda')
    # opt.device = torch.device('gpu')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('[Info] parameters:\nepoch={epoch}\nbatch_size={batch_size}\nd_model={d_model}\nd_time_model={d_time_model}\nd_rnn={d_rnn}\nd_inner_hid={d_inner_hid}\nd_k={d_k}\nd_v={d_v}\nn_head={n_head}\nn_layers={n_layers}\ndropout={dropout}\nlr={lr}\nsmooth={smooth}\n\n'.format(epoch=opt.epoch, batch_size=opt.batch_size, d_model=opt.d_model, d_time_model=opt.d_time_model, d_rnn= opt.d_rnn, d_inner_hid=opt.d_inner_hid, d_k=opt.d_k, d_v=opt.d_v, n_head=opt.n_head, n_layers=opt.n_layers, dropout=opt.dropout, lr=opt.lr, smooth=opt.smooth))
    with open(opt.log, 'a') as f:
        f.write('Epoch, V-Log-likelihood, V-Accuracy, V-RMSE, Max_v_ll, Max_v_accuracy, Min_v_RMSE, T-Log-likelihood, T-Accuracy, T-RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_time_model=opt.d_time_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data = "data/data_mimic/fold2/"
    batch = 4
    n_head = 4
    n_layers = 4
    d_model = 512  # 512
    d_time_model = 512  # 减小蒙特卡洛计算时间 512
    d_rnn = 64  # 64
    d_inner = 1024
    d_k = 512  # 512
    d_v = 512  # 512
    dropout = 0.1  # 0.1
    lr = 1e-4
    smooth = 0.1
    epoch = 300
    log = "log.txt"

    cmd_args = f"-data {data} -batch {batch} -n_head {n_head} -n_layers {n_layers} -d_model {d_model} -d_time_model {d_time_model} -d_rnn {d_rnn} -d_inner {d_inner} -d_k {d_k} -d_v {d_v} -dropout {dropout} -lr {lr} -smooth {smooth} -epoch {epoch} -log {log}"
    main(cmd_args)
