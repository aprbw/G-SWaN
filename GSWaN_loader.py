import os

import numpy as np
import pandas as pd
import torch


class StandardScaler:

    def __init__(self, mean, std, fill_zeroes=True):
        '''
        fill_zeros = zeros to mean
        '''
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_spatiotemporal_dataloader(data,
                                  scaler,
                                  batch_size=32,
                                  shuffle=False,
                                  n_obvs_wind=12,  # number of observation window in timesteps
                                  n_horizon=1,  # prediction horizon in timesteps
                                  n_pred_wind=12  # number of prediction window in timesteps
                                  ):
    x = np.empty((1 + data.shape[0] - n_obvs_wind - n_horizon - n_pred_wind, data.shape[1], data.shape[2], n_obvs_wind))
    y = np.empty((1 + data.shape[0] - n_obvs_wind - n_horizon - n_pred_wind, 1, data.shape[2], n_pred_wind))
    for iw in range(n_obvs_wind):
        x[:, :, :, iw] = data[iw:iw - n_obvs_wind - n_horizon - n_pred_wind + 1, :, :]
    for iw in range(n_pred_wind):
        y[:, 0, :, iw] = data[n_obvs_wind + n_horizon - 1 + iw:-n_pred_wind + iw, 0, :]
    x[:, 0, :, :] = scaler.transform(x[:, 0, :, :])  # only scale x
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x), torch.FloatTensor(y))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader, dataset


class GSWaN_Datahandler:
    def __init__(self,
                 data_fn,
                 data_channel=0,
                 add_abs_time=False,
                 add_day_of_week=True,
                 batch_size=32,
                 r_batchsize_val=2.,
                 r_batchsize_tst=2.,
                 r_frac_train=1.,
                 r_val=.1,
                 r_tst=.2,
                 n_obvs_wind=12,  # number of observation window in timesteps
                 n_horizon=1,  # prediction horizon in timesteps
                 n_pred_wind=12,  # number of prediction window in timesteps
                 fill_zeroes=True,
                 keep_all=False):
        self.data_fn = data_fn
        self.data_channel = data_channel
        self.add_abs_time = add_abs_time
        self.add_day_of_week = add_day_of_week
        self.batch_size = batch_size
        self.r_batchsize_val = r_batchsize_val
        self.r_batchsize_tst = r_batchsize_tst
        self.r_frac_train = r_frac_train
        self.r_val = r_val
        self.r_tst = r_tst
        self.n_obvs_wind = n_obvs_wind
        self.n_horizon = n_horizon
        self.n_pred_wind = n_pred_wind
        self.fill_zeroes = fill_zeroes
        self.keep_all = keep_all

        # load data
        _, file_extension = os.path.splitext(data_fn)
        if file_extension == '.h5':
            data = pd.read_hdf(data_fn).values
        elif file_extension == '.npz':
            data = np.load(data_fn)['data']
        elif file_extension == '.csv':
            data = pd.read_csv(data_fn, header=None).values
        else:
            raise NotImplementedError('file_extension == ' + file_extension)

        # data_channel
        if len(data.shape) > 2:
            print('data has multiple channel, use channel:', data_channel)
            data = data[:, :, data_channel]

        # max min value for clamps, mean and std for scaler
        self.data_min = data.min()
        self.data_mean = data.mean()
        self.data_max = data.max()
        self.data_std = data.std()
        self.data_shape = data.shape

        # add absolute temporal time and day of week
        ati = np.tile(np.arange(data.shape[0]), [data.shape[1], 1]).T  # absolute temporal time
        dow = (ati % (7 * 24 * 12)) / (7 * 24 * 12 - 1)
        data = np.expand_dims(data, 1)
        ati = np.expand_dims(ati, 1)
        dow = np.expand_dims(dow, 1)
        if add_day_of_week:
            data = np.concatenate([data, dow], 1)
        if add_abs_time:
            data = np.concatenate([data, ati], 1)
        if keep_all:
            self.ati = ati
            self.dow = dow
            self.data = data

        # train val test separation
        trn = data[:int(data.shape[0] * (1. - r_val - r_tst)), :, :]
        self.trn_mean = trn[:, 0, :].mean()
        self.trn_std = trn[:, 0, :].std()
        trn = trn[int(trn.shape[0] * (1. - r_frac_train)):, :, :]  # r_frac_train
        val = data[int(data.shape[0] * (1. - r_val - r_tst)):-int(data.shape[0] * r_tst), :, :]
        tst = data[-int(data.shape[0] * r_tst):, :, :]
        self.trn_shape = trn.shape
        self.val_shape = val.shape
        self.tst_shape = tst.shape
        if keep_all:
            self.trn = trn
            self.val = val
            self.tst = tst

        # standard scaler
        self.scaler = StandardScaler(self.trn_mean, self.trn_std, self.fill_zeroes)

        # spatio-temporal split
        self.trn_dataloader, trn_dataset = get_spatiotemporal_dataloader(trn, self.scaler, batch_size, True,
                                                                         n_obvs_wind, n_horizon, n_pred_wind)
        self.val_dataloader, val_dataset = get_spatiotemporal_dataloader(val, self.scaler,
                                                                         int(batch_size * r_batchsize_val), False,
                                                                         n_obvs_wind, n_horizon, n_pred_wind)
        self.tst_dataloader, tst_dataset = get_spatiotemporal_dataloader(tst, self.scaler,
                                                                         int(batch_size * r_batchsize_tst), False,
                                                                         n_obvs_wind, n_horizon, n_pred_wind)
        if keep_all:
            self.trn_dataset = trn_dataset
            self.val_dataset = val_dataset
            self.tst_dataset = tst_dataset

        self.y_test = tst_dataset.tensors[1]
