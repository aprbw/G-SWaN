import time

import numpy as np
import pandas as pd
import torch

import GSWaN_loader
import config_global_dummy as G
import util
from EpochAndBatches import EpochAndBatches as EnB
from augment import Augment
from model import GWNet
from util import str2bool


def main(args, **model_kwargs):
    print(vars(args))
    # arian output dictionary   
    G.args = args
    args.enb_output_filepath = args.save
    G.enb = EnB(args.epochs, 0, args.project_name, args.sweep_name, args.run_name, args.enb_output_filepath)
    G.enb.verbose = args.verbose
    G.enb.args = args
    G.enb.save()

    # deterministic run for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load data
    data = GSWaN_loader.GSWaN_Datahandler(args.data_fn,
                                        add_abs_time=True,
                                        batch_size=args.batch_size,
                                        r_batchsize_val=2.,
                                        r_batchsize_tst=4.,
                                        r_frac_train=args.frac_train)
    G.enb.log_msg('data loaded')
    G.enb.log_msg('data.trn_shape' + str(data.trn_shape))
    G.enb.log_msg('data.val_shape' + str(data.val_shape))
    G.enb.log_msg('data.tst_shape' + str(data.tst_shape))
    G.enb.n_batch = data.trn_shape[0] // args.batch_size
    scaler = data.scaler
    # load adj data
    aptinit, supports = util.make_graph_inputs(args, device)

    # model
    augment = Augment(args)
    model = GWNet.from_args(args, device, supports, aptinit, **model_kwargs)
    predictive_head = model.predictive_head
    G.enb.log_msg(repr(model))
    G.enb.log_msg(repr(predictive_head))
    model.to(device)
    predictive_head.to(device)
    # engine setup
    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(predictive_head.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)
    # loss
    # optim
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.scheduler_patience, factor=args.scheduler_factor)
    # train; scxe = spatial contrastive cross entropy
    train_loss, train_mae, train_mape, train_rmse, train_scxe, train_tcmse = [], [], [], [], [], []
    G.enb.best_epoch = None
    G.enb.best_model_state_dict = model.state_dict()
    G.enb.best_predictive_head_state_dict = predictive_head.state_dict()
    G.enb.lowest_val_loss_yet = None
    G.enb.epochs_since_best_mae = None
    G.enb.train_loss_at_best_vaild = None
    G.enb.train_mae_at_best_vaild = None
    G.enb.train_mape_at_best_vaild = None
    G.enb.train_rmse_at_best_vaild = None
    G.enb.train_scxe_at_best_vaild = None
    G.enb.train_tcmse_at_best_vaild = None
    G.enb.best_valid_mae = None
    G.enb.best_valid_mape = None
    G.enb.best_valid_rmse = None
    G.enb.lowest_val_loss_yet = float("inf")  # high value, will get overwritten
    G.enb.epochs_since_best_mae = 0
    G.enb.save()
    mb = range(1, args.epochs + 1)
    for _ in mb:
        if G.enb.epochs_since_best_mae >= args.es_patience:
            break
        G.enb.next_epoch()
        for i_iter, (x, y) in enumerate(iter(data.trn_dataloader)):
            # load
            G.enb.next_batch()
            # x
            trainx = torch.Tensor(x).to(device)  # BDNL (D=2)
            augment.augment(trainx)
            trainx = torch.nn.functional.pad(trainx, (1, 0, 0, 0))
            # y
            trainy = torch.Tensor(y).to(device)  # BDNL (D=2)
            y = trainy[:, 0, :, :]  # BNL
            if y.max() == 0:
                continue

            # train()
            model.train()
            predictive_head.train()
            optimizer.zero_grad()
            # forward
            r = model.forward_main(trainx)
            # prediction
            h = predictive_head(r).transpose(1, 3)
            h = scaler.inverse_transform(h)  # [batch_size,1,num_nodes, seq_length]
            assert h.shape[1] == 1
            # loss
            mae_, mape_, rmse_ = util.calc_metrics(h.squeeze(1), y, null_val=0.0)
            loss_ = 1 * mae_
            # backwards
            loss_.backward()
            if args.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                torch.nn.utils.clip_grad_norm_(predictive_head.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            # logging
            loss = loss_.detach().item()
            mae = mae_.detach().item()
            mape = mape_.detach().item()
            rmse = rmse_.detach().item()
            G.enb.log_batch('loss:trn_loss', loss)
            G.enb.log_batch('loss:trn_mae', mae)
            G.enb.log_batch('loss:trn_mape', mape)
            G.enb.log_batch('loss:trn_rmse', rmse)
            G.enb.log_batch('gpu-mem', torch.cuda.memory_reserved(device))
            train_loss.append(loss)
            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            if args.n_iters is not None and i_iter >= args.n_iters:
                break
        G.enb.log_epoch('loss:trn_loss', np.mean(train_loss))
        G.enb.log_epoch('loss:trn_mae', np.mean(train_mae))
        G.enb.log_epoch('loss:trn_mape', np.mean(train_mape))
        G.enb.log_epoch('loss:trn_rmse', np.mean(train_rmse))
        G.enb.log_epoch('loss:trn_scxe', np.mean(train_scxe))
        G.enb.log_epoch('loss:trn_tcmse', np.mean(train_tcmse))
        # eval
        with torch.no_grad():
            total_time, valid_mae, valid_mape, valid_rmse = \
                eval_(data.val_dataloader, device, model, predictive_head, scaler, data.data_max)
        G.enb.log_epoch('loss:val_mae', np.mean(valid_mae))
        G.enb.log_epoch('loss:val_mape', np.mean(valid_mape))
        G.enb.log_epoch('loss:val_rmse', np.mean(valid_rmse))
        # misc log
        G.enb.log_epoch('gpu-mem', torch.cuda.memory_reserved(device))
        G.enb.log_epoch('total_time', total_time)
        # scheduler
        scheduler.step(metrics=torch.Tensor(valid_mae).to(device).sum())
        # check if better
        u_valid_loss = np.mean(valid_mae)
        if u_valid_loss < G.enb.lowest_val_loss_yet:
            G.enb.log_msg('new validation low, saving model.' +
                          ' G.enb.lowest_val_loss_yet: ' + str(G.enb.lowest_val_loss_yet) +
                          ' m.valid_loss: ' + str(u_valid_loss))
            G.enb.best_epoch = G.enb.i_epoch
            G.enb.best_model_state_dict = model.state_dict()
            G.enb.best_predictive_head_state_dict = predictive_head.state_dict()
            G.enb.lowest_val_loss_yet = u_valid_loss
            G.enb.epochs_since_best_mae = 0
            G.enb.train_loss_at_best_vaild = np.mean(train_loss)
            G.enb.train_mae_at_best_vaild = np.mean(train_mae)
            G.enb.train_mape_at_best_vaild = np.mean(train_mape)
            G.enb.train_rmse_at_best_vaild = np.mean(train_rmse)
            G.enb.train_scxe_at_best_vaild = np.mean(train_scxe)
            G.enb.train_tcmse_at_best_vaild = np.mean(train_tcmse)
            G.enb.best_valid_mae = np.mean(valid_mae)
            G.enb.best_valid_mape = np.mean(valid_mape)
            G.enb.best_valid_rmse = np.mean(valid_rmse)
        else:
            G.enb.epochs_since_best_mae += 1
    G.enb.last_train_loss = np.mean(train_loss)
    G.enb.last_train_mae = np.mean(train_mae)
    G.enb.last_train_mape = np.mean(train_mape)
    G.enb.last_train_rmse = np.mean(train_rmse)
    G.enb.last_train_scxe = np.mean(train_scxe)
    G.enb.last_train_tcmse = np.mean(train_tcmse)
    # Metrics on test data
    G.enb.log_msg('Training complete. Testing...')
    model.load_state_dict(G.enb.best_model_state_dict)
    predictive_head.load_state_dict(G.enb.best_predictive_head_state_dict)
    (G.enb.test_time_start,
     G.enb.test_time_end,
     G.enb.test_time_duration,
     G.enb.test_met_df,
     pred) = test_(model, predictive_head, data, args)
    G.enb.log_msg('test_met_df' + str(G.enb.test_met_df))
    G.enb.test_mae = G.enb.test_met_df['mae'].mean()
    G.enb.test_mape = G.enb.test_met_df['mape'].mean()
    G.enb.test_rmse = G.enb.test_met_df['rmse'].mean()
    G.enb.script_end_time = time.time()
    G.enb.script_duration = G.enb.script_end_time - G.enb.script_start_time
    G.enb.save()
    return G.enb


def test_(model, predictive_head, data, args):
    model.eval()
    outputs = []
    realy = data.y_test[:, 0, :, :]
    test_time_start = time.time()
    for _, (x, y) in enumerate(iter(data.tst_dataloader)):
        testx = torch.Tensor(x).to(args.device)
        with torch.no_grad():
            preds = model.forward_main(testx)
            preds = predictive_head(preds).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    test_time_end = time.time()
    test_time_duration = test_time_end - test_time_start
    yhat = torch.cat(outputs, dim=0)
    pred = torch.empty_like(yhat)[:realy.size(0), ...]
    realy = realy[:pred.size(0), ...].to(pred.device)
    test_met = []
    for i in range(args.seq_length):
        pred[:, :, i] = data.scaler.inverse_transform(yhat[:, :, i])
        pred[:, :, i] = torch.clamp(pred[:, :, i], min=0., max=data.data_max)
        test_met.append([x.item() for x in util.calc_metrics(pred[:, :, i], realy[:, :, i])])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_time_start, test_time_end, test_time_duration, test_met_df, pred


def eval_(ds, device, model, predictive_head, scaler, max_val_clamp):
    """Run validation."""
    valid_mae = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    total_time = 0
    for i_batch, (x, y) in enumerate(iter(ds)):
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        y = testy[:, 0, :, :]  # torch.unsqueeze(testy, dim=1)
        model.eval()
        r = model.forward_main(testx)
        h = predictive_head(r).transpose(1, 3)
        h = scaler.inverse_transform(h)  # [batch_size,1,num_nodes, seq_length]
        h = torch.clamp(h, min=0., max=max_val_clamp)
        assert h.shape[1] == 1
        # loss
        mae_, mape_, rmse_ = util.calc_metrics(h.squeeze(1), y, null_val=0.0)
        mae = mae_.detach().item()
        mape = mape_.detach().item()
        rmse = rmse_.detach().item()
        valid_mae.append(mae)
        valid_mape.append(mape)
        valid_rmse.append(rmse)
        G.enb.log_batch('loss:val_mae', np.mean(valid_mae), 28)
        G.enb.log_batch('loss:val_mape', np.mean(valid_mape), 28)
        G.enb.log_batch('loss:val_rmse', np.mean(valid_rmse), 28)
        G.enb.log_batch('eval gpu-mem', torch.cuda.memory_reserved(device), 28)
        total_time = time.time() - s1
    return total_time, valid_mae, valid_mape, valid_rmse


if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--save', type=str, default='experiment', help='save path')
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20,
                        help='quit if no improvement after this many iterations')
    # Arian's
    parser.add_argument('--project_name', type=str, default='default_project', help='project name')
    parser.add_argument('--sweep_name', type=str, default='default_sweep', help='sweep name')
    parser.add_argument('--run_name', type=str, default='default_run', help='run name')
    parser.add_argument('--enb_output_filepath', type=str, default='default_run', help='enb_output_filepath')
    parser.add_argument('--verbose', type=int, default=30,
                        help='0: print just start and end\n' +
                             '1: print per epoch\n' +
                             '2: print per batch\n' +
                             '3: print per gauss batch.')
    parser.add_argument('--seed', type=int, default='42',
                        help='RNG random seed for deterministic runs and reproducibility.')
    parser.add_argument('--scheduler_patience', type=int, default='10',
                        help='Number of epochs with no improvement, after which learning rate will be reduced.')
    parser.add_argument('--scheduler_factor', type=float, default='0.4',
                        help='Factor by which the learning rate will be reduced')
    parser.add_argument('--frac_train', type=float, default=1.0,
                        help='fraction of training dataset used for data efficiency analysis')
    parser.add_argument('--is_pad_at_engine', type=str2bool, nargs='?', const=True, default=False,
                        help='There is this padding in the original code, and I have no idea why...')

    # augment
    parser.add_argument('--augment_occlude_spatial_probability', type=float, default=0.05,
                        help='augment by occluding a station. Probability of each station occluded.')
    parser.add_argument('--augment_occlude_spatial_scale', type=float, default=0.05,
                        help='augment by occluding a station. Scale of occlusion: *=uniform_noise*scale.')
    parser.add_argument('--augment_occlude_temporal_probability', type=float, default=0.,
                        help='augment by occluding a timestep. Probability of each timestep occluded.')
    parser.add_argument('--augment_occlude_temporal_scale', type=float, default=0.,
                        help='augment by occluding a timestep. Scale of occlusion: *=uniform_noise*scale.')
    parser.add_argument('--augment_swap_spatial_k', type=int, default=0,
                        help='augment by swapping (permute) k stations')
    parser.add_argument('--augment_swap_temporal_k', type=int, default=0,
                        help='augment by swapping (permute) k timestep')
    parser.add_argument('--augment_scramble_spatial_probability', type=float, default=0.,
                        help='augment by scrambling the timesteps of each station with probability.')
    parser.add_argument('--augment_scramble_temporal_probability', type=float, default=0.05,
                        help='augment by scrambling the station of each timestep with probability.')
    parser.add_argument('--augment_uniform_noise_scale', type=float, default=0.05,
                        help='the scale of augmentation by uniform noise')

    # misc
    parser.add_argument('--activation', type=str, default='mish',
                        help='activation function: relu, mish')
    parser.add_argument('--is_batch_norm', type=str2bool, nargs='?', const=True, default=True,
                        help='batch normalization')

    # GAT
    parser.add_argument('--is_gat', type=str2bool, nargs='?', const=True, default=True,
                        help='bse graph transformer')
    parser.add_argument('--softmax_temp', type=float, default='7.0', help='only in GCN transformer')
    parser.add_argument('--gcn_n_head', type=int, default=3, help='number of heads in graph transformer')
    parser.add_argument('--spatial_PE_gcn', type=str2bool, nargs='?', const=True, default=True,
                        help='spatial positional embedding')
    parser.add_argument('--is_gcn_attention', type=str2bool, nargs='?', const=True, default=False,
                        help='use attention at the end of GCN')
    parser.add_argument('--gcn_head_aggregate', type=str, default='CAT',
                        help='SUM or conCATenate the final gcn heads')

    args = parser.parse_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} mins")
