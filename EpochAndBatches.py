import os
import pickle
import time
from pprint import pformat

import pandas as pd


class EpochAndBatches:
    """
    verbose:
    0 = none
    1 = per run
    10 = per epoch
    20 = per batch
    90 = everything
    """

    def __init__(self, n_epoch, n_batch, project_name='Default Project', sweep_name='Default Sweep', run_name=None,
                 enb_output_path=''):
        self.project_name = project_name
        self.sweep_name = sweep_name
        self.l_log = []
        self.dflog = pd.DataFrame()
        self.script_start_time = time.time()
        self.pid = os.getpid()
        self.cwd = os.getcwd()
        self.verbose = 15
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.i_epoch = -1
        self.i_batch = -1
        self.is_save_per_log = False
        self.is_save_per_batch = False
        self.is_save_per_epoch = True
        if run_name is None:
            self.run_name = 'default_run_' + str(int(self.script_start_time)) + '_' + str(self.pid)
        else:
            self.run_name = run_name
        self.enb_output_path = enb_output_path
        self.enb_output_filepath = self.enb_output_path + self.run_name + '.enb'
        self.log_msg('EnB: Epoch And Batches START', i_verbose=0)
        return

    def save(self):
        self.dflog = pd.DataFrame.from_records(self.l_log)
        if self.verbose >= 92:
            print('EnB saving to ', self.enb_output_filepath)
        with open(self.enb_output_filepath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        if self.verbose >= 91:
            print('EnB saved to ', self.enb_output_filepath)
        return

    @staticmethod
    def load(enb_output_filepath):
        """
        usage:
        enb2 = EnB.load(enb.enb_output_filepath)
        :param enb_output_filepath:
        :return:
        """
        print('EnB loading ', enb_output_filepath, ' ...')
        with open(enb_output_filepath, 'rb') as f:
            loaded_enb = pickle.load(f)
        print('EnB ', enb_output_filepath, ' loaded.')
        return loaded_enb

    def log(self, i_dict, i_verbose=1):
        """
        i_dict={
            label,key,value
            [,i_epoch,i_batch]
        }
        """
        self.l_log.append(i_dict)
        # self.dflog = self.dflog.append(i_dict, ignore_index=True)
        if self.verbose >= i_verbose:
            if 'key' in i_dict.keys():
                str_print = 'EnB '
                if 'label' in i_dict.keys() and (i_dict['label'] == 'per_epoch' or i_dict['label'] == 'per_batch'):
                    str_print += 'i_epoch=' + str(i_dict['i_epoch']) + ' '
                if 'label' in i_dict.keys() and i_dict['label'] == 'per_batch':
                    str_print += 'i_batch=' + str(i_dict['i_batch']) + ' '
                str_print += '| ' + str(i_dict['key']) + ' = ' + str(i_dict['value'])
                print(str_print)
            else:
                print(i_dict)
        if self.is_save_per_log:
            self.save()
        return

    def log_msg(self, msg, label='msg', i_verbose=1):
        self.log({'label': label,
                  'key': 'msg',
                  'value': msg,
                  'i_epoch': self.i_epoch,
                  'i_batch': self.i_batch,
                  'time': time.time()},
                 i_verbose)
        return

    def next_epoch(self, force_log=False):
        # call at the start of each epoch
        self.i_epoch += 1
        self.i_batch = -1
        t_i = time.time()
        if self.i_epoch == 0:
            self.fit_start_time = t_i
        t_elapsed = t_i - self.fit_start_time
        t_total = t_elapsed * self.n_epoch / self.i_epoch if self.i_epoch > 0 else 0
        t_left = t_total - t_elapsed
        t_each = t_elapsed / self.i_epoch if self.i_epoch > 0 else 0
        if self.verbose >= 11 or force_log:
            self.log_msg('next_epoch: ' +
                         str(self.i_epoch) + '/' + str(self.n_epoch) +
                         '  |  ' + str(time.strftime('%H:%M:%S', time.gmtime(t_elapsed))) +
                         ' + ' + str(time.strftime('%H:%M:%S', time.gmtime(t_left))) +
                         ' = ' + str(time.strftime('%H:%M:%S', time.gmtime(t_each))) +
                         ' / ' + str(time.strftime('%H:%M:%S', time.gmtime(t_total))))
        self.log_epoch('time:elapsed', t_elapsed, 31)
        self.log_epoch('time:total', t_total, 31)
        self.log_epoch('time:left', t_left, 31)
        self.log_epoch('time:each', t_each, 31)
        if self.is_save_per_epoch:
            self.save()
        return

    def next_batch(self, force_log=False):
        # call at the start of each batch
        self.i_batch += 1
        t_i = time.time()
        if self.i_batch == 0:
            self.i_batch_start_time = t_i
        t_elapsed = t_i - self.i_batch_start_time
        t_total = t_elapsed * self.n_batch / self.i_batch if self.i_batch > 0 else 0
        t_left = t_total - t_elapsed
        t_each = t_elapsed / self.i_batch if self.i_batch > 0 else 0
        if self.verbose >= 21 or force_log:
            self.log_msg('next_batch: ' +
                         'epoch=' + str(self.i_epoch) + '/' + str(self.n_epoch) +
                         ' batch=' + str(self.i_batch) + '/' + str(self.n_batch) +
                         '  |  ' + str(time.strftime('%H:%M:%S', time.gmtime(t_elapsed))) +
                         ' + ' + str(time.strftime('%H:%M:%S', time.gmtime(t_left))) +
                         ' = ' + str(time.strftime('%H:%M:%S', time.gmtime(t_each))) +
                         ' / ' + str(time.strftime('%H:%M:%S', time.gmtime(t_total))))
        self.log_batch('time:elapsed', t_elapsed, 32)
        self.log_batch('time:total', t_total, 32)
        self.log_batch('time:left', t_left, 32)
        self.log_batch('time:each', t_each, 32)
        if self.is_save_per_batch:
            self.save()
        return

    def log_epoch(self, k, v, i_verbose=17):
        self.log({'label': 'per_epoch',
                  'i_epoch': self.i_epoch,
                  'key': k,
                  'value': v,
                  'time': time.time()},
                 i_verbose)
        return

    def log_batch(self, k, v, i_verbose=27):
        self.log({'label': 'per_batch',
                  'i_epoch': self.i_epoch,
                  'i_batch': self.i_batch,
                  'key': k,
                  'value': v,
                  'time': time.time()},
                 i_verbose)
        return

    def __repr__(self):
        return 'Epoch And Batches\n' + pformat(vars(self))
