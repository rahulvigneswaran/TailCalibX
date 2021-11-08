"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import yaml
import csv
import h5py
import torch

import libs.utils.globals as g



class Logger(object):
    def __init__(self, logdir):
        if g.log_offline:
            self.logdir = logdir
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            self.cfg_file = os.path.join(self.logdir, 'cfg.yaml')
            self.acc_file = os.path.join(self.logdir, 'acc.csv')
            self.loss_file = os.path.join(self.logdir, 'loss.csv')
            self.ws_file = os.path.join(self.logdir, 'ws.h5')
            self.acc_keys = None
            self.loss_keys = None
            self.logging_ws = False

    def log_cfg(self, cfg):
        if g.log_offline:
            print('===> Saving cfg parameters to: ', self.cfg_file)
            with open(self.cfg_file, 'w') as f:
                yaml.dump(cfg, f)

    def log_acc(self, accs):
        if g.log_offline:
            if self.acc_keys is None and not(g.resume):
                self.acc_keys = [k for k in accs.keys()]
                with open(self.acc_file, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                    writer.writeheader()
                    writer.writerow(accs)
            else:
                if self.acc_keys is None:
                    self.acc_keys = [k for k in accs.keys()]
                with open(self.acc_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                    writer.writerow(accs)
        

    def log_loss(self, losses):
        if g.log_offline:
            valid_losses = losses
            if self.loss_keys is None and not(g.resume):
                self.loss_keys = [k for k in valid_losses.keys()]
                with open(self.loss_file, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                    writer.writeheader()
                    writer.writerow(valid_losses)
            else:
                if self.loss_keys is None:
                    self.loss_keys = [k for k in valid_losses.keys()]
                with open(self.loss_file, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                    writer.writerow(valid_losses)
    
    def log_ws(self, e, ws):
        if g.log_offline:
            mode = 'a' if self.logging_ws else 'w'
            self.logging_ws = True
            
            key = 'Epoch{:02d}'.format(e)
            with h5py.File(self.ws_file, mode) as f:
                g = f.create_group(key)
                for k, v in ws.items():
                    g.create_dataset(k, data=v)
 