# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict, deque
import os

import torch

from .comm import is_main_process

class SmoothedValue(object):

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):

    def __init__(self, delimiter="\t", meter_creator=SmoothedValue):
        self.meters = defaultdict(meter_creator)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f} ({:.4f})".format(
                name, meter.median, meter.global_avg))
        return self.delimiter.join(loss_str)


class TensorboardLogger(MetricLogger):

    def __init__(self, log_dir, delimiter='\t', philly_log_dir=None):
        super(TensorboardLogger, self).__init__(delimiter)
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('To use tensorboard please install tensorboardX '
                              '[ pip install tensorboardx ].')
        self.philly_tb_logger = None
        self.philly_tb_logger_avg = None
        self.philly_tb_logger_med = None
        if is_main_process():
            self.tb_logger = SummaryWriter(log_dir)
            self.tb_logger_avg = SummaryWriter(os.path.join(log_dir, 'avg'))
            self.tb_logger_med = SummaryWriter(os.path.join(log_dir, 'med'))
            if philly_log_dir is not None:
                self.philly_tb_logger = SummaryWriter(philly_log_dir)
                self.philly_tb_logger_avg = SummaryWriter(
                    os.path.join(philly_log_dir, 'avg'))
                self.philly_tb_logger_med = SummaryWriter(
                    os.path.join(philly_log_dir, 'med'))
        else:
            self.tb_logger = None
            self.tb_logger_avg = None
            self.tb_logger_med = None

    def get_logs(self, iteration):
        if self.tb_logger:
            for group_name, values in self.meters.items():
                for name, meter in values.items():
                    self.tb_logger.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.last_value,
                        iteration,
                    )
                    self.tb_logger_avg.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.avg,
                        iteration,
                    )
                    self.tb_logger_med.add_scalar(
                        '{}/{}'.format(group_name, name),
                        meter.median,
                        iteration,
                    )
                    if self.philly_tb_logger:
                        self.philly_tb_logger.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.last_value,
                            iteration,
                        )
                        self.philly_tb_logger_avg.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.avg,
                            iteration,
                        )
                        self.philly_tb_logger_med.add_scalar(
                            '{}/{}'.format(group_name, name),
                            meter.median,
                            iteration,
                        )
            for group_name, values in self.params.items():
                for name, param in values.items():
                    self.tb_logger.add_scalar(
                        '{}/{}'.format(group_name, name),
                        param,
                        iteration,
                    )
                    if self.philly_tb_logger:
                        self.philly_tb_logger.add_scalar(
                            '{}/{}'.format(group_name, name),
                            param,
                            iteration,
                        )
        return super(TensorboardLogger, self).get_logs(iteration)

    def close(self):
        if is_main_process():
            self.tb_logger.close()
            self.tb_logger_avg.close()
            self.tb_logger_med.close()
            if self.philly_tb_logger:
                self.philly_tb_logger.close()
                self.philly_tb_logger_avg.close()
                self.philly_tb_logger_med.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count