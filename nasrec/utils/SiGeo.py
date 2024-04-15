from itertools import count
import time
import copy

# Other imports
from typing import Union, Any, Optional, List, Dict
from collections import defaultdict
from math import sqrt

import numpy as np
from nasrec.supernet.modules import (
    DotProduct,
    ElasticLinear,
    SigmoidGating,
    Sum,
    Transformer,
)
import sklearn.metrics
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from nasrec.supernet.supernet import SuperNet
from fvcore.nn import FlopCountAnalysis


def compute_zero_shot_metric(
        model,
        optimizer: Any,
        lr_scheduler,
        train_loader,
        test_loader,
        loss_fn,
        l2_loss_fn,
        train_batch_size: int,
        gpu: Union[int, None],
        display_interval: int = 100,
        test_interval: int = 2000,
        grad_clip_value: float = None,
        use_amp: bool = False,
        eval_batch: int = 4,
        zico_coef: float = 1,
        fr_coef: float = 1,
        loss_coef: float = 1,
):
    """Train & Test the model for 1 epoch on the training data pipeline.
    Testing process is inserted in-between.
    Args:
        :params model: Backbone model.
        :params optimizer: Optimizer to optimize the model.
        :params train_loader: Training data loader.
        :params test_loader: Testing data loader.
        :params loss_fn: Loss function.
        :params l2_loss_fn: Loss function for L2 regularization.
        :params train_batch_size: Training batch size.
        :params gpu: GPU ID to use.
        :params display_interval: Interval of displaying training results.
        :params test_interval: Interval of carrying the testing to get test auroc.
        :params summary_writer_log_dir: The path to log tensorboard summary.
        :params grad_clip_value: Gradient clipping by norm.
        :params use_amp: Whether use mixed precision (experimental).
        :params eval_batch: Number of evaluation for zero-shot metrics.
        :params zico_coef: the first term in SiGeo
        :params fr_coef: the second term in SiGeo
        :params loss_coef: the last term in SiGeo
    """
    # Get test batch size from test loader.
    _, _, y_val = next(iter(test_loader))
    test_batch_size = y_val.size(0)

    model.train()
    start_batch = time.time()
    y_pred = []
    y_true = []
    test_auroc = 1,

    logs = {
        "test_loss": [],
        "test_AUROC": [],
        "test_Accuracy": [],
        "epoch": [],
        "iters": [],
    }

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    gradients = {}
    grad_dict_list = []
    for batch_num, (int_x, cat_x, y) in enumerate(train_loader):
        int_x = int_x.to(gpu, non_blocking=True)
        cat_x = cat_x.to(gpu, non_blocking=True)
        y = y.to(gpu, non_blocking=True)
        # Do a vanilla forward pass.
        start_gpu = time.time()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            res = model(int_x, cat_x)
            loss = loss_fn(res, y)
            l2_loss = l2_loss_fn(model)
            total_loss = loss + l2_loss

        # Drop the last batch which may contain insufficient examples.
        if len(y) == train_batch_size:
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                # Clip gradients.
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                # Update scale
                scaler.update()
                # optimizer.step()
            else:
                total_loss.backward()
                if grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
        lr_scheduler.step()
        # print(f"batch num: {batch_num}; collect param: {batch_num == eval_batch - 1}")
        theta_dict = collect_gradients(model, gradients, batch_num == eval_batch - 1)
        if batch_num >= eval_batch - 1:
            break

    score1, score2 = caculate_fr_zico(gradients, theta_dict)
    print(f"zico: {score1.item()}, fr: {score2.item()}, train loss: {total_loss.item()}")
    logs["test_loss"].append(- zico_coef * score1.item() - fr_coef * score2.item() + loss_coef * total_loss.item())
    logs["test_AUROC"].append(-1)
    logs["test_Accuracy"].append(-1)

    print("zero-shot score: {}".format(score1 + score2 - total_loss.item()))
    print(
        "total evaluation time for a subnet: {:.5f} (s)".format(
            time.time() - start_batch
        )
    )
    return logs


def caculate_fr_zico(grad_dict, theta_dict, theta_dict_copy=None):
    nsr_mean_sum_abs, nsr_sum_grad = 0, 0
    for mod in grad_dict.keys():
        gradient_vector = torch.stack(grad_dict[mod])
        nsr_std = torch.std(gradient_vector, axis=0)
        nonzero_idx = torch.nonzero(nsr_std)[0]
        nsr_mean_abs = torch.mean(torch.abs(gradient_vector), axis=0)
        tmpsum = torch.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += torch.log(tmpsum)
        nsr_sum_grad += torch.sum(torch.sum(gradient_vector, axis=0) * theta_dict[mod])

    fr_mean_sum_abs = torch.abs(nsr_sum_grad)
    return nsr_mean_sum_abs, torch.log(fr_mean_sum_abs + 1e-10)


def collect_gradients(module, gradients, collect_param=False):
    parameters = {} if collect_param else None
    children = dict(module.named_children())
    module_name = str(module)
    if children == {}:
        for name, param in module.named_parameters():
            if param.grad is not None:
                param_numpy = param.grad.data.flatten()
                if module_name + str(param_numpy.shape) in gradients:
                    gradients[module_name + str(param_numpy.shape)].append(param_numpy)
                else:
                    gradients[module_name + str(param_numpy.shape)] = [param_numpy]
                if collect_param:
                    parameters[module_name + str(param_numpy.shape)] = param.data.flatten()
    else:
        for name, sub_module in module.named_children():
            _parameters = collect_gradients(sub_module, gradients, collect_param)
            if collect_param:
                parameters.update(_parameters)
    return parameters
