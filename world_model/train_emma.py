import os
import sys

sys.path.append("..")
import pprint
import wandb
import time

from collections import defaultdict

import numpy as np
import torch

torch.backends.cudnn.deterministic = True

from downstream import ImitationTrainer, Evaluator, ConvEMMA
from messenger.envs import make_env
from train_wm import make_dataset
import flags


def make_policy(args):
    if args.emma_policy.base_arch == "conv":
        policy = ConvEMMA().to(args.device)
    else:
        print("Architecture not supported!")
        sys.exit(1)

    if args.emma_policy.weights_path is not None:
        policy.load_state_dict(torch.load(args.emma_policy.weights_path))
        print(f"Loaded model from {args.emma_policy.weights_path}")

    return policy


def train(
    args,
    trainer,
    evaluator,
    dataset,
    policy,
    train_env,
    eval_env,
    optimizer,
    eval_episodes=24,
):
    best_eval_metric = defaultdict(lambda: defaultdict(lambda: None))
    train_metric = defaultdict(list)
    for i in range(args.max_batches):
        if i % args.log_every_batches == 0:
            # logging
            wandb_stats = {}
            wandb_stats["step"] = i
            wandb_stats["lr"] = optimizer.param_groups[0]["lr"]
            log_str = []
            for k in train_metric:
                train_metric[k] = np.average(train_metric[k])
                wandb_stats[f"train/{k}"] = train_metric[k]
                log_str.append(f"{k} {train_metric[k]:.2f}")
            log_str = ", ".join(log_str)
            print()
            print("After %d batches" % i)
            print("  TRAIN", log_str)
            print()

            # reset train metric
            train_metric = defaultdict(list)

            print("  EVALUATION")
            for split in dataset:
                eval_metric = evaluator.evaluate(
                    policy,
                    eval_env,
                    eval_episo