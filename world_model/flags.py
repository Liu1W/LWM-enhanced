import jsonargparse
import random
import numpy as np
import torch


def make():
    parser = jsonargparse.ArgumentParser()

    # General arguments
    parser.add_argument("--version", type=str, default="", help="experiment version")
    parser.add_argument(
        "--save_dir", default=None, type=str, help="Local output file name or path."
    )
    parser.add_argument(
        "--seed", default=123, type=int, help="Set the seed for the model and training."
    )
    parser.add_argument(
        "--device", default=0, type=int, help="cuda device ordinal to train on."
    )
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name")
    parser.add_argument("--eval_mode", type=int, default=0, help="evaluation mode")
    parser.add_argument("--debug", type=int, default=0, help="debug mode")

    # Text arguments
    parser.add_argument(
        "--manual",
        type=str,
        choices=["none", "standard", "standardv2", "emma", "direct", "oracle"],
        help="which type of manuals to pass to the model",
    )
    parser.add_argument(
        "--gpt_groundings_path",
        default="chatgpt_groundings/chatgpt_grounding_few_shot.json",
        type=str,
        help="path to chatgpt groundings",
    )

    # World model arguments
    parser.add_argument(
        "--wm_weights_path",
        type=str,
        default=None,
        help="Path to world model state dict.",
    )
    parser.add_argument(
        "--hidden_size", default=256, type=int, help="World model hidden size."
    )
    parser.add_argument(
        "--encoder_num_heads", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--encoder_layers", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--encoder_tokens_per_block", type=int, default=3, help="tokens per blocks"
    )
    parser.add_argument("--encoder_max_blocks", type=int, default=3, help="max blocks")
    parser.add_argument(
        "--decoder_num_heads", type=int, default=4, help="attribute embedding size"
    )
    parser.add_argument(
        "--decoder_layers", type=int, default=4, help="action embedding size"
    )
    parser.add_argument(
        "--decoder_tokens_per_block", type=int, default=15, help="tokens per blocks"
    )
    parser.add_argument("--decoder_max_blocks", type=int, default=33, help="max blocks")
    parser.add_argument(
        "--special_init",
        type=int,
        default=1,
        help="customized parameter initialization",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        default="custom_dataset/wm_data_mixed_100k_train.pickle",
        help="path to the dataset file",
    )

    # Training arguments
    parser.add_argument(
        "--max_rollout_length",
        default=32,
        type=int,
        help="Max length of a rollout to train for",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch_size of training input"
    )
    parser.add_argument(
        "--max_batches", default=100000, type=int, help="max training batches"
    )
    parser.add_argument(
        "--grad_acc_steps",
        default=1,
        type=int,
        help="number of gradient accumu