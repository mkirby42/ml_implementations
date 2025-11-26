import argparse
import logging
import torch
import os
# import wandb
import numpy as np

from tqdm import tqdm
from data.load_data import DataFactory
from trainer import THOCTrainer

import torch
import random
import pandas as pd
from torch.autograd import Variable

def SEED_everything(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

# def read_xlsx_and_convert_to_csv(path):
#     excelFile = pd.read_excel(path, skiprows=[0])
#     filename = path[:-5]
#     excelFile.to_csv(f"{filename}.csv", index=None, header=True)

# from secret import WANDB_API_KEY

# parse arguments
parser = argparse.ArgumentParser(description='THOC-Pytorch')

## data
parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--eval_batch_size", type=int, required=False, default=64 * 3, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required=False, default=100, help=f"window size")
parser.add_argument("--stride", type=int, required=False, default=1, help=f"stride")
parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
parser.add_argument("--exp_id", type=str, default="test")
parser.add_argument("--scaler", type=str, default="std")
parser.add_argument("--window_anomaly", action="store_true", help=f"window-base anomaly")
parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

## save
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--logs", type=str, default="./logs")
parser.add_argument("--outputs", type=str, default="./outputs")

## THOC params
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--L2_reg", type=float, default=1.0)
parser.add_argument("--LAMBDA_orth", type=float, default=1.0)
parser.add_argument("--LAMBDA_TSS", type=float, default=1.0)

args = parser.parse_args()
args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
args.output_path = os.path.join(args.outputs, f"{args.exp_id}")

os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.logging_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)

args.home_dir = "."
args.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
args.model = "THOC"
args.exp_id = f"{args.model}_data_{args.dataset}_lr_{args.lr:.5f}_ws_{args.window_size}_l2_{args.L2_reg}"
args.exp_id += f"_orth_{args.LAMBDA_orth}_tss_{args.LAMBDA_TSS}"

# wandb
# wandb.login(key=WANDB_API_KEY)
# WANDB_PROJECT_NAME, WANDB_ENTITY = "THOC-Pytorch", "carrtesy"
# wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
# wandb.config.update(args)

# Logger
def make_logger(filename, logger_name=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

logger = make_logger(os.path.join(args.logging_path, f'{args.exp_id}.log'))
logger.info(f"Configurations: {args}")

# Data
logger.info(f"Preparing {args.dataset} dataset...")
datafactory = DataFactory(args, logger)
train_dataset, train_loader, test_dataset, test_loader = datafactory()
args.num_channels = train_dataset.x.shape[1]

# Trainer
trainer = THOCTrainer(
    args=args,
    logger=logger,
    train_loader=train_loader,
    test_loader=test_loader,
)

# 4. train
logger.info(f"Preparing {args.model} Training...")
trainer.train()

# 5. infer
logger.info(f"Loading from best path...")
trainer.load(os.path.join(args.checkpoint_path, f"best.pth"))
result = trainer.infer()
# wandb.log(result)
logger.info(f"=== Final Result ===")
for key in result:
    logger.info(f"{key}: {result[key]}")