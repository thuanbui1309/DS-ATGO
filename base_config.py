import os
import time
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Spiking Neural Networks')
parser.add_argument("--seed", type=int, default=2025, help='random seed')
parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers (default: 0)')

parser.add_argument("--data_set", type=str, default='***', help='dataset path')
parser.add_argument("--data_type", type=str, default='CIFAR10', help='dataset type')
parser.add_argument("--network_type", type=str, default='ResNet19', help='network architecture')
parser.add_argument('--data_augment', action='store_true', default=True, help='image augmentation')
parser.add_argument("--output_size", type=int, default=10, help='category')
parser.add_argument("--num_epoch", type=int, default=400, help='train epochs')
parser.add_argument("--batch_size", type=int, default=100, help='mini-batch')
parser.add_argument("--T", type=int, default=2, help='time steps')

parser.add_argument("--loss_function", type=str, default='LabelSmooth', choices=['ce', 'mse', 'LabelSmooth'], help='loss function')
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer')
parser.add_argument("--learning_rate", type=float, default=1e-1, help='learning rate')
parser.add_argument("--lr_scheduler", type=str, default='CosineAnnealingLR', choices=['StepLR', 'CosineAnnealingLR'])

# --- multi-seed harness / logging (added for reproduce; does not change training dynamics) ---
parser.add_argument("--seeds", type=int, nargs='+', default=None,
                    help='list of seeds to run sequentially; overrides --seed when set')
parser.add_argument("--output_dir", type=str, default='runs',
                    help='root dir for per-run logs and checkpoints')
parser.add_argument("--exp_name", type=str, default=None,
                    help='experiment name (default: <data_type>_T<T>)')
parser.add_argument("--save_ckpt", action='store_true', default=True,
                    help='save best-model checkpoint per run (default: on)')
parser.add_argument("--no_save_ckpt", dest='save_ckpt', action='store_false',
                    help='disable checkpoint saving')
parser.add_argument("--aggregate", action='store_true', default=False,
                    help='only aggregate existing seed summaries in the exp dir, no training')
parser.add_argument("--smoke_epochs", type=int, default=0,
                    help='if >0, override num_epoch for a quick smoke test (writes to <exp>_smoke<N>)')
parser.add_argument("--resume", action='store_true', default=True,
                    help='auto-resume from last full-state checkpoint (ckpt_last.pth) if present (default: on)')
parser.add_argument("--no_resume", dest='resume', action='store_false',
                    help='ignore any ckpt_last.pth and train from scratch')
parser.add_argument("--ckpt_every", type=int, default=1,
                    help='write a full-state resumable checkpoint every N epochs (default: 1)')
args = parser.parse_known_args()[0]