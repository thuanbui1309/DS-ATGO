import os
import sys
import time
import json
import tqdm
import torch.utils.data
from datetime import datetime

home_dir = os.getcwd()
sys.path.insert(0, home_dir)

from functional import *
from base_config import args, device
from data_loader import loader
from network_model import ResNet19
from run_utils import RunLogger, aggregate_exp, git_commit_hash


def train_one_run(seed, run_dir):
    """One full training run for a single seed. Training dynamics identical to the
    original DS-ATGO main loop; only logging / checkpointing is added around it."""
    setup_seed(seed)
    snn = ResNet19.ResNet19(args.T, args.output_size)
    snn.to(device)

    loss_function = set_loss_function(args.loss_function)
    optimizer = set_optimizer(args.optimizer, snn, args.learning_rate)
    scheduler = set_lr_scheduler(args.lr_scheduler, optimizer, args.num_epoch)
    train_loader, test_loader = loader.DataLoader(
        args.data_type, args.data_set, args.batch_size, args.data_augment, args.num_workers)

    logger = RunLogger(run_dir)
    num_epoch = args.smoke_epochs if args.smoke_epochs > 0 else args.num_epoch
    start_time = time.time()
    best_acc, best_epoch, test_acc = 0., 0, 0.
    for epoch in range(num_epoch):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        snn.train()
        total, correct, train_loss = 0., 0., 0.
        for images, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            reset_net(snn)
            images, labels = images.to(device), labels.to(device)
            outputs_T = snn(images)
            loss = criterion(outputs_T, labels, loss_function)
            train_loss += loss.cpu().detach().item()
            loss.backward()
            optimizer.step()

            total += labels.numel()
            correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        train_acc = 100. * float(correct / total)
        scheduler.step()

        total, correct, test_loss = 0., 0., 0.
        snn.eval()
        with torch.no_grad():
            for images, labels in tqdm.tqdm(test_loader):
                reset_net(snn)
                images, labels = images.to(device), labels.to(device)
                outputs_T = snn(images)
                loss = criterion(outputs_T, labels, loss_function)
                test_loss += loss.cpu().detach().item()

                total += labels.numel()
                correct += (outputs_T.mean(dim=1).argmax(dim=1) == labels).float().sum().item()
        test_acc = 100. * float(correct / total)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        if test_acc >= best_acc:
            best_acc, best_epoch = test_acc, epoch + 1
            if args.save_ckpt:
                torch.save({'seed': seed, 'epoch': best_epoch, 'best_acc': best_acc,
                            'state_dict': snn.state_dict()}, os.path.join(run_dir, 'best.pth'))

        tp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print('%s | seed %d | Epoch [%d/%d] train_loss %.4f train_acc %.2f | '
              'test_loss %.4f test_acc %.2f | best %.2f@%d | %.0fs (elapsed %.0fh%.0fm)'
              % (tp, seed, epoch + 1, num_epoch, train_loss, train_acc, test_loss, test_acc,
                 best_acc, best_epoch, epoch_time, elapsed // 3600, (elapsed % 3600) // 60))

        logger.log_epoch({
            'epoch': epoch + 1, 'train_loss': round(train_loss, 6), 'train_acc': round(train_acc, 4),
            'test_loss': round(test_loss, 6), 'test_acc': round(test_acc, 4),
            'best_acc': round(best_acc, 4), 'best_epoch': best_epoch, 'lr': lr,
            'epoch_time_s': round(epoch_time, 2), 'timestamp': tp})

    total_time = time.time() - start_time
    summary = {
        'seed': seed, 'data_type': args.data_type, 'T': args.T, 'network': args.network_type,
        'num_epoch': num_epoch, 'smoke': args.smoke_epochs > 0,
        'best_acc': round(best_acc, 4), 'best_epoch': best_epoch,
        'final_test_acc': round(test_acc, 4),
        'total_time_s': round(total_time, 1),
        'mean_epoch_time_s': round(total_time / max(num_epoch, 1), 2),
        'git_commit': git_commit_hash(), 'completed': True,
        'config': {k: getattr(args, k) for k in vars(args)},
    }
    logger.write_summary(summary)
    return summary


def _print_aggregate(agg):
    if not agg:
        print('[aggregate] no completed seeds found.')
        return
    print('seeds=%s  best_acc = %.2f ± %.2f  (all: %s)' % (
        agg['seeds'], agg['best_acc_mean'], agg['best_acc_std'],
        ', '.join('%.2f' % a for a in agg['best_acc_all'])))


def main():
    exp_name = args.exp_name or ('%s_T%d' % (args.data_type, args.T))
    if args.smoke_epochs > 0:
        exp_name += '_smoke%d' % args.smoke_epochs   # isolate smoke runs from real ones
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if args.aggregate:
        print('\n===== AGGREGATE %s =====' % exp_dir)
        _print_aggregate(aggregate_exp(exp_dir))
        return

    seeds = args.seeds if args.seeds else [args.seed]
    for seed in seeds:
        run_dir = os.path.join(exp_dir, 'seed_%d' % seed)
        summ_path = os.path.join(run_dir, 'summary.json')
        if os.path.exists(summ_path):
            try:
                with open(summ_path) as f:
                    if json.load(f).get('completed'):
                        print('[skip] seed %d already completed -> %s' % (seed, run_dir))
                        continue
            except Exception:
                pass  # partial/corrupt summary -> rerun this seed
        print('\n===== RUN seed %d -> %s =====' % (seed, run_dir))
        train_one_run(seed, run_dir)

    print('\n===== AGGREGATE %s =====' % exp_dir)
    _print_aggregate(aggregate_exp(exp_dir))


if __name__ == '__main__':
    print(' Arguments: ')
    for arg in vars(args):
        print('\t {:25} : {}'.format(arg, getattr(args, arg)))
    main()
