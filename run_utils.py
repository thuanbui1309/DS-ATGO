"""Lightweight run logging + multi-seed aggregation for reproduce experiments.

Non-invasive: this module only records metrics and computes mean/std across seeds.
It never touches the training dynamics, RNG, or model definition.
"""
import os
import csv
import json
import glob
import subprocess


def git_commit_hash():
    """Best-effort commit hash (with -dirty flag) for provenance; 'unknown' on failure."""
    try:
        h = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


class RunLogger:
    """Per-run logger: appends per-epoch metrics to metrics.csv, writes summary.json."""
    FIELDS = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
              'best_acc', 'best_epoch', 'lr', 'epoch_time_s', 'timestamp']

    def __init__(self, run_dir, resume=False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, 'metrics.csv')
        if resume and os.path.exists(self.csv_path):
            return   # keep existing rows; reconcile_to_epoch() trims any past the checkpoint
        with open(self.csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def reconcile_to_epoch(self, epoch):
        """Drop metrics rows beyond `epoch` so the CSV matches the resumed checkpoint state
        (a row may have been written after the last checkpoint but before the crash)."""
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline='') as f:
                rows = [r for r in csv.DictReader(f) if r.get('epoch') and int(r['epoch']) <= epoch]
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in self.FIELDS})

    def log_epoch(self, row):
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(
                {k: row.get(k, '') for k in self.FIELDS})

    def write_summary(self, summary):
        with open(os.path.join(self.run_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


def _std(xs, ddof=1):
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return var ** 0.5


def aggregate_exp(exp_dir):
    """Scan <exp_dir>/seed_*/summary.json (completed only), write aggregate.json, return dict."""
    summaries = []
    for p in sorted(glob.glob(os.path.join(exp_dir, 'seed_*', 'summary.json'))):
        try:
            with open(p) as f:
                s = json.load(f)
        except Exception:
            continue
        if s.get('completed'):
            summaries.append(s)
    if not summaries:
        return None
    accs = [float(s['best_acc']) for s in summaries]
    mean = sum(accs) / len(accs)
    agg = {
        'exp_dir': exp_dir,
        'num_seeds': len(accs),
        'seeds': [s['seed'] for s in summaries],
        'best_acc_mean': round(mean, 4),
        'best_acc_std': round(_std(accs, ddof=1), 4),   # sample std (ddof=1), matches paper convention
        'best_acc_all': [round(a, 4) for a in accs],
        'per_seed': [{'seed': s['seed'], 'best_acc': s['best_acc'],
                      'best_epoch': s['best_epoch']} for s in summaries],
    }
    with open(os.path.join(exp_dir, 'aggregate.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    return agg
