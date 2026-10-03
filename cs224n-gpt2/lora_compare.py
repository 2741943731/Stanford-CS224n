'''
Small-scale controlled comparison: full fine-tuning vs LoRA (q/v) on paraphrase detection.

This script is experiment scaffolding only -- the LoRA implementation itself lives in
modules/lora.py and is used unchanged here.

Usage (CPU, a few minutes):
  python lora_compare.py --train_size 2000 --dev_size 200 --steps 250 --batch_size 8 --lr 1e-4

Run a single mode:
  python lora_compare.py --modes full
'''

import argparse
import gc
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_paraphrase_data, ParaphraseDetectionDataset
from evaluation import model_eval_paraphrase
from modules.lora import apply_lora
from optimizer import AdamW
from paraphrase_detection import ParaphraseGPT, add_arguments


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def build_model(args, mode):
  '''Build a fresh pretrained model and apply the given parameterisation.'''
  model = ParaphraseGPT(args)
  if mode == 'lora':
    apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
               lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules)
  elif mode == 'tied_head':
    # Closest analogue of "last linear layer": only the weight-tied output layer moves.
    for p in model.parameters():
      p.requires_grad = False
    model.gpt.word_embedding.weight.requires_grad = True
  elif mode == 'full':
    pass
  else:
    raise ValueError(f'unknown mode: {mode}')
  return model


def build_loaders(args):
  '''Rebuilt per run so that every mode sees the same shuffle order.'''
  seed_everything(args.seed)
  train_raw = load_paraphrase_data(args.train_path)[:args.train_size]
  dev_raw = load_paraphrase_data(args.dev_path)[:args.dev_size]
  train_ds = ParaphraseDetectionDataset(train_raw, args)
  dev_ds = ParaphraseDetectionDataset(dev_raw, args)
  train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=train_ds.collate_fn)
  dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                      collate_fn=dev_ds.collate_fn)
  return train_dl, dev_dl


def run_mode(args, mode):
  train_dl, dev_dl = build_loaders(args)
  model = build_model(args, mode).to(args.device)

  trainable = [p for p in model.parameters() if p.requires_grad]
  n_trainable = sum(p.numel() for p in trainable)
  names = [n for n, p in model.named_parameters() if p.requires_grad]
  assert n_trainable > 0, f'{mode}: nothing to train'
  print(f'\n=== mode={mode} | trainable params = {n_trainable:,} '
        f'| tensors = {len(names)} | first = {names[0]}')

  optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.)
  model.train()

  losses, step_times = [], []
  step = 0
  t0 = time.time()
  stop = False
  while not stop:
    for batch in train_dl:
      b_ids, b_mask = batch['token_ids'].to(args.device), batch['attention_mask'].to(args.device)
      labels = batch['labels'].flatten().to(args.device)

      t_step = time.time()
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels)
      loss.backward()
      optimizer.step()
      step_times.append(time.time() - t_step)

      losses.append(loss.item())
      step += 1
      if step % args.log_every == 0:
        recent = sum(losses[-args.log_every:]) / args.log_every
        print(f'  step {step:>4d}  loss(avg {args.log_every}) = {recent:.3f}  '
              f'step_time = {statistics.median(step_times[-args.log_every:]):.2f}s')
      if step >= args.steps:
        stop = True
        break
  train_seconds = time.time() - t0

  model.eval()
  acc, f1, *_ = model_eval_paraphrase(dev_dl, model, args.device)

  result = {
    'mode': mode,
    'trainable': n_trainable,
    'opt_mem_mb': 2 * n_trainable * 4 / 1e6,
    'step_median': statistics.median(step_times),
    'train_seconds': train_seconds,
    'loss_first': sum(losses[:20]) / min(20, len(losses)),
    'loss_last': sum(losses[-20:]) / min(20, len(losses)),
    'dev_acc': acc,
    'dev_f1': f1,
    'ckpt_mb': 2 * n_trainable * 4 / 1e6,
  }

  del model, optimizer, trainable
  gc.collect()
  return result


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_path', type=str, default='data/quora-train.csv')
  parser.add_argument('--dev_path', type=str, default='data/quora-dev.csv')
  parser.add_argument('--train_size', type=int, default=2000)
  parser.add_argument('--dev_size', type=int, default=200)
  parser.add_argument('--steps', type=int, default=250)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--seed', type=int, default=11711)
  parser.add_argument('--log_every', type=int, default=25)
  parser.add_argument('--modes', type=str, nargs='+',
                      default=['full', 'lora'], choices=['full', 'lora', 'tied_head'])
  parser.add_argument('--lora_r', type=int, default=8)
  parser.add_argument('--lora_alpha', type=int, default=16)
  parser.add_argument('--lora_dropout', type=float, default=0.0)
  parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['query', 'value'])
  parser.add_argument('--model_size', type=str, default='gpt2',
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
  return parser.parse_args()


def main():
  args = get_args()
  args = add_arguments(args)
  args.device = torch.device('cpu')
  print(f'train={args.train_size} examples, steps={args.steps}, batch={args.batch_size}, '
        f'lr={args.lr}, device={args.device}')

  results = [run_mode(args, mode) for mode in args.modes]

  header = f'{"mode":>10s} {"trainable":>12s} {"opt m+v(MB)":>11s} {"median step":>11s} ' \
           f'{"loss(first20)":>13s} {"loss(last20)":>12s} {"dev acc":>8s} {"dev F1":>7s}'
  print('\n' + header)
  print('-' * len(header))
  for r in results:
    print(f'{r["mode"]:>10s} {r["trainable"]:>12,} {r["opt_mem_mb"]:>11.1f} '
          f'{r["step_median"]:>10.2f}s {r["loss_first"]:>13.3f} {r["loss_last"]:>12.3f} '
          f'{r["dev_acc"]:>8.3f} {r["dev_f1"]:>7.3f}')


if __name__ == '__main__':
  main()
