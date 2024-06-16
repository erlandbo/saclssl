import torch
from torch.utils.data import DataLoader
import argparse
import math
import sys
import time
import os
import logging
from data import build_dataset
from torch import nn
from utils import load_pretrained_weights, save_checkpoint_linear, AvgMetricMeter
import wandb


@torch.no_grad()
def validate(backbone, classifier, criterion, loader):
    loss_metric, acc_metric = AvgMetricMeter(), AvgMetricMeter()
    backbone.eval()
    classifier.eval()
    for batch_idx, batch in enumerate(loader):
        x, target, idx = batch
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            backbone_feats = backbone(x)
        logits = classifier(backbone_feats.detach())

        loss = criterion(logits, target)
        loss_metric.update(loss.item(), n=target.shape[0])

        correct = torch.argmax(logits, dim=1) == target
        acc = torch.sum(correct.float()) / correct.shape[0]
        acc_metric.update(acc.item(), correct.size(0))

    return loss_metric, acc_metric


def main_linear():
    parser = argparse.ArgumentParser(description='eval linear')

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--base_lr', default=30, type=float,help='linearly scale lr with batch-size')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--lr_anneal', default="cosine", choices=["cosine", "no_anneal", "multi_step"])
    parser.add_argument('--optimizer', default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument('--classifier', default="linear", choices=["linear", "mlp"])
    parser.add_argument('--lr_anneal_steps', default=[70, 90], nargs='+', type=int,)
    parser.add_argument('--lr_anneal_steps_constant', default=0.25, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--num_workers', default=20, type=int)

    parser.add_argument('--dataset', default='tinyimagenet', type=str, choices=["cifar10", "cifar100", "tinyimagenet", "stl10"])

    parser.add_argument('--backbone_checkpoint_path', required=True, type=str, help="path to model weights")

    parser.add_argument('--use_fp32', default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument('--validate_interval', default=1, type=int)

    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--wandb_logging', type=bool, default=False)

    args = parser.parse_args()

    args.lr = args.base_lr * args.batch_size / 256.0

    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="train_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split,
        random_state=args.random_state
    )

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    if args.wandb_logging:
        run = wandb.init(project="linear_" + args.dataset, config=args.__dict__)

    print(args)

    # Load backbone
    backbone, in_features = load_pretrained_weights(args.backbone_checkpoint_path)
    print("Loading model from {}".format(args.backbone_checkpoint_path))

    for name, param in backbone.named_parameters():
        param.requires_grad = False

    if args.classifier == "linear":
        linear_classifier = nn.Linear(in_features, NUM_CLASSES)
        linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
        linear_classifier.bias.data.zero_()

        linear_classifier = nn.Sequential(nn.Dropout(p=args.dropout), linear_classifier)
    else:
        # use simple MLP classifier
        # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/models/self_supervised/evaluator.py
        linear_classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, NUM_CLASSES, bias=True),
        )

    print(linear_classifier)

    backbone.cuda()
    linear_classifier.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            linear_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            linear_classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Unknown optimizer", args.optimizer)

    print(optimizer)

    if args.lr_anneal == "cosine":
        T_max = args.epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif args.lr_anneal == "multi_step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_anneal_steps, gamma=args.lr_anneal_steps_constant)
    elif args.lr_anneal == "no_anneal":
        lr_scheduler = None
    else:
        raise ValueError("Unknown lr anneal", args.lr_anneal)

    scaler = torch.cuda.amp.GradScaler() if not args.use_fp32 else None

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    start_epoch = 1

    torch.backends.cudnn.benchmark = True

    loss_metric, acc_metric = AvgMetricMeter(), AvgMetricMeter()

    for epoch in range(start_epoch, args.epochs + 1):

        backbone.eval()
        linear_classifier.train()

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            x, target, _ = batch

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(not args.use_fp32):

                with torch.no_grad():
                    backbone_feats = backbone(x)

                logits = linear_classifier(backbone_feats.detach())
                loss = criterion(logits, target)

            if not args.use_fp32:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_metric.update(loss.item(), x.size(0))
            correct = torch.argmax(logits, dim=1) == target
            acc = torch.sum(correct.float()) / correct.shape[0]
            #train_acc = (torch.argmax(logits, dim=1) == target).float().sum() / target.size(0)
            acc_metric.update(acc.item(), correct.size(0))

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                sys.exit(1)

        train_stats = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": loss_metric.compute_global(), "train_acc": acc_metric.compute_global()}

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Validate
        if args.val_split > 0.0 and epoch % args.validate_interval == 0:
            val_loss_metric, val_acc_metric = validate(backbone, linear_classifier, criterion, valloader)
            train_stats["val_loss"] = val_loss_metric.compute_global()
            train_stats["val_acc"] = val_acc_metric.compute_global()

        if args.wandb_logging:
            wandb.log({key: val for key, val in train_stats.items()})

        print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items()))

        loss_metric.reset()
        acc_metric.reset()

    test_loss_metric, test_acc_metric = validate(
        backbone,
        linear_classifier,
        criterion,
        testloader
    )
    test_stats = {"test_loss": test_loss_metric.compute_global(), "test_acc": test_acc_metric.compute_global()}
    print(" ".join(("{key}:{val}".format(key=key, val=val)) for key, val in test_stats.items()))
    if args.wandb_logging:
        wandb.log({key: val for key, val in test_stats.items()})


if __name__ == "__main__":
    main_linear()

