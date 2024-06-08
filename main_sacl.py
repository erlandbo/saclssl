import torch
from torch.utils.data import DataLoader
import argparse
import sys
import time
import os
import logging
from data import build_dataset
from networks import get_arch2infeatures, SimCLRNet
from criterions.scl import SCL
from criterions.sacl_batchmix import SACLBatchMix
from criterions.sacl import SACL
import torchvision
from torch import nn
from eval_knn import KNNEvaluator
from criterions.simclr import SimCLRLoss
from utils import load_pretrained_weights, save_checkpoint, AvgMetricMeter
import math
import wandb
import numpy as np


def main_train():

    parser = get_main_parser()
    args = parser.parse_args()

    if args.sweep_hparams:
        run = wandb.init()
        print("wandb config", wandb.config)
        args.__dict__.update(wandb.config)

    print("Loaded args", args)
    contrastive_train_dataset, contrastive_val_dataset, contrastive_test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="contrastive_pretrain",
        val_split=args.val_split,
        random_state=args.random_state
    )

    print("SSL Trainsize:{} Valsize:{}".format( len(contrastive_train_dataset), len(contrastive_val_dataset)))

    trainloader = DataLoader(contrastive_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    knn_train_dataset, knn_val_dataset, knn_test_dataset, knn_NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split,
        random_state=args.random_state
    )

    print("KNN Trainsize:{} Valsize:{}".format( len(knn_train_dataset), len(knn_val_dataset)))

    knn_trainloader = DataLoader(knn_train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True, drop_last=False)
    knn_valloader = DataLoader(knn_val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers,pin_memory=True, drop_last=False)

    if args.lr_scale == "linear":
        lr = args.base_lr * args.batch_size / 256.0
    elif args.lr_scale == "squareroot":
        lr = args.base_lr * math.sqrt(args.batch_size / 256.0)
        # lr = 0.075 * args.batch_size ** 0.5
    elif args.lr_scale == "no_scale":
        lr = args.base_lr
    else:
        raise ValueError("Unknown learning rate scale: {}".format(args.lr_scale))

    in_features = get_arch2infeatures(args.arch)
    args.__dict__.update({
            "N": contrastive_train_dataset.__len__(),
            "in_features": in_features,
            "lr": lr
        }
    )

    args.rho = args.N**2.0 / (args.N**2.0 + args.rho_const * args.batch_size)

    args.s_init = args.N**(-2.0) * 10**args.s_init_t

    args.savedir = args.logdir + "/{current_time}_{criterion}_single_s{single_s}_{metric}_{dataset}_{batchsize}_epochs_{epochs}_{arch}_lr{lr}_temp{temp}_alpha{alpha}_rho{rho}_sinit{s_init}_outfeatures{outfeatures}".format(
        current_time=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
        criterion=args.criterion,
        metric=args.metric,
        dataset=args.dataset,
        batchsize=args.batch_size,
        epochs=args.epochs,
        arch=args.arch,
        lr=args.lr,
        single_s=args.single_s,
        outfeatures=args.out_features,
        temp=args.temp,
        alpha=args.alpha,
        s_init=args.s_init,
        rho=args.rho,
    )

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    logging.shutdown()
    logging.basicConfig(format='%(message)s', filename=args.savedir + "/train.log", level=logging.INFO)

    print("Updated args", args)

    # Initialize with updated args
    if args.wandb_logging:
        # run = wandb.init(project=f"{args.criterion}_{args.dataset}", config=args.__dict__)
        project = f"{args.criterion}_{args.dataset}" if args.wandb_project_name == "" else args.wandb_project_name
        run = wandb.init(project=project, config=args.__dict__)

    backbone = torchvision.models.__dict__[args.arch](zero_init_residual=args.zero_init_residual)
    backbone.fc = nn.Identity()
    if args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    if args.backbone_checkpoint_path is not None:
        backbone, in_features = load_pretrained_weights(args.backbone_checkpoint_path)
        args.in_features = in_features

    model = SimCLRNet(
        backbone,
        args.in_features,
        hidden_dim=args.hidden_dim,
        out_features=args.out_features,
        num_layers=args.num_layers,
        norm_hidden_layer=args.norm_hidden_layer,
        bn_last=args.bn_last
    )
    print(model)

    if args.criterion == "scl":
        criterion = SCL(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.criterion == "sacl":
        criterion = SACL(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.criterion == "sacl_batchmix":
        criterion = SACLBatchMix(
            metric=args.metric,
            N=args.N,
            rho=args.rho,
            alpha=args.alpha,
            s_init=args.s_init,
            single_s=args.single_s,
            temp=args.temp
        )
    elif args.criterion == "simclr":
        criterion = SimCLRLoss(
            metric=args.metric,
            temperature=args.temp
        )
    else:
        raise ValueError("Invalid criterion", args.criterion)

    print(criterion)

    for name, buffer in criterion.named_buffers(): print(name, buffer)

    model.cuda()
    criterion.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer", args.optimizer)

    if args.lr_anneal == "cosine":
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        T_max = 1 + args.epochs - args.warmup_epochs
        lr_schedule_warmup = np.linspace(0.0, args.lr, args.warmup_epochs)
        lr_schedule_anneal = 0.5 * args.lr * (1.0 + np.cos(np.pi * np.arange(T_max) / T_max))
        lr_scheduler = np.hstack((lr_schedule_warmup, lr_schedule_anneal))
    elif args.lr_anneal == "linear":
        T_max = 1 + args.epochs - args.warmup_epochs
        lr_schedule_warmup = np.linspace(0.0, args.lr, args.warmup_epochs)
        lr_schedule_anneal = np.linspace(start=args.lr, stop=0.0, num=T_max)
        lr_scheduler = np.hstack((lr_schedule_warmup, lr_schedule_anneal))
    elif args.lr_anneal == "no_anneal":
        lr_schedule_warmup = np.linspace(0.0, args.lr, args.warmup_epochs)
        T_max = (args.epochs + 1 - args.warmup_epochs)
        lr_schedule_anneal = np.zeros(T_max) + args.lr
        lr_scheduler = np.hstack([lr_schedule_warmup, lr_schedule_anneal])
    else:
        raise ValueError("Unknown lr scheduler", args.lr_anneal)

    print("Lr anneal ", args.lr_anneal)
    print(lr_scheduler)
    print(optimizer)

    scaler = torch.cuda.amp.GradScaler() if not args.use_fp32 else None

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume_checkpoint_path is not None:
        resume_checkpoint = torch.load(args.resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        criterion.load_state_dict(resume_checkpoint['criterion_state_dict'])
        lr_scheduler = resume_checkpoint['scheduler_state_dict']
        scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
        start_epoch = resume_checkpoint['epoch']
        best_val_acc = resume_checkpoint['best_val_acc']
        print("Loaded model weights from resume checkpoint from {}".format(args.resume_checkpoint_path))

    if args.freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        print("Freezed backbone weights")

    loss_metric = AvgMetricMeter()

    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        if args.freeze_backbone:
            model.backbone.eval()
            model.projection.train()
        else:
            model.train()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler[epoch]

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            (x1, x2), target, idx = batch

            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)
            idx = idx.cuda(non_blocking=True)

            x = torch.cat([x1, x2], dim=0)

            with torch.cuda.amp.autocast(not args.use_fp32):
                z, _ = model(x)
                loss = criterion(z, idx)

            if not args.use_fp32:
                scaler.scale(loss).backward()

                if args.clip_grad_val is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_val)

                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()

                if args.clip_grad_val is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_val)

                optimizer.step()

            loss_metric.update(loss.item(), n=x.shape[0])

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss")
                wandb.alert(f'Loss is NaN')     # Will alert you via email or slack that your metric has reached NaN
                raise Exception(f'Loss is NaN')  # This could be exchanged for exit(1) if you do not want a traceback

        epoch_loss = loss_metric.compute_global()

        train_stats = {
            "epoch": epoch,
            "lr": lr_scheduler[epoch],
            "train_loss": epoch_loss,
            "rho": criterion.criterion.rho.item() if args.criterion in ["scl", "sacl", "sacl_batchmix"] else 0.0,
            "Z_hat": torch.mean(criterion.criterion.s_inv / criterion.criterion.N.pow(2)).item() if args.criterion in ["scl", "sacl", "sacl_batchmix"] else 0.0,
            "E_attr": criterion.criterion.E_attr.item() if args.criterion in ["scl", "sacl", "sacl_batchmix"] else 0.0,
            "E_rep": criterion.criterion.E_rep.item() if args.criterion in ["scl", "sacl", "sacl_batchmix"] else 0.0,
        }

        if epoch % args.validate_interval == 0:
            val_acc = KNNEvaluator(
                knn_NUM_CLASSES,
                feature_type=args.knn_feature_type,
                k=args.knn_k,
                fx_distance=args.knn_fx_distance,
                weights=args.knn_weights,
                eps=args.knn_eps,
                temp=args.knn_temp
            ).fit_predict(model, knn_trainloader, knn_valloader)
            train_stats["val_acc"] = val_acc
            # Save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, scaler, criterion, lr_scheduler, epoch, best_val_acc, args, filename="/checkpoint_best.pth")

        if args.sweep_hparams or args.wandb_logging:
            wandb.log({key: val for key, val in train_stats.items()})

        train_stats_string = " ".join(("{key}:{val}".format(key=key, val=val)) for key, val in train_stats.items())
        train_stats_string = "time:{} ".format(time.time() - start_time) + train_stats_string
        if not args.sweep_hparams:
            logging.info(train_stats_string)

        print(train_stats_string)

        # Save checkpoint
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scaler, criterion, lr_scheduler, epoch, best_val_acc, args)

        loss_metric.reset()

    # Save last checkpoint
    save_checkpoint(model, optimizer, scaler, criterion, lr_scheduler, epoch, best_val_acc, args, filename="/checkpoint_last.pth")


def get_main_parser():
    parser = argparse.ArgumentParser(description='Pretrain')
    # Model parameters
    parser.add_argument('--arch', default="resnet18", type=str, choices=["resnet18", "resnet50"], help='backbone depth resnet 18, 50')
    parser.add_argument('--hidden_dim', default=2048, type=int, help='hidden dim for projector')
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--out_features', default=128, type=int)
    parser.add_argument('--norm_hidden_layer', default=True, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization last mlp layer')
    parser.add_argument('--bn_last', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--first_conv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--drop_maxpool', default=False, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization last mlp layer')
    parser.add_argument('--zero_init_residual', default=False, action=argparse.BooleanOptionalAction)
    # Finetune
    parser.add_argument('--freeze_backbone', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--backbone_checkpoint_path', type=str, default=None, help="Path to load a pretrained backbone.")
    # Hyperparameters and optimization parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--base_lr', default=0.03, type=float, help='Automatically set from batchsize if None')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5.0e-4, type=float)
    parser.add_argument('--clip_grad_val', default=None, type=float)
    parser.add_argument('--lr_anneal', default="cosine", type=str, choices=["cosine", "linear", "step", "no_anneal"])
    parser.add_argument('--lr_scale', default="linear", type=str, choices=["no_scale", "linear","squareroot"])
    parser.add_argument('--optimizer', default="sgd", type=str)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # Loss function
    parser.add_argument('--metric', default="cosine", type=str, choices=["cauchy", "gaussian", "cosine", "unit_gaussian"])
    parser.add_argument('--criterion', default="sacl_batchmix", type=str, choices=["scl", "sacl", "sacl_batchmix", "simclr"])
    parser.add_argument('--rho_const', default=1.0, type=float, help='constant for rho for scaling automatically from batchsize')
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--s_init_t', default=1.0, type=float)
    parser.add_argument('--single_s', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--temp', default=0.1, type=float, help='Set constant temperature for cosine-similarity and gausssian')

    # Data
    parser.add_argument('--dataset', default='cifar10', type=str, choices=["cifar10", "cifar100", "stl10",  "tinyimagenet"])
    parser.add_argument('--use_fp32', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--random_state', default=42, type=int)
    # Logging and checkpoint
    parser.add_argument('--logdir', type=str, default='logs/pretrain', help='log directory')
    parser.add_argument('--checkpoint_interval', default=100, type=int)
    parser.add_argument('--resume_checkpoint_path', default=None, type=str)
    # kNN validation
    parser.add_argument('--validate_interval', default=100, type=int)
    parser.add_argument('--knn_k', default=200, type=int, help='k-nearest neighbors')
    parser.add_argument('--knn_temp', default=0.5, type=float, help='Temperature cosine similarity exp(cossim/temp)')
    parser.add_argument('--knn_eps', default=1e-8, type=float, help='Epsilon for inverse euclidean weighting')
    parser.add_argument('--knn_fx_distance', default="cosine", type=str,choices=["cosine", "euclidean"], help='Function for computing distance')
    parser.add_argument('--knn_weights', default="distance", type=str,choices=["distance", "uniform"], help='Weights computing distance')
    parser.add_argument('--knn_feature_type', default="backbone_features", type=str,choices=["backbone_features", "output_features"], help='Weights computing distance')
    # Wandb
    parser.add_argument('--sweep_hparams', default=False, action=argparse.BooleanOptionalAction, help="wandb sweep hyperparameters. Must be run from search_wandb.py")
    parser.add_argument('--wandb_project_name', default='', type=str, help="Default use project name: criterion_metric_dataset")
    parser.add_argument('--wandb_logging', default=False, action=argparse.BooleanOptionalAction)

    return parser


if __name__ == "__main__":
    main_train()
