import torch
import torchvision
from torch import nn
import logging


def build_resnet_backbone(args):
    backbone = torchvision.models.__dict__[args.arch](zero_init_residual=args.zero_init_residual)
    backbone.fc = nn.Identity()
    if args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    return backbone


def load_pretrained_weights(backbone_checkpoint_path):
    pretrained_checkpoint = torch.load(backbone_checkpoint_path)
    backbone_args = pretrained_checkpoint['args']
    backbone = build_resnet_backbone(backbone_args)
    pretrained_model_state_dict = pretrained_checkpoint["model_state_dict"]
    state_dict = {key.replace("backbone.", ""): val for key, val in pretrained_model_state_dict.items() if key.startswith("backbone.")}
    backbone.load_state_dict(state_dict, strict=True)
    return backbone, backbone_args.in_features


def save_checkpoint(model, optimizer, scaler, criterion, lr_scheduler, epoch, best_val_acc,  args, filename="/checkpoint.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler,
        "criterion_state_dict": criterion.state_dict(),
        "scaler_state_dict": scaler.state_dict() if not args.use_fp32 else None,
        "epoch": epoch,
        "args": args,
        "best_val_acc": best_val_acc,
    }, args.savedir + filename)


def save_checkpoint_linear(backbone, linear_classifier, optimizer, scaler, criterion, lr_scheduler, epoch, best_val_acc, args, filename="/checkpoint.pth"):
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "linear_classifier_state_dict": linear_classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler,
        "criterion_state_dict": criterion.state_dict(),
        "scaler_state_dict": scaler.state_dict() if not args.use_fp32 else None,
        "epoch": epoch,
        "args": args,
        "best_val_acc": best_val_acc,
    }, args.savedir + filename)


class AvgMetricMeter:
    def __init__(self):
        self.running_sum = 0.0
        self.total_count = 0.0
        self.update_count = 0.0
        self.values = []

    def update(self, value, n=1.0):
        self.values.append(value)
        self.update_count += 1
        self.running_sum += value * n
        self.total_count += n

    def compute_global(self):
        return self.running_sum / self.total_count

    def compute_samplewise(self):
        return sum(self.values) / self.update_count

    def reset(self):
        self.running_sum = 0
        self.total_count = 0
        self.update_count = 0
        self.values = []


