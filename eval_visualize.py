import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import argparse
from data import build_dataset
import torchvision
from torch import nn
from networks import SimCLRNet
import os


def main():

    parser = argparse.ArgumentParser(description='eval visualization')

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--dataset', default='tinyimagenet', type=str, choices=["cifar10", "cifar100", "stl10", "tinyimagenet"])
    parser.add_argument('--feature_type', default='output_features', type=str, choices=["backbone_features", "output_features"], help='feature type to visualize')

    parser.add_argument('--model_checkpoint_path', required=True, type=str, help="path to model weights")

    args = parser.parse_args()

    print(args)

    # No validation set
    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=0.0
    )

    visualize_dataset = ConcatDataset([train_dataset, test_dataset])
    loader = DataLoader(visualize_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model_checkpoint = torch.load(args.model_checkpoint_path)
    model_args = model_checkpoint['args']

    print("Loading model from {}".format(args.model_checkpoint_path))
    print("Model args: {}".format(model_args))

    backbone = torchvision.models.__dict__[model_args.arch](zero_init_residual=model_args.zero_init_residual)
    backbone.fc = nn.Identity()
    if model_args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if model_args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    model = SimCLRNet(
        backbone,
        model_args.in_features,
        hidden_dim=model_args.hidden_dim,
        out_features=model_args.out_features,
        num_layers=model_args.num_layers,
        norm_hidden_layer=model_args.norm_hidden_layer,
        bn_last=model_args.bn_last
    )

    model.cuda()
    model.load_state_dict(model_checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        feature_bank, target_bank = [], []
        for batch in loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if args.feature_type == "output_features":
                feature = output_feature
            elif args.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type", args.feature_type)

            feature_bank.append(feature)
            target_bank.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).cpu().numpy()
        target_bank = torch.cat(target_bank, dim=0).cpu().numpy()

    assert feature_bank.shape[-1] == 2, "Features must be in 2d for visualization"

    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(40, 40))
    ax.scatter(*feature_bank.T, c=target_bank, cmap="jet")

    filename = args.model_checkpoint_path.split("/")[-2:].replace("/", "").replace(".", "")

    plt.savefig("plots/2dvis_{}.png".format(filename))


if __name__ == "__main__":
    main()
