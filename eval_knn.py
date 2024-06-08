import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
from data import build_dataset
import torchvision
from torch import nn
from networks import SimCLRNet
from utils import AvgMetricMeter
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class KNNEvaluator():
    """
    https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/callbacks/knn_online.py
    Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        """
    def __init__(self, num_classes, feature_type="backbone_features", k=20, fx_distance="cosine", weights="distance", eps=1e-8, temp=0.5):
        self.num_classes = num_classes
        self.fx_distance = fx_distance
        self.k = k
        self.temp = temp
        self.weights = weights
        self.eps = eps
        self.feature_type = feature_type

    def predict(self, points_data, points_target, memory_bank, target_bank):
        B = points_data.shape[0]  # effective batchsize

        if self.fx_distance == 'cosine':
            sim_mat = torch.mm(points_data, memory_bank.T)  # (B,N)
            distances_neighbours, idx_neighbours = sim_mat.topk(k=self.k, dim=1)  # (B, k)
            sim_neighbours = torch.exp(distances_neighbours / self.temp)
        elif self.fx_distance == 'euclidean':
            sim_mat = 1.0 / (torch.cdist(points_data, memory_bank).pow(2) + self.eps)  # (B,N)
            sim_neighbours, idx_neighbours = sim_mat.topk(k=self.k, dim=1)  # (B, k)

        neighbours_targets = torch.gather(target_bank.expand(B, -1), dim=-1, index=idx_neighbours)  # (N,) -> (B,N) -> (B,k)
        neighbours_one_hot = torch.zeros(B * self.k, self.num_classes, device=points_data.device)  # (B*k, C)
        class_count_neighbours = torch.scatter( neighbours_one_hot, dim=1, index=neighbours_targets.view(-1, 1), value=1.0).view(B, self.k, self.num_classes)  # (B,k,C)
        if self.weights == "distance":
            y_prob = torch.sum( class_count_neighbours * sim_neighbours.view(B, self.k, -1) , dim=1)  # bcast sum( (B,k,C) * (B,k,1), dim=1) -> (B,C)
        elif self.weights == "uniform":
            y_prob = torch.sum( class_count_neighbours, dim=1)  # bcast sum( (B,k,C) , dim=1) -> (B,C)
        y_pred = torch.argmax(y_prob, dim=1)  # (B,)
        # y_preds = torch.argsort(y_prob, dim=1, descending=True)  # (B,)
        correct = torch.sum(y_pred == points_target).item()
        return correct

    @torch.no_grad()
    def fit_predict(self, model, memory_loader, points_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in memory_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            else:
                feature = backbone_feature
            if self.fx_distance == "cosine":
                feature = F.normalize(feature, dim=1)
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        acc_metric = AvgMetricMeter()

        for batch in points_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            else:
                feature = backbone_feature
            if self.fx_distance == "cosine":
                feature = F.normalize(feature, dim=1)
            correct = self.predict(points_data=feature, points_target=target, memory_bank=memory_bank, target_bank=target_bank)
            acc_metric.update(correct / target.shape[0], n=target.shape[0])

        top1 = acc_metric.compute_global()

        return top1


def main():

    parser = argparse.ArgumentParser(description='eval knn')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--nn_k', default=200, type=int, help='k-nearest neighbors')
    parser.add_argument('--temp', default=0.5, type=float, help='Temperature cosine similarity exp(cossim/temp)')
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon for inverse euclidean weighting')
    parser.add_argument('--fx_distance', default="cosine", type=str,choices=["cosine", "euclidean"], help='Function for computing distance')
    parser.add_argument('--weights', default="distance", type=str,choices=["distance", "uniform"], help='Weights computing distance')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--dataset', default='tinyimagenet', type=str, choices=["cifar10", "cifar100", "stl10", "tinyimagenet"])
    parser.add_argument('--feature_type', default='backbone_features', type=str, choices=["backbone_features", "output_features"])

    parser.add_argument('--model_checkpoint_path', required=True, type=str, help="path to model weights")

    #parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

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

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model_checkpoint = torch.load(args.model_checkpoint_path)
    model_args = model_checkpoint['args']
    print("Model args: {}".format(model_args))

    backbone = torchvision.models.__dict__[model_args.arch](zero_init_residual=model_args.zero_init_residual)
    backbone.fc = nn.Identity()
    if model_args.first_conv:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
    if model_args.drop_maxpool:
        backbone.maxpool = nn.Identity()

    in_features = model_args.in_features

    model = SimCLRNet(
        backbone,
        in_features=in_features,
        hidden_dim=model_args.hidden_dim,
        out_features=model_args.out_features,
        num_layers=model_args.num_layers,
        norm_hidden_layer=model_args.norm_hidden_layer,
        bn_last=model_args.bn_last
    )

    model.cuda()

    model.load_state_dict(model_checkpoint["model_state_dict"])
    print("Loading model from {}".format(args.model_checkpoint_path))

    model.eval()

    knn_classifier = KNNEvaluator(
        num_classes=NUM_CLASSES,
        feature_type=args.feature_type,
        k=args.nn_k,
        fx_distance=args.fx_distance,
        weights=args.weights,
        eps=args.eps,
        temp=args.temp
    )
    acc = knn_classifier.fit_predict(model, trainloader, testloader)

    print("kNN Accuracy: {:.2f}%".format(acc * 100.0))


if __name__ == "__main__":
    main()
