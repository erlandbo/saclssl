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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class KNNDinoClassifier():
    def __init__(self, num_classes, feature_type="backbone_features", k=20, metric="cosine", temp=0.07):
        self.metric = metric
        self.k = k
        self.temp = temp
        self.feature_type = feature_type
        self.num_classes = num_classes

    @torch.no_grad()
    def knn_classifier(self, train_features, train_labels, test_features, test_labels, k, T, num_classes):
        # Copied from https://github.com/facebookresearch/dino/blob/main/eval_knn.py
        top1, top5, total = 0.0, 0.0, 0
        train_features = train_features.t()
        num_test_images, num_chunks = test_labels.shape[0], 100
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = test_features[
                       idx : min((idx + imgs_per_chunk), num_test_images), :
                       ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(T).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total += targets.size(0)
        top1 = top1 / total  #  top1 * 100.0 / total
        top5 = top5 / total  # top5 * 100.0 / total
        return top1, top5

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        test_points, test_targets = [], []
        for batch in test_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")

            test_points.append(feature)
            test_targets.append(target)

        test_points = torch.cat(test_points, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        memory_bank = F.normalize(memory_bank, dim=1)
        test_points = F.normalize(test_points, dim=1)

        acc1, acc5 = self.knn_classifier(
            train_features=memory_bank,
            train_labels=target_bank,
            test_features=test_points,
            test_labels=test_targets,
            k=self.k,
            T=self.temp,
            num_classes=self.num_classes
        )
        return acc1


class KNNClassifier():
    def __init__(self, feature_type="backbone_features", k=20, metric="cosine", temp=0.07):
        self.metric = metric
        self.k = k
        self.temp = temp
        self.feature_type = feature_type

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        test_points, test_targets = [], []
        for batch in test_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")

            test_points.append(feature)
            test_targets.append(target)

        test_points = torch.cat(test_points, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        memory_bank = memory_bank.cpu().numpy()
        target_bank = target_bank.cpu().numpy()
        test_points = test_points.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        if self.metric == "cosine":
            def temp_scaled_cossim(cos_distance):
                # Cosine distance is defined as 1.0 minus the cosine similarity.
                # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
                # cos_distance = 1 - cos_sim -> cos_sim = cos_distance * -1 + 1
                cos_sim = -1.0 * cos_distance + 1.0
                return np.exp( cos_sim / self.temp)
            knn = KNeighborsClassifier(n_neighbors=self.k, metric="cosine", n_jobs=-1, algorithm='brute', weights=temp_scaled_cossim)
            knn.fit(memory_bank, target_bank)
            acc = knn.score(test_points, test_targets)
            return acc
        elif self.metric == "euclidean":
            knn = KNeighborsClassifier(n_neighbors=self.k, metric="euclidean", weights="uniform", n_jobs=-1)
            knn.fit(memory_bank, target_bank)
            acc = knn.score(test_points, test_targets)
            return acc
        else:
            raise ValueError("Unknown metric", self.metric)


class LogRegClassifier():
    def __init__(self, feature_type="backbone_features", solver="saga"):
        self.solver = solver
        self.feature_type = feature_type

    @torch.no_grad()
    def evaluate(self, model, train_loader, test_loader):
        model.eval()
        memory_bank, target_bank = [], []
        for batch in train_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")
            memory_bank.append(feature)
            target_bank.append(target)

        memory_bank = torch.cat(memory_bank, dim=0)
        target_bank = torch.cat(target_bank, dim=0)

        test_points, test_targets = [], []
        for batch in test_loader:
            x, target, _ = batch
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output_feature, backbone_feature = model(x)
            if self.feature_type == "output_features":
                feature = output_feature
            elif self.feature_type == "backbone_features":
                feature = backbone_feature
            else:
                raise ValueError("Unknown feature type")

            test_points.append(feature)
            test_targets.append(target)

        test_points = torch.cat(test_points, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        memory_bank = memory_bank.cpu().numpy()
        target_bank = target_bank.cpu().numpy()
        test_points = test_points.cpu().numpy()
        test_targets = test_targets.cpu().numpy()

        logreg = LogisticRegression(solver=self.solver, n_jobs=-1)
        logreg.fit(memory_bank, target_bank)
        acc = logreg.score(test_points, test_targets)
        return acc


def main():

    parser = argparse.ArgumentParser(description='eval sklearn classifiers')

    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--temp', default=0.07, type=float)
    parser.add_argument('--metric', default="cosine", type=str,choices=["cosine", "euclidean"])
    parser.add_argument('--classifier', default="knn", type=str,choices=["knn", "logreg", "knn_dino"])
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--dataset', default='tinyimagenet', type=str, choices=["cifar10", "cifar100", "stl10", "tinyimagenet"])
    parser.add_argument('--feature_type', default='backbone_features', type=str, choices=["backbone_features", "output_features"])
    parser.add_argument('--val_split', default=0.0, type=float)
    parser.add_argument('--model_checkpoint_path', required=True)

    args = parser.parse_args()

    print(args)

    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=args.val_split
    )

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
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
    ).cuda()

    model.load_state_dict(model_checkpoint["model_state_dict"])
    print("Loading model from {}".format(args.model_checkpoint_path))

    model.eval()

    if args.classifier == "knn":
        knn_classifier = KNNClassifier(
            feature_type=args.feature_type,
            k=args.k,
            metric=args.metric,
        )
        acc = knn_classifier.evaluate(model, trainloader, testloader)
        print("{}NN {} Accuracy: {:.2f}%".format(args.k, args.metric, acc * 100.0))

    elif args.classifier == "knn_dino":
        knn_classifier = KNNDinoClassifier(
            feature_type=args.feature_type,
            k=args.k,
            metric=args.metric,
            num_classes=NUM_CLASSES
        )
        acc = knn_classifier.evaluate(model, trainloader, testloader)
        print("{}NN dino {} Accuracy: {:.2f}%".format(args.k, args.metric, acc * 100.0))

    elif args.classifier == "logreg":
        logreg_classifier = LogRegClassifier()
        acc = logreg_classifier.evaluate(model, trainloader, testloader)

        print("LogReg Accuracy: {:.2f}%".format(acc * 100.0))
    else:
        raise ValueError("Unknown classifier type")


if __name__ == "__main__":
    main()
