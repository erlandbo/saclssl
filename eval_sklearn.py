import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from data import build_dataset
import torchvision
from torch import nn
from networks import SimCLRNet
import os


def to_np_features(model, loader, feature_type="backbone_features"):
    X, y = [], []
    for batch in loader:
        x, target, _ = batch
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output_feature, backbone_feature = model(x)
        if feature_type == "output_feature":
            feature = output_feature
        else:
            feature = backbone_feature
        X.append(feature)
        y.append(target)
    return torch.cat(X, dim=0).cpu().numpy(), torch.cat(y, dim=0).cpu().numpy()


def main():

    parser = argparse.ArgumentParser(description='eval sklearn')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dataset', default='tinyimagenet', type=str, choices=["cifar10", "cifar100", "stl10_unlabeled", "stl10_labeled", "imagenette", "tinyimagenet"])
    parser.add_argument('--feature_type', default='backbone_features', type=str, choices=["backbone_features", "output_features"], help='feature type to visualize')

    parser.add_argument('--model_checkpoint_path', required=True, type=str, help="path to model weights")

    parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--nn_k', default=5, type=int)
    parser.add_argument('--random_state', default=5, type=int)
    parser.add_argument('--method', default='logreg', type=str, choices=["logreg", "knn"])

    args = parser.parse_args()

    print(args)

    train_dataset, val_dataset, test_dataset, NUM_CLASSES = build_dataset(
        args.dataset,
        train_transform_mode="train_classifier" if args.method == "logreg" else "test_classifier",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=0.0,
        random_state=args.random_state
    )

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

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

    X_train, y_train = to_np_features(model, trainloader, args.feature_type)
    X_test, y_test = to_np_features(model, testloader, args.feature_type)

    if args.method == "logreg":
        logreg = LogisticRegression(solver="saga", n_jobs=20)
        logreg.fit(X_train, y_train)
        acc = logreg.score(X_test, y_test)
    elif args.method == "knn":
        knn = KNeighborsClassifier(n_neighbors=args.nn_k, n_jobs=-1, p=2, metric="minkowski")
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
    else:
        raise ValueError("Unsupported method", args.method)

    print("Accuracy: {:.4f}%".format(acc * 100))


if __name__ == "__main__":
    main()
