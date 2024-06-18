python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric euclidean --k 5 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric euclidean --k 15 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric euclidean --k 20 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric euclidean --k 200 --model_checkpoint_path /checkpoint_last.pth
\
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric cosine --k 5 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric cosine --k 15 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric cosine --k 20 --model_checkpoint_path /checkpoint_last.pth
python eval_sklearn.py --dataset tinyimagenet --feature_type backbone_features --metric cosine --k 200 --model_checkpoint_path /checkpoint_last.pth
