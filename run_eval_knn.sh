python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance euclidean --nn_k 5 --weights uniform --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance euclidean --nn_k 15 --weights uniform --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance euclidean --nn_k 20 --weights uniform --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance euclidean --nn_k 200 --weights uniform --model_checkpoint_path /checkpoint_last.pth
\
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance cosine --nn_k 5 --weights distance --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance cosine --nn_k 15 --weights distance --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance cosine --nn_k 20 --weights distance --model_checkpoint_path /checkpoint_last.pth
python eval_knn.py --dataset tinyimagenet --feature_type backbone_features --fx_distance cosine --nn_k 200 --weights distance --model_checkpoint_path /checkpoint_last.pth
