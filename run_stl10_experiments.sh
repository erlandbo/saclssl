# sacl
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --no-use_fp32 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --knn_k 20 --knn_fx_distance cosine --hidden_dim 1024 --knn_temp 0.07 --knn_weights distance --temp 0.1 --out_features 128 --criterion sacl --batch_size 128 --validate_interval 10 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --wandb_project_name stl10 --dataset stl10
\
# saclbmix
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --no-use_fp32 --val_split 0.0 --num_workers 20 ----hidden_dim 1024 --knn_k 20 --knn_fx_distance cosine --hidden_dim 1024 --knn_temp 0.07 --knn_weights distance --temp 0.1 --out_features 128 --criterion sacl_batchmix --batch_size 128 --validate_interval 10 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --wandb_project_name stl10 --dataset stl10
\
# scl
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --no-use_fp32 --val_split 0.0 --num_workers 20 --knn_k 20 --knn_fx_distance cosine --hidden_dim 1024 --knn_temp 0.07 --knn_weights distance --temp 0.1 --out_features 128 --criterion scl --batch_size 128 --validate_interval 10 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1000000 --momentum 0.9 --s_init_t -3.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --wandb_project_name stl10 --dataset stl10
\
# simclr
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --no-use_fp32 --val_split 0.0 --num_workers 20 --knn_k 20 --knn_temp 0.07 --knn_fx_distance cosine --hidden_dim 1024 --knn_weights distance --temp 0.5 --out_features 128 --criterion simclr --batch_size 128 --validate_interval 10 --metric cosine --epochs 1000 --momentum 0.9 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --wandb_project_name stl10 --dataset stl10
