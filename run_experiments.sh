python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --hidden_dim 1024 --temp 0.1 --dataset tinyimagenet --out_features 128 --criterion sacl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --hidden_dim 1024 --temp 0.1 --dataset tinyimagenet --out_features 128 --criterion sacl_batchmix --batch_size 128 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.05 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset tinyimagenet --out_features 128 --criterion scl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.125 --rho_const 1e5 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.5 --dataset tinyimagenet --out_features 128 --criterion simclr --batch_size 256 --metric cosine --epochs 1000 --momentum 0.9 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
\
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --hidden_dim 1024 --temp 0.1 --out_features 128 --criterion sacl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --dataset stl10
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --hidden_dim 1024 --temp 0.1 --out_features 128 --criterion sacl_batchmix --batch_size 128 --metric cosine --epochs 1000 --alpha 0.5 --rho_const 1 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --dataset stl10
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.05 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --out_features 128 --criterion scl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.125 --rho_const 1e5 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --dataset stl10
python main_sacl.py --first_conv --no-drop_maxpool --base_lr 0.25 --weight_decay 1e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.5 --out_features 128 --criterion simclr --batch_size 128 --metric cosine --epochs 1000 --momentum 0.9 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging --dataset stl10
\
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar10 --out_features 128 --criterion sacl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar10 --out_features 128 --criterion sacl_batchmix --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar10 --out_features 128 --criterion scl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.5 --dataset cifar10 --out_features 128 --criterion simclr --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
\
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar100 --out_features 128 --criterion sacl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar100 --out_features 128 --criterion sacl_batchmix --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.1 --dataset cifar100 --out_features 128 --criterion scl --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
python main_sacl.py --first_conv --drop_maxpool --base_lr 0.03 --weight_decay 5e-4 --val_split 0.0 --num_workers 20 --hidden_dim 1024 --temp 0.5 --dataset cifar100 --out_features 128 --criterion simclr --batch_size 128 --metric cosine --epochs 1000 --alpha 0.25 --rho_const 1e4 --momentum 0.9 --s_init_t -1.0 --lr_anneal cosine --warmup_epochs 10 --lr_scale linear --wandb_logging
