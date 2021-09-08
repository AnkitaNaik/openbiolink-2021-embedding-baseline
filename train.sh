# RESCAL
DGLBACKEND=pytorch python3 -m dglke.train --model_name RESCAL --dataset OBL2021 --batch_size 2048 --neg_sample_size 1024 --hidden_dim 300 --gamma 12.0 --lr 0.05 --max_step 350000 --log_interval 1000 --batch_size_eval 512 --regularization_coef 3.00E-07 -rn 3 --gpu 0 1 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part

# TransE
DGLBACKEND=pytorch python3 -m dglke.train --model_name TransE_l2 --dataset OBL2021 --batch_size 2048 --neg_sample_size 1024  --hidden_dim 360 --gamma 8.0 --lr 0.1 --max_step 550000 --log_interval 1000 --batch_size_eval 512 --regularization_coef 3.00E-09 -rn 3 --gpu 0 1 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part

# Distmult
DGLBACKEND=pytorch python3 -m dglke.train --model_name DistMult --dataset OBL2021 --batch_size 2048 --neg_sample_size 1024  --hidden_dim 380 --gamma 12.0 --lr 0.15 --max_step 950000 --log_interval 1000 --batch_size_eval 128 --regularization_coef 4.00E-07 -rn 3 --gpu 0 1 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part

# Rotate
DGLBACKEND=pytorch python3 -m dglke.train --model_name RotatE --dataset OBL2021 --batch_size 2048 --neg_sample_size 1024  --hidden_dim 128 --gamma 12.0 --lr 0.05 --max_step 550000 --log_interval 1000 --batch_size_eval 16 --regularization_coef 1.00E-07 -rn 3 --gpu 0 1 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part --double_ent

# TransR
DGLBACKEND=pytorch python3 -m dglke.train --model_name TransR --dataset OBL2021 --batch_size 1024 --neg_sample_size 1024  --hidden_dim 200 --gamma 8.0 --lr 0.03 --max_step 650000 --log_interval 1000 --batch_size_eval 32 --regularization_coef 8.00E-09 -rn 3 --gpu 0 1 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part

# ComplEx
DGLBACKEND=pytorch python3 -m dglke.train --model_name ComplEx --dataset OBL2021 --batch_size 1024 --neg_sample_size 512  --hidden_dim 380 --gamma 12.0 --lr 0.1 --max_step 360000 --log_interval 1000 --batch_size_eval 512 --regularization_coef 2.00E-06 -rn 3 --gpu 0 1 2 3 4 5 6 7 --mix_cpu_gpu --async_update --force_sync_interval 1000 -adv -a 1.0 --rel_part