export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id Newautodl041201_Linear_interpolateTo1536_ETTh2_3072_96_largebatchsize_wd0_lasso2e-4 \
  --model Linear_interpolate \
  --data ETTh2 \
  --features M \
  --seq_len 3072 \
  --interpolate_len 1536 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 192 \
  --d_ff 192 \
  --batch_size 8192 \
  --learning_rate 0.003 \
  --weight_decay 0.0 \
  --l1_lambda 0.0002 \
  --coef 0.985 \
  --train_epoch 100 \
  --patience 30 \
  --in_patch_size 3 --in_patch_stride 2\
  --out_patch_size 1 --out_patch_stride 1\
  --itr 1 >> Newautodl041201_Linear_interpolateTo1536_ETTh2_3072_96_largebatchsize_wd0_lasso2e-4.log &


nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id autodl24031251CI_1e-1norm____128_1layer_iMLP_res__031121ver_ETTh2_336_192_org_new \
  --model iMLP_res_gate \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 2048 \
  --train_epoch 40 \
  --learning_rate 0.0002 \
  --itr 1 >> autodl24031251CI_1e-1norm____128_1layer_iMLP_res__031121ver_ETTh2_336_192_org_new.log &

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1