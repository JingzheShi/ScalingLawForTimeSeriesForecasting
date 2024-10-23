export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer


nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id Newautodl040401_Linear_nopatch_ETTm1_512_96_largebatchsize_lr3e-3_wd6e-3_lasso0.0 \
  --model Linear \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 192 \
  --d_ff 192 \
  --batch_size 8192 \
  --learning_rate 0.003 \
  --coef 0.985 \
  --weight_decay 0.006 \
  --l1_lambda 0.0 \
  --train_epoch 100 \
  --patience 10 \
  --itr 1 >> Newautodl040401_Linear_nopatch_ETTm1_512_96_largebatchsize_lr3e-3_wd6e-3_lasso0.0.log &

  --in_patch_size 3 --in_patch_stride 2\
  --out_patch_size 1 --out_patch_stride 1\


nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id norm____384_3layer_iMLP_res__031121ver_ETTm1_336_192_org_new \
  --model iMLP_res_gate \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 384 \
  --d_ff 384 \
  --batch_size 4096 \
  --train_epoch 40 \
  --learning_rate 0.0002 \
  --itr 1 >> autodl24031251CI_1e-1norm____384_3layer_iMLP_res__031121ver_ETTm1_336_192_org_new.log &
nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id autodl24031351CInorm____512_0layer_iMLP_res__031121ver_ETTm2_336_192_org_new \
  --model iMLP_res_gate \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 4096 \
  --train_epoch 40 \
  --learning_rate 0.0005 \
  --linear_embedding \
  --itr 1 >> autodl24031351CI_1e-1norm____512_0layer_iMLP_res__031121ver_ETTm2_336_192_org_new.log &
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
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
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
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
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
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