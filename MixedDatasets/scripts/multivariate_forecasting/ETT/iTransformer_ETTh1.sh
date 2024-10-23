export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Newautodl240451_96_1layer_iMLPresGate_031102ver_ETTh1_336_192_largebatchsize_nopatch \
  --model iMLP_res_gate \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 96 \
  --d_ff 96 \
  --batch_size 8192 \
  --learning_rate 0.0003 \
  --train_epoch 30 \
  --itr 1 >> Newautodl240451_96_1layer_iMLPresGate_031102ver_ETTh1_336_192_largebatchsize_nopatch.log &

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Newautodl040211_96_0layer_iMLPresNoGate_031102ver_ETTh1_720_192_largebatchsize_3211_wd0lasso1e-5 \
  --model iMLP_res_gate_patch \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --e_layers 0 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 192 \
  --d_ff 192 \
  --batch_size 8192 \
  --learning_rate 0.003 \
  --train_epoch 60 \
  --patience 10 \
  --in_patch_size 3 --in_patch_stride 2\
  --l1_lambda 0.00001 \
  --out_patch_size 1 --out_patch_stride 1\
  --itr 1 >> Newautodl040211_96_0layer_iMLPresNoGate_031102ver_ETTh1_720_192_largebatchsize_3211_wd0lasso1e-5.log &

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id Newautodl040301_Linear_nopatch_ETTh1_336_720_largebatchsize_wd0_lasso2e-4 \
  --model Linear \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
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
  --itr 1 >> Newautodl040301_Linear_nopatch_ETTh1_336_720_largebatchsize_wd0_lasso2e-4.log &
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1