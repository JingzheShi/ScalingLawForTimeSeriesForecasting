nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id New041301__3843l_MLPinterpolate_weather_interto64__336_192__p100_lr1e-3_wd0_lasso0_bs32648 \
  --model MLP_interpolate \
  --in_patch_stride 1 --in_patch_size 1 \
  --out_patch_stride 1 --out_patch_size 1 \
  --data custom \
  --features M \
  --interpolate_len 64 \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 128 \
  --des 'Exp' \
  --e_layers 3 \
  --d_model 384 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_ff 192 \
  --itr 1 \
  --batch_size 32648 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --weight_decay 0.0 \
  --l1_lambda 0.0 \
  --percent 100 \
  --patience 10 \
  --coef 0.98 >> New041301__3843l_MLPinterpolate_weather_interto64__336_192__p100_lr1e-3_wd0_lasso0_bs32648.log &