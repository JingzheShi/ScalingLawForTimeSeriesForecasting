export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

nohup python -u run.py \
  --exp_name MTSF_wMask \
  --is_training 0 \
  --root_path /root/autodl-tmp/iTransformer_datasets/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model_load_from /root/iTransformer/checkpoints/traffic_192_144_wMask_wFreqMaskLoss_30maskratio0.1reconweight_iMLP_______202402250702____________iMLP_wMask_custom_M_ft192_sl48_ll144_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --model iMLP_wMask_adaptive \
  --data Solar \
  --features M \
  --seq_len 192 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0002 \
  --itr 1 >> 2024022601.log &

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model iTransformer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model iTransformer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model iTransformer \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1
