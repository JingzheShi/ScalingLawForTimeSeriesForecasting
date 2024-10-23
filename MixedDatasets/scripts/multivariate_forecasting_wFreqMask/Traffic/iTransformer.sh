export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

nohup python3 -u run.py \
  --exp_name MTSF_wFreqMask \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96_wMask_wMaskLoss_70maskratio_freqloss_——————2024021611————————AllMLP \
  --model iTransformer_wFreqMask \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 12 \
  --train_epochs 12 \
  --learning_rate 0.0002 \
  --itr 1 >> traffic_96_96_wMask_wMaskLoss_70maskratio_freqloss_——————2024021611——————AllMLP.log &


python3 -u run.py \
  --exp_name MTSF_wFreqMask \
  --is_training 0\
  --model_load_from /root/iTransformer/checkpoints/traffic_96_96_wMask_wMaskLoss_70maskratio_freqloss_——————2024021611————————AllMLP_iTransformer_wFreqMask_custom_M_ft96_sl48_ll96_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96_wMask_nomaskloss \
  --model iTransformer_wFreqMask \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0001\
  --itr 1