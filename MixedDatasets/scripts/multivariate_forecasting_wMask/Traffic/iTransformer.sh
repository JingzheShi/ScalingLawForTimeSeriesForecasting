export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer
nohup python3 -u run.py \
    --is_training 1 \
    --data Mix \
    --model_id iMLP60Mask_traffic_________________24022407__1.2aug__________ \
    --model iMLP_wMask \
    --exp_name MTSF_wMask \
    --e_layers 4 \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 12 \
    --train_epochs 30 \
    --learning_rate 0.0002 \
    --itr 1 >> MixTraffic_96to96_wFreqMask60_15e_________24022407_1.2aug______.log &\
--model_load_from /root/iTransformer/checkpoints/huge_traffic_96_96_wMask_wFreqMaskLoss_30maskratio0.1reconwht__org__autodl24030403____iMLP_wMask_res_custom_M_ft96_sl48_ll96_pl768_dm8_nh6_el1_dl768_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \

nohup python3 -u run.py \
  --exp_name MTSF_wMask \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id _384_naivereconloss_4layer_iMLP_res_030601_ver_traffic_336_192_org_30ratio0.01weight___autodl24030704____\
  --model iMLP_wMask_res \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384 \
  --batch_size 12 \
  --train_epochs 30 \
  --recon_loss_weight 0.01 \
  --learning_rate 0.0004 \
  --itr 1 >> _384_naivereconloss_4layer_iMLP_res_030601__traffic_336_192_org_30ratio0.01weight___autodl24030704____.log &

--model_load_from /root/iTransformer/checkpoints/base_traffic_512_96_wMask_wFreqMaskLoss_30maskratio0.1reconwht__org__autodl24030501_selfsuper____iMLP_wMask_res_custom_M_ft512_sl48_ll96_pl512_dm8_nh6_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
python3 -u run.py \
  --exp_name MTSF_wMask \
  --is_training 0\
  --model_load_from /root/iTransformer/checkpoints/traffic_192_144_wMask_wFreqMaskLoss_30maskratio0.1reconweight_iMLP_______202402250702____________iMLP_wMask_custom_M_ft192_sl48_ll144_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_192_96_wMask \
  --model iMLP_wMask_adaptive \
  --data custom \
  --features M \
  --seq_len 192 \
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