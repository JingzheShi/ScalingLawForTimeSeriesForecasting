export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer
nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id Newautodl040684__3843lresgateMLPpatch8411and256_b22648__traffic_1024_720___100_lr10e-4__wd1e-5lasso0 \
  --model iMLP_res_gate_patch_test \
  --in_patch_size 8 --in_patch_stride 4 \
  --out_patch_size 1 --out_patch_stride 1 \
  --data custom \
  --features M \
  --seq_len 1024 \
  --pred_len 720 \
  --label_len 1 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 384\
  --linear_embedding \
  --d_ff 384\
  --itr 1 \
  --batch_size 22648 \
  --train_epochs 8 \
  --learning_rate 0.001\
  --weight_decay 0.00001\
  --l1_lambda 0.0\
  --percent 100 \
  --patience 3 \
  --coef 0.98 >> Newautodl040684__3843lresgateMLPpatch8411and256_bs22648__traffic_1024_720_p100___lr10e-4___wd1e-5lasso0___.log &
nohup python3 -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id autodl24040672CI__gatedprojector__512_4layer_5e-4lr_0.0weightdecay1e-5_iMLP_res_031121ver_rand100p_traffic_512_336_nopatch____ \
  --model iMLP_res_gate \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.0005 \
  --weight_decay 0.00001 \
  --patience 5 \
  --percent 100 \
  --itr 1 >> autodl24040672_gatedprojector_512_4layer_5e-4lr_0.0weightdecay1e-5_iMLP_res_031121ver_rand100p_traffic_512_336_nopatch.log  &

nohup python3 -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id autodl240401_CI__gatedprojector__384_3layer_2e-4lr_0.0weightdecay_iMLP_res_031121ver_rand5p_traffic_512_192_____4211 \
  --model iMLP_res_gate_patch \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.0002 \
  --weight_decay 0.0 \
  --in_patch_size 4 --in_patch_stride 2 \
  --out_patch_size 1 --out_patch_stride 1 \
  --patience 5 \
  --percent 5 \
  --itr 1 >> autodl240401_gatedprojector_384_3layer_2e-4lr_0.0weightdecay_iMLP_res_031121ver_rand5p_traffic_512_192__4211.log  &



nohup python3 -u run.py \
  --is_training 0 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id autodl24031022CI____384_4layer_iMLP_res_gate_031021ver_traffic_336_336_org____autodl24031022_____\
  --model_load_from /root/iTransformer/checkpoints/autodl24031022CI____384_4layer_iMLP_res_gate_031021ver_traffic_336_336_org____autodl24031022_____iMLP_res_gate_custom_M_ft336_sl48_ll336_pl384_dm8_nh3_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --model iMLP_res_gate \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.0002 \
  --patience 20 \
  --itr 1 >> autodl24031101__384_3layer_iMLP_res_gate_031021ver_traffic_336_96_org.log  &
# /root/iTransformer/checkpoints/__5pFS__384_4layer_weather_336_96_from_pretrained__24030811_iMLP_res_custom_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
nohup python3 -u run.py \
  --is_training 1 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id _smalllr_5pFS__384_4layer_traffic_336_96_from_pretrained__24030951 \
  --model iMLP_res \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384\
  --itr 1 \
  --batch_size 256 \
  --train_epochs 30 \
  --learning_rate 0.0002 \
  --five_percent_few_shot >> _smalllr_5pFS_384_4layer_traffic_336_96_from_pretrained_24030951.log &

python3 -u run.py \
  --is_training 0 \
  --root_path /root/autodl-tmp/iTransformer_datasets/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_336_96_org____24022412____ \
  --model_load_from /root/iTransformer/checkpoints/________384_4layer_iMLP_res_030601ver_self_created_336_96_org____autodl24030804_____iMLP_res_self_created_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --model iMLP_res \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1



nohup python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id iTF_traffic_512_96 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 12 \
  --train_epochs 30 \
  --learning_rate 0.0004 \
  --itr 1 >> iTF_traffic_512_96.log &

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
  --learning_rate 0.001\
  --itr 1