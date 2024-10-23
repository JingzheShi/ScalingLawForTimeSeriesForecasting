export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer
--model_load_from /root/iTransformer/checkpoints/________384_4layer_iMLP_res_030601ver_self_created_336_96_org____autodl24030804_____iMLP_res_self_created_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Newautodl040641__1923lresgateMLPpatch6311and128___weather_512_336___100_lr3e-3__wd0lasso10e-5 \
  --model iMLP_res_gate_patch_test \
  --in_patch_size 6 --in_patch_stride 3 \
  --out_patch_size 1 --out_patch_stride 1 \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --label_len 128 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 192\
  --linear_embedding \
  --d_ff 192\
  --itr 1 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.003\
  --weight_decay 0.0\
  --l1_lambda 0.0001\
  --percent 100 \
  --patience 10 \
  --coef 0.98 >> Newautodl040641__1283lresgateMLPpatch6311and128__weather_512_336__p100___lr3e-3___wd0lasso10e-5___.log &



nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Newautodl040403__3841lresgatenarrowMLP__nopatch___weather_336_192___100_lr3e-3__wd0lasso2e-5 \
  --model iMLP_gate_narrow \
  --in_patch_size 6 --in_patch_stride 2\
  --out_patch_size 1 --out_patch_stride 1\
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 192 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 384\
  --linear_embedding \
  --d_ff 128\
  --itr 1 \
  --batch_size 16384 \
  --train_epochs 30 \
  --learning_rate 0.003\
  --weight_decay 0.0\
  --l1_lambda 0.00002\
  --percent 100 \
  --patience 10 \
  --coef 0.98 >> Newautodl040403__3841lresgatenarrowMLP__nopatch___weather_336_192___100_lr3e-3__wd0lasso2e-5.log &


nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Newautodl040404_Linear_nopatch_weather_512_192_largebatchsize_lr3e-3_wd0.0_lasso3e-4 \
  --model Linear \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 192 \
  --d_ff 192 \
  --batch_size 8192 \
  --learning_rate 0.003 \
  --coef 0.985 \
  --weight_decay 0.0 \
  --l1_lambda 0.0003 \
  --train_epoch 100 \
  --patience 10 \
  --itr 1 >> Newautodl040404_Linear_nopatch_weather_512_192_largebatchsize_lr3e-3_wd0.0_lasso3e-4.log &








































nohup python3 -u run.py   --exp_name MTSF \
  --is_training 1  \
   --root_path ./dataset/weather/ \
     --data_path weather.csv  \
      --model_id autodl24033104__1281lresgateMLP_patch1111__weather_512_192____  \
       --model iMLP_res_gate_patch\
          --data custom \
            --features M \
              --seq_len 512\
                 --pred_len 192 \
                   --label_len 48 \
                     --e_layers 1  \
                      --enc_in 21  \
                       --dec_in 21 \
                         --c_out 21 \
                           --des 'Exp'   --d_model 128  --linear_embedding   --d_ff 128  --itr 1   --batch_size 16384   --train_epochs 30   --learning_rate 0.003 --weight_decay 0.0003 --in_patch_size 4 --patience 10 --in_patch_stride 1 --out_patch_size 1 --out_patch_stride 1
















nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id autodl24031431_128643216_iMLP_pure_64_10layers_031102ver_weather_1024_720_1e-3lr \
  --model iMLP_pure \
  --data custom \
  --features M \
  --seq_len 1024 \
  --pred_len 720 \
  --e_layers 10 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 64\
  --d_ff 64\
  --itr 1 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.001 >> autodl24031431_128643216_iMLP_pure_weather_1024_720.log &



nohup python3 -u run.py \
  --exp_name MTSF \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id autodl24031331_196_1layer_linearemb_iMLPresGate_031102ver_weather_336_192_1e-3lr_withshrimpAug \
  --model iMLP_res_gate \
  --data Mix \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 196\
  --d_ff 196\
  --itr 1 \
  --batch_size 8192 \
  --train_epochs 30 \
  --learning_rate 0.001 --linear_embedding >> autodl24031331_196_1layer_linearemb_iMLPresGate_031102ver+1e-3lr_weather_336_192_withshrimpAug.log &




nohup python3 -u run.py \
  --is_training 1 \
  --exp_name MTSF_wMask \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id _ssmalllr_5pFS__iMLPwmask_res_weather_96_96_from_scratch_onlydecoder__24031002 \
  --model iMLP_wMask_res \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384\
  --itr 1 \
  --batch_size 12 \
  --train_epochs 30 \
  --learning_rate 0.00004 \
  --five_percent_few_shot >> _ssmalllr_5pFS_iMLPwmask_res_weather_96_96_from_scratch_onlydecoder_24031002.log &













--model_load_from /root/iTransformer/checkpoints/__5pFS__384_4layer_weather_336_96_from_pretrained__24030811_iMLP_res_custom_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
/root/iTransformer/checkpoints/__5pFS__384_4layer_weather_336_96_from_pretrained__24030811_iMLP_res_custom_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth
python -u run.py \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_336_96 \
  --model iMLP_res \
  --model_load_from /root/iTransformer/checkpoints/_384_4layer_iMLP_res_030601ver_traffic_336_96_org____autodl24030704_____iMLP_res_custom_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384\
  --itr 1



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1














nohup python3 -u run.py \
  --exp_name MTSF_obtainPCA \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id obtain_pca_weather_336_192 \
  --model iMLP_res_gate \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --batch_size 512 \
  --train_epochs 30 \
  --learning_rate 0.0002 >> autodl24031104__128_1layer_iMLP_res_gate_031102ver_weather_336_96_org.log &

python3 -u run.py \
  --exp_name MTSF_withPCA \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id autodl24031531__128_1layer_PCAMLP_weather_336_192 \
  --model PCALinear \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --label_len 336 \
  --e_layers 1 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128\
  --d_ff 128\
  --itr 1 \
  --batch_size 2048 \
  --train_epochs 30 \
  --learning_rate 0.0002 >> autodl24031531__PCAMLP_gate_031102ver_weather_336_96_org.log &