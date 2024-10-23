export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model iMLP_res \
  --model_load_from /root/iTransformer/checkpoints/________384_4layer_iMLP_res_030601ver_self_created_336_96_org____autodl24030804_____iMLP_res_self_created_M_ft336_sl48_ll96_pl384_dm8_nh4_el1_dl384_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
  --model_id fsiawfneuoijvsndjwae \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 384\
  --d_ff 384\
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Newautodl240421____512_3layer_iMLP_res_gate_031121ver_ECL_512_192_org_linearemb__4211__ \
  --model iMLP_res_gate_patch \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 8192 \
  --learning_rate 0.0003 \
  --weight_decay 0.00002 \
  --in_patch_size 4 --in_patch_stride 2\
  --out_patch_size 1 --out_patch_stride 1\
  --percent 100 \
  --linear_embedding \
  --itr 1 >> Newautodl240421____512_3layer_iMLP_res_gate_031121ver_ECL_512_192org_linearemb__4211__.log &

nohup python3 -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Newautodl240421____512_3layer_iMLP_res_gate_031121ver_ECL_512_192_org_linearemb__nopatch__ \
  --model iMLP_res_gate \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 8192 \
  --learning_rate 0.0003 \
  --weight_decay 0.00002 \
  --in_patch_size 4 --in_patch_stride 2\
  --out_patch_size 1 --out_patch_stride 1\
  --percent 100 \
  --linear_embedding \
  --itr 1 >> Newautodl240421____512_3layer_iMLP_res_gate_031121ver_ECL_512_192org_linearemb__nopatch__.log &
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1