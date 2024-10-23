Run PCA.

## Step 1. Train model

Train model as usual. For example,

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_192_trytry \
  --model iTransformer \
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
  --batch_size 24 \
  --learning_rate 0.0005 \
  --itr 1 \
  --max_iter 100
```

## Step 2. Save Mid tensor

For example, 

```bash
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_192_trytry \
  --model iTransformer \
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
  --batch_size 24 \
  --learning_rate 0.0005 \
  --itr 1 \
  --max_iter 100 \
  --save_mid_tensor_for_PCA true \
  --model_load_from /root/PCA_exps/checkpoints/ECL_512_192_trytry_iTransformer_custom_M_ft512_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth
```

Note that `max_iter` \* `batch_size` tensors will be saved.

## Step 3. conduct PCA and draw graph

For middle results:

```bash
python analyze_PCA.py \
--input_tensor_dir ./PCA_exps/mid_results/ \
--output_position ./PCA_exps/mid_results.pth \
--to_image ./PCA_exps_mid_results.png \
--component_number 500
```

And for original time series:

```bash
python analyze_PCA.py \
--input_tensor_dir ./PCA_exps/org_tensor/ \
--output_position ./PCA_exps/org_tensor.pth \
--to_image ./PCA_exps_org_tensor.png \
--component_number 500
```

