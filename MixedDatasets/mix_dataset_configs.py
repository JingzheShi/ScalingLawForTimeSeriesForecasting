SEQ_LEN = 96
LABEL_LEN = 48
PRED_LEN = 96
mix_dataset_dict_0_aug = [
    ('self_created', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = None,
                data_path = None,
                seq_len = 336,
                label_len = 336,
                pred_len = 192,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
    ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/weather/',
                data_path = 'weather.csv',
                seq_len = 336,
                label_len = 336,
                pred_len = 192,
                features = 'M',
                target = 'OT',
                num_workers=0,
            ))
]

mix_dataset_dict_aug = [
    ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/weather/',
                data_path = 'weather.csv',
                seq_len = 336,
                label_len = 336,
                pred_len = 192,
                features = 'M',
                target = 'OT',
                num_workers=0,
            ))
]
mix_dataset_dict_1 = [
    ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/weather/',
                data_path = 'weather.csv',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
    ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/traffic/',
                data_path = 'traffic.csv',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
]
mix_dataset_dict_2 = [
            # weather dataset.
            ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/weather/',
                data_path = 'weather.csv',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            # solar dataset.
            ('Solar', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/Solar/',
                data_path = 'solar_AL.txt',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/traffic/',
                data_path = 'traffic.csv',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('PEMS', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/PEMS/',
                data_path = 'PEMS03.npz',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('PEMS', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/PEMS/',
                data_path = 'PEMS07.npz',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('custom', dict(
                embed = 'timeF',
                freq = 'h',
                root_path = './dataset/exchange_rate/',
                data_path = 'exchange_rate.csv',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('ETTh1',dict(
                root_path = './dataset/ETT-small/',
                data_path = 'ETTh1.csv',
                embed = 'timeF',
                freq = 'h',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('ETTh2',dict(
                root_path = './dataset/ETT-small/',
                data_path = 'ETTh2.csv',
                embed = 'timeF',
                freq = 'h',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('ETTm1',dict(
                root_path = './dataset/ETT-small/',
                data_path = 'ETTm1.csv',
                embed = 'timeF',
                freq = 'h',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('ETTm2',dict(
                root_path = './dataset/ETT-small/',
                data_path = 'ETTm2.csv',
                embed = 'timeF',
                freq = 'h',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            )),
            ('custom',dict(
                root_path = './dataset/electricity/',
                data_path = 'electricity.csv',
                embed = 'timeF',
                freq = 'h',
                seq_len = SEQ_LEN,
                label_len = LABEL_LEN,
                pred_len = PRED_LEN,
                features = 'M',
                target = 'OT',
                num_workers=0,
            ))
            
]
import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from experiments.exp_long_term_forecasting_wMask import Exp_Long_Term_Forecast_wMask
from experiments.exp_long_term_forecasting_wFreqMask import Exp_Long_Term_Forecast_wFreqMask
import random
import numpy as np
fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='iTransformer')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=False, default='traffic_96_96_wMask_wMaskLoss_96maskratio', help='model id')
parser.add_argument('--model', type=str, required=False, default='iTransformer_wFreqMask',
                    help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
parser.add_argument('--model_load_from',type=str,default=None)


# data loader
parser.add_argument('--data', type=str, required=False, default='Mix', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/Solar/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='solar_AL.txt', help='data csv file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=6, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=12, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# iTransformer
parser.add_argument('--exp_name', type=str, required=False, default='MTSF_wFreqMask',
                    help='experiemnt name, options:[MTSF, partial_train, MTSF_wMask]')
parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                        'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')



parser.add_argument('--cut_freq', type=int, default = None)



















# from mix_dataset_configs import mix_dataset_dict_1
parser.add_argument('--datasetsDict_list',type=dict,default=mix_dataset_dict_0_aug)

if __name__ == '__main__':
    args = parser.parse_args()
    from data_provider.data_factory import data_provider

    train_dataset_debug, train_dataloader_debug = data_provider(args, 'train')
    from tqdm import tqdm
    for _ in range(2):
        for idx, batch in tqdm(enumerate(train_dataloader_debug)):
            pass
            if idx % 1000 == 0:
                for item in batch:
                    pass
                    # print(item.shape)
