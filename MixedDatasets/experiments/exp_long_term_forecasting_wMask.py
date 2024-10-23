from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from random import uniform

warnings.filterwarnings('ignore')

class LearnableInterpolation(nn.Module):
    def __init__(self, channel_num = 21,mark_num = 4, ratio_range = [0.7,1.3], bias_range = [-4.0, 4.0]):
        super().__init__()
        self.channel_num = channel_num
        self.mark_num = mark_num
        self.ratio_minus_one = nn.Parameter(torch.zeros(self.channel_num).float(),requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.channel_num).float(),requires_grad=True)
        self.mark_ratio_minus_one = nn.Parameter(torch.zeros(self.mark_num).float(),requires_grad=True)
        self.mark_bias = nn.Parameter(torch.zeros(self.mark_num).float(),requires_grad=True)
        self.min_ratio = min(ratio_range, default = 0.85)
        self.max_ratio = max(ratio_range, default = 1.15)
        self.min_bias = min(bias_range, default = -2.0)
        self.max_bias = max(bias_range, default = 2.0)
        self.softmax_scale = 3.0
        
    def reset_weight(self):
        with torch.no_grad():
            self.ratio_minus_one[:] = 0.0
            self.bias[:] = 0.0
            self.mark_ratio_minus_one[:] = 0.0
            self.mark_bias[:] = 0.0
    def input_transformation(self, org_tensor, to_length, is_mark = False):
        # org_tensor: B L N
        B,L_org,N = org_tensor.shape
        L_to = to_length
        if not is_mark:
            ratio = self.ratio_minus_one + 1
            bias = self.bias
        else:
            ratio = self.mark_ratio_minus_one + 1
            bias = self.mark_bias
        ratio = torch.nn.functional.relu(ratio - self.min_ratio) + self.min_ratio
        ratio = - torch.nn.functional.relu(self.max_ratio - ratio) + self.max_ratio
        bias = torch.nn.functional.relu(bias - self.min_bias) + self.min_bias
        bias = - torch.nn.functional.relu(self.max_bias - bias) + self.max_bias
        org_position = torch.arange(-L_org,0,1,device = org_tensor.device).float() # L_org
        org_position = org_position.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B,1,N,L_to) # B L_org N L_to
        
        transformed_position = org_position * ratio.unsqueeze(0).unsqueeze(-1) + bias.unsqueeze(0).unsqueeze(-1)
        
        target_position = torch.arange(-L_to,0,1,device = org_tensor.device).float() # L_to
        target_position = target_position.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 1 1 L_to
        
        deltas = (transformed_position - target_position)**2 # B L_org N L_to
        weights = torch.nn.functional.softmax(-deltas * self.softmax_scale,dim=-3) # B L_org N L_to
        out_tensor = torch.einsum('bont, bon->btn',weights,org_tensor)
        return out_tensor
    
    def output_transformation(self, pred_tensor, org_length):
        B, L_to, N = pred_tensor.shape
        L_org = org_length
        ratio = self.ratio_minus_one + 1
        ratio = torch.nn.functional.relu(ratio - self.min_ratio) + self.min_ratio
        ratio = - torch.nn.functional.relu(self.max_ratio - ratio) + self.max_ratio
        bias = torch.nn.functional.relu(self.bias - self.min_bias) + self.min_bias
        bias = - torch.nn.functional.relu(self.max_bias - bias) + self.max_bias
        
        org_position = torch.arange(0,L_org,1,device = pred_tensor.device).float() # L_org
        org_position = org_position.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B,1,N,L_to) # B L_org N L_to
        transformed_position = org_position * ratio.unsqueeze(0).unsqueeze(-1) + bias.unsqueeze(0).unsqueeze(-1)
        
        target_position = torch.arange(0,L_to,1,device = pred_tensor.device).float() # L_to
        target_position = target_position.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 1 1 L_to
        
        deltas = (transformed_position - target_position)**2 # B L_org N L_to
        weights = torch.nn.functional.softmax(-deltas * self.softmax_scale,dim=-1) # B L_org N L_to
        out_tensor = torch.einsum('bont, btn->bon',weights,pred_tensor)
        return out_tensor
        
        
            
        


















class Deformation(nn.Module):
    def __init__(self,ratios=[1.0,1.15,0.85]):
        super(Deformation,self).__init__()
        self.ratios = ratios
        ratios_weight = torch.zeros(len(self.ratios)).float()
        ratios_weight[0] = 1.0
        self.ratios_weight = nn.Parameter(ratios_weight,requires_grad=True)
    def reset_weight(self):
        with torch.no_grad():
            # self.ratios_weight.data = torch.zeros_like(self.ratios_weight.data)
            self.ratios_weight.data[:] = 0.0
            self.ratios_weight.data[0] = 1.0
    def input_transformation(self, org_tensor, to_length):
        print(org_tensor.shape)
        # org_tensor: B L N
        org_tensor = org_tensor.permute(0,2,1) # B N L
        interpolated_tensor_list = list()
        length_list = [int(to_length * ratio) for ratio in self.ratios]
        for length in length_list:
            
            old_interpolated_tensor = torch.nn.functional.interpolate(org_tensor[..., -length:], size = to_length, mode = 'linear', align_corners = False)
            # print("org_tensor[...,-length:].shape",org_tensor[..., -length:].shape)
            # print("old_interpolated_tensor.shape",old_interpolated_tensor.shape)
            
            interpolated_tensor_list.append(old_interpolated_tensor)
        interpolated_tensor = torch.stack(interpolated_tensor_list, dim = -1)
        # interpolated_tensor: B N L R
        weighted_tensor = torch.einsum('bnlr,r->bnl',interpolated_tensor,self.ratios_weight - self.ratios_weight.mean() + 1/len(self.ratios))
        # weighted_tensor: B N L
        return weighted_tensor.permute(0,2,1)
    def output_transformation(self, model_predict_tensor, to_length):
        # return model_predict_tensor[:,:96,:]
        # model_predict_tensor: B L N
        model_predict_tensor = model_predict_tensor.permute(0,2,1) # B N L
        B, N, L = model_predict_tensor.shape
        output_tensor = torch.zeros(B,N,to_length,device = model_predict_tensor.device,dtype=model_predict_tensor.dtype)
        length_list = [int(L * ratio) for ratio in self.ratios]
        for idx,length in enumerate(length_list):
            recovered_tensor = torch.nn.functional.interpolate(model_predict_tensor, size = length, mode = 'linear', align_corners = False)
            # print("model_predict_tensor.shape",model_predict_tensor.shape)
            # print("recovered_tensor.shape",recovered_tensor.shape)
            output_tensor = output_tensor + recovered_tensor[..., :to_length] * (self.ratios_weight[idx] - self.ratios_weight.mean() + 1/len(self.ratios))
        return output_tensor.permute(0,2,1)
        
        
            
            


class Exp_Long_Term_Forecast_wMask(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_wMask, self).__init__(args)
        self.mask_ratio = args.mask_ratio
        self.recon_loss_weight = args.recon_loss_weight
        self.test_time_training_iter_big = 15# mse:0.3711753487586975, mae:0.23760731518268585 with 0
        self.test_time_training_iter_small = 1
        self.model_state_dict = None
        
        
        

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.model_load_from is not None:
            pass
            model.load_state_dict(torch.load(self.args.model_load_from))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.deformation = LearnableInterpolation()
        self.deformation.cuda()
        print("Model size:", str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)+'M')
        return model

    def reload_model(self):
        assert self.args.model_load_from is not None
        if self.model_state_dict is None:
            self.model_state_dict = torch.load(self.args.model_load_from)
        self.load_from_state_dict(self.model_state_dict)
        self.model.load_state_dict(self.model_state_dict)
        

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # print("Only optimize decoder.")
        param_list = []
        freeze_param_name_list = [
           # 'past_projector', 'encoder',  'mask_encoder','enc_embedding', # 'projector', 
        ]
        for name, param in self.model.named_parameters():
            if (not any(name.startswith(freeze_param_name + '.') for freeze_param_name in freeze_param_name_list)):
                param_list.append(param)
        # for name, param in self.model.named_parameters():
        #     if (not name.startswith('projector.')):
        #         param_list.append(param)
        # model_optim = optim.SGD(param_list, lr=self.args.learning_rate, weight_decay=0.00, momentum=0.0)
        model_optim = optim.Adam(param_list, lr=self.args.learning_rate)
        return model_optim
    
    def _select_optimizer_for_TTT(self):
        # optimizer for test-time-training.
        # model_optim = optim.Adam(self.model.parameters(), lr=1e-1,weight_decay = 1e-2)
        param_list = []
        freeze_param_name_list = [
           'projector',  'past_projector', 'encoder', # 'mask_encoder','enc_embedding', 
        ]
        for name, param in self.model.named_parameters():
            if (not any(name.startswith(freeze_param_name + '.') for freeze_param_name in freeze_param_name_list)):
                param_list.append(param)
        # for name, param in self.model.named_parameters():
        #     if (not name.startswith('projector.')):
        #         param_list.append(param)
        model_optim = optim.SGD(param_list, lr=0.02, weight_decay=0.00, momentum=0.0)
        # model_optim = optim.Adam(param_list, lr=3e-5, weight_decay=1e-4)
        return model_optim
    
    def _select_optimizer_for_deformationTTT(self):
        # optimizer = optim.SGD(self.deformation.parameters(), lr=0.02, weight_decay = 0.0001)
        optimizer = optim.Adam(self.deformation.parameters(), lr=0.02, weight_decay = 0.001)
        # optimizer = optim.Adam(self.deformation.parameters(), lr=0.0, weight_decay = 0.000)
        return optimizer
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_criterion_for_recons(self):
        def loss_for_recons(output, batch_x):
            return (output - batch_x).pow(2).mean()
            L = batch_x.shape[-2]
            ratio = self.mask_ratio
            batch_x_mean = batch_x.mean(dim=-2,keepdim=True)
            batch_x_enc = batch_x_mean - batch_x
            
            stdev = torch.sqrt(torch.var(batch_x_enc, dim=-2, keepdim=True, unbiased=False) + 1e-5)
            batch_x_enc = batch_x_enc / stdev
            low_specx = torch.fft.rfft(batch_x_enc, dim = -2, norm = 'ortho')
            low_specx = torch.view_as_real(low_specx[:,0:L//2,:])
            low_specx_real = low_specx[:,:,:,0]
            low_specx_imag = low_specx[:,:,:,1]
            low_specx = torch.cat([low_specx_real,low_specx_imag],dim=-2)
            # low_specx: B L N
            
            
            
            
            
            return (output - low_specx).pow(2).mean()
        return loss_for_recons




    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def mask_x(self,batch_x):
        return self.mask_x_freq(batch_x)
        # ratio = uniform(0,self.mask_ratio)
        ratio = self.mask_ratio
        # print('current ratio is', ratio)
        # batch_x: B L N
        not_mask_indicator = torch.rand_like(batch_x) > ratio
        # not_mask_indicator is a boolean tensor with the same shape as batch_x.
        not_mask_indicator = not_mask_indicator.float()
        # not_mask_indicator is a float tensor with the same shape as batch_x.
        # x[i,j,k] is masked if not_mask_indicator[i,j,k] == 0
        mask_x = batch_x * (1.0 - not_mask_indicator)
        mask_x_mean = mask_x.mean(dim=-2,keepdim=True)
        masked_x = batch_x * not_mask_indicator + mask_x_mean * (1.0 - not_mask_indicator)
        # we mask the input x by replacing the masked values with the mean of the masked values.
        
        return masked_x, not_mask_indicator
    
    def mask_x_freq(self,batch_x):
        # batch_x: B L N
        L = batch_x.shape[-2]
        ratio = self.mask_ratio
        batch_x_mean = batch_x.mean(dim=-2,keepdim=True)
        batch_x_enc = batch_x_mean - batch_x
        # with torch.no_grad():
        stdev = torch.sqrt(torch.var(batch_x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x_enc = batch_x_enc / stdev
        low_specx = torch.fft.rfft(batch_x_enc, dim = 1, norm = 'ortho')
        low_specx = torch.view_as_real(low_specx[:,0:L//2,:])
        low_specx_real = low_specx[:,:,:,0]
        low_specx_imag = low_specx[:,:,:,1]
        low_specx = torch.cat([low_specx_real,low_specx_imag],dim=-2)
        # low_specx: B L N
        
        
        not_mask_indicator = torch.rand_like(low_specx) > ratio
        
        not_mask_indicator = not_mask_indicator.float()
        mask_low_specx = low_specx * (1.0 - not_mask_indicator)
        
        low_spec_complex = torch.complex(mask_low_specx[:,:L//2,:],mask_low_specx[:,L//2:,:])
        
        
        masked_x = batch_x_mean + stdev * torch.fft.irfft(low_spec_complex, n = L, dim = -2, norm = 'ortho')
        # not_mask_indicator = torch.fft.irfft(not_mask_indicator, n = L, dim = -2, norm = 'ortho')
        # not_mask_indicator = not_mask_indicator / (not_mask_indicator.mean(dim=-2,keepdim=True)+1e-5)
        # print(batch_x.shape)
        # print(masked_x.shape)
        mask_indicator = 1.0 - not_mask_indicator
        real_space_not_mask_indicator = torch.fft.irfft(torch.complex(not_mask_indicator[:,:L//2,:],not_mask_indicator[:,L//2:,:]), n = L, dim = -2, norm = 'ortho')
        return masked_x, not_mask_indicator
        
    def ttt_with_deform(self,old_x,batch_y,batch_x_mark,batch_y_mark):
        criterion = self._select_criterion_for_recons()
        self.deformation.reset_weight()
        optimizer = self._select_optimizer_for_deformationTTT()
        self.deformation.train()
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        previous_loss_list = []
        patience_counter = 0
        for _ in range(self.test_time_training_iter_big):
            
            for iter_idx in range(self.test_time_training_iter_small):
                optimizer.zero_grad()
                # batch_x = old_x[...,-192:,:]#.permute(0,2,1).permute(0,2,1)
                batch_x = self.deformation.input_transformation(old_x, self.model.seq_len)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    pass
                else:
                    batch_x_mark = self.deformation.input_transformation(batch_x_mark, self.model.seq_len, is_mark = True)
                
                masked_x, not_mask_indicator = self.mask_x(batch_x)
                
                
                
                if self.args.output_attention:
                    outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                else:
                    outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
            
                loss_recons = criterion(past_retrieved, batch_x)
                
                # loss_recons = criterion(past_retrieved, batch_x)
                
                # print("batch_x.std", batch_x.std(dim=-2,keepdim=True).mean())
                
                
                
                recon_loss_div_std = loss_recons / (1.0)
                # recon_loss_div_std = loss_recons / (batch_x.std(dim=-2, keepdim=True) + 1e-6)

                recon_loss_div_std = recon_loss_div_std.mean()
                
                if recon_loss_div_std > min(previous_loss_list, default = 1e6):
                    patience_counter += 1
                else:
                    patience_counter *= 0
                
                previous_loss_list.append(recon_loss_div_std.item())
                if patience_counter >= 100:
                    break
                print("ttt big iter {} small iter {}, recon loss / batch_x.std {}".format(_, iter_idx, recon_loss_div_std.item()))
                loss_recons = 1 * 1 * loss_recons
                loss_recons.backward(retain_graph=True)
                optimizer.step()
        self.deformation.eval()
        # print(self.deformation.ratios_weight)
        print(self.deformation.bias[0])
        print(self.deformation.ratio_minus_one[0])
        return self.deformation
        
    
    
    def test_time_train(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion = self._select_criterion_for_recons()
        self.reload_model()
        self.model.train()
        
        optimizer = self._select_optimizer_for_TTT()
        
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        previous_loss_list = []
        patience_counter = 0
        for _ in range(self.test_time_training_iter_big):
            masked_x, not_mask_indicator = self.mask_x(batch_x)
            for iter_idx in range(self.test_time_training_iter_small):
                
                
                optimizer.zero_grad()
                if self.args.output_attention:
                    outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                else:
                    outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
            
                loss_recons = criterion(past_retrieved, batch_x)
                
                # loss_recons = criterion(past_retrieved, batch_x)
                
                # print("batch_x.std", batch_x.std(dim=-2,keepdim=True).mean())
                
                
                
                recon_loss_div_std = loss_recons / (1.0)
                # recon_loss_div_std = loss_recons / (batch_x.std(dim=-2, keepdim=True) + 1e-6)

                recon_loss_div_std = recon_loss_div_std.mean()
                
                if recon_loss_div_std > min(previous_loss_list, default = 1e6):
                    patience_counter += 1
                else:
                    patience_counter *= 0
                
                previous_loss_list.append(recon_loss_div_std.item())
                if patience_counter >= 100:
                    break
                print("ttt big iter {} small iter {}, recon loss / batch_x.std {}".format(_, iter_idx, recon_loss_div_std.item()))
                loss_recons = 1 * -1 * loss_recons
                loss_recons.backward()
                optimizer.step()
        self.model.eval()
        return self.model
            
        
        
    
    
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_recons = self._select_criterion_for_recons()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_recon_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                
                
                
                masked_x, not_mask_indicator = self.mask_x(batch_x)
                

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                            outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
                            outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        
                        loss = criterion(outputs, batch_y)
                        loss_recons = criterion_recons(past_retrieved, batch_x)
                        
                        train_loss.append(loss.item())
                        
                        train_recon_loss.append(loss_recons.item())
                        if self.recon_loss_weight < 0:
                            loss = loss_recons
                        else:
                            loss = (loss + self.recon_loss_weight * loss_recons)/(1.0+self.recon_loss_weight)
                        
                        
                        
                else:
                    if self.args.output_attention:
                        outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                        outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,past_retrieved = self.model.forecast_and_retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
                        outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y)
                    pred_loss = loss.item()
                    loss_recons = criterion_recons(past_retrieved, batch_x)
                    recon_loss = loss_recons.item()
                    
                    
                    train_loss.append(loss.item())
                    # loss_recons = torch.tensor(0.0)
                    train_recon_loss.append(loss_recons.item())
                    if self.recon_loss_weight < 0:
                        loss = loss_recons
                    else:
                        loss = (loss + self.recon_loss_weight * loss_recons)/(1.0+self.recon_loss_weight)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total loss: {2:.7f} pred loss: {3:.7f} recon loss: {4:.7f}".format(i + 1, epoch + 1, loss.item(), pred_loss, recon_loss))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_recon_loss = np.average(train_recon_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train pred Loss: {2:.7f} Train recon Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, train_recon_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    
    
    
    
    def load_from_state_dict(self, state_dict):
        def get_new_dict(state_dict, name):
            new_dict = dict()
            for key,value in state_dict.items():
                if key.startswith(name):
                    new_dict[key[len(name)+1:]] = value
            return new_dict
        self.model.enc_embedding.load_state_dict(get_new_dict(state_dict,'enc_embedding'))
        self.model.mask_encoder.load_state_dict(get_new_dict(state_dict, 'mask_encoder'))
        self.model.projector.load_state_dict(get_new_dict(state_dict, 'projector'))
        self.model.past_projector.load_state_dict(get_new_dict(state_dict, 'past_projector'))
        self.model.encoder.load_state_dict(get_new_dict(state_dict,'encoder'))
        
        # randomly initialize self.model.encoder:
        # self.model.encoder = self._build_model().encoder.cuda()
        
    
        
        
        
        
        
        
        
        
        
        
    def test(self, setting, test=0, ttt = False, ttt_with_deform = False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            pass
            print('loading model')
            try:
                self.load_from_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            except:
                print("No trained model found. Using originally intialized model from", self.args.model_load_from)
                self.load_from_state_dict(torch.load(self.args.model_load_from))
            

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            # if i>5:
            #     break
            # if i > 1:
            #     break
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
            if ttt:
                self.test_time_train(batch_x, batch_y, batch_x_mark, batch_y_mark)
            if ttt_with_deform:
                deform = self.ttt_with_deform(batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            # batch_x = self.deformation.input_transformation(batch_x, self.model.seq_len)
            
            # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
            #     pass
            # else:
            #     batch_x_mark = self.deformation.input_transformation(batch_x_mark, self.model.seq_len, is_mark=True)
            #     batch_y_mark = self.deformation.input_transformation(batch_y_mark, self.model.label_len + self.model.seq_len, is_mark=True)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -96:, :]).float()
            # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            if ttt_with_deform:
                pass
                # outputs = self.deformation.output_transformation(outputs, 96)
                # outputs = self.deformation.output_transformation(outputs, self.args.pred_len)
            # print("outputs.shape==",outputs.shape)
            # print("batch_y.shape==",batch_y.shape)
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            # outputs = outputs[:, -96:, f_dim:]
            # batch_y = batch_y[:, -96:, f_dim:].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

            pred = outputs
            true = batch_y
            print("current mse:", torch.nn.MSELoss()(torch.tensor(pred),torch.tensor(true)).item())

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return