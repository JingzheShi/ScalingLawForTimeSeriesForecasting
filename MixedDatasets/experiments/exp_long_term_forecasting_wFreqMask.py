from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from random import uniform
from copy import deepcopy

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_wFreqMask(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_wFreqMask, self).__init__(args)
        self.cut_freq = args.cut_freq if args.cut_freq is not None else args.seq_len // 2
        self.mask_ratio = 0.7
        print('mask_ratio=',self.mask_ratio)
        self.recon_loss_weight = args.recon_loss_weight
        print('recon_loss_weight=',self.recon_loss_weight)
        self.test_time_training_iter_big = 10 # mse:0.3711753487586975, mae:0.23760731518268585 with 0
        self.test_time_training_iter_small = 1
        self.model_state_dict = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.model_load_from is not None:
            model.load_state_dict(torch.load(self.args.model_load_from))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print("Model size:", str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)+'M')
        return model

    def reload_model(self):
        assert self.args.model_load_from is not None
        if self.model_state_dict is None:
            self.model_state_dict = torch.load(self.args.model_load_from)
        self.model.load_state_dict(self.model_state_dict)
        

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_optimizer_for_TTT(self):
        # optimizer for test-time-training.
        # model_optim = optim.Adam(self.model.parameters(), lr=1e-1,weight_decay = 1e-2)
        param_list = []
        freeze_param_name_list = [
            'projector', 'past_projector', 
        ]
        # freeze_param_name_list = [
        #     'encoder', 'enc_embedding','mask_encoder',
        # ]
        freeze_param_name_list = [
            'projector', 'enc_embedding','mask_encoder',
        ]
        for name, param in self.model.named_parameters():
            if (not any(name.startswith(freeze_param_name + '.') for freeze_param_name in freeze_param_name_list)):
                param_list.append(param)
        # for name, param in self.model.named_parameters():
        #     if (not name.startswith('projector.')):
        #         param_list.append(param)
        model_optim = optim.SGD(param_list, lr=0.01, weight_decay=0.01, momentum=0.9)
        # model_optim = optim.Adam(param_list, lr=1e-3, weight_decay=0.01)
        return model_optim
    def _select_criterion(self):
        return nn.MSELoss()
    def _select_criterion_for_training(self):
        return nn.MSELoss()
        def loss_function(pred, target):
            assert len(pred.shape) == 3
            assert pred.shape == target.shape
            finidx = pred.shape[-2] // 2
            diff = pred - target # B cut_freq*2 N
            one_through_finidx = torch.arange(1,finidx+1,device = pred.device).float()
            weight = torch.cat([one_through_finidx,one_through_finidx],dim = 0)
            weight = weight.unsqueeze(0).unsqueeze(-1)
            weight = weight ** -(1/8)
            weight = weight / weight.mean()
            diff = diff * weight
            return (diff.pow(2)).mean()
            
            
            total_element_number = pred.shape[0] * pred.shape[1] * pred.shape[2]
            return (((pred[:,0,:]-target[:,0,:]).pow(2)).sum() + ((pred[:,finidx,:]-target[:,finidx,:]).pow(2)).sum() + ((pred-target).pow(2)).sum()) / total_element_number
            # this is because the 0th freq after fft has double weight than the other freqs.
        return loss_function
        criterion = nn.MSELoss()
        return criterion
    
    def _select_criterion_for_retrieve(self):
        
        # input: pred, target of shape B, cut_freq * 2, N.
        # calculate L2 loss in freq space, for each freq. Note that the 0th freq of the cut_freq freqs has double weight.
        
        def loss_function(pred, target):
            assert len(pred.shape) == 3
            assert pred.shape == target.shape
            finidx = pred.shape[-2] // 2
            total_element_number = pred.shape[0] * pred.shape[1] * pred.shape[2]
            return (((pred[:,0,:]-target[:,0,:]).pow(2)).sum() + ((pred[:,finidx,:]-target[:,finidx,:]).pow(2)).sum() + ((pred-target).pow(2)).sum()) / total_element_number
            # this is because the 0th freq after fft has double weight than the other freqs.
        return loss_function

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
        # ratio = uniform(0,self.mask_ratio)
        ratio = self.mask_ratio
        # print('current ratio is', ratio)
        # batch_x: B L N
        not_mask_indicator = torch.rand([batch_x.shape[0],self.cut_freq*2,batch_x.shape[2]],device = batch_x.device) > ratio
        # not_mask_indicator is a boolean tensor with the same shape as batch_x.
        not_mask_indicator = not_mask_indicator.float()
        
        return deepcopy(batch_x), not_mask_indicator
        
        
    def test_time_train(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        self.reload_model()
        
        self.model.train()
        
        optimizer = self._select_optimizer_for_TTT()
        
        criterion = self._select_criterion_for_retrieve()
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        for _ in range(self.test_time_training_iter_big):
            not_mask_indicator = (torch.rand([batch_x.shape[0],self.cut_freq*2,batch_x.shape[2]],device = batch_x.device) > self.mask_ratio).float()
            masked_x = batch_x
            for iter_idx in range(self.test_time_training_iter_small):
                
                
                optimizer.zero_grad()
                if self.args.output_attention:
                    target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                else:
                    target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
            
                loss_recons = criterion(target, past_retrieved)
                if loss_recons < 0.06:
                    break
                print("ttt big iter {} small iter {}, recon loss {}".format(_, iter_idx, loss_recons.item()))
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
        criterion = self._select_criterion_for_training()
        criterion_test = self._select_criterion()
        criterion_past_retrieve = self._select_criterion_for_retrieve()

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
                        f_dim = -1 if self.args.features == 'MS' else 0
                        if self.args.output_attention:
                            retrieve_target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                            outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            # outputs, gts = self.model.forecast_when_training(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y[:, -self.args.pred_len:, f_dim:].to(self.device))[0]
                        else:
                            retrieve_target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
                            outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # outputs, gts = self.model.forecast_when_training(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y[:, -self.args.pred_len:, f_dim:].to(self.device))
                            
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        gts = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        
                        
                        loss = criterion(outputs, gts)
                        loss_recons = criterion_past_retrieve(past_retrieved*(1-not_mask_indicator), retrieve_target*(1-not_mask_indicator))
                        
                        train_loss.append(loss.item())
                        
                        train_recon_loss.append(loss_recons.item())
                        loss = (loss + self.recon_loss_weight * loss_recons)/(1.0+self.recon_loss_weight)
                        
                        
                        
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0
                    if self.args.output_attention:
                        retrieve_target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)[0]
                        outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        retrieve_target,past_retrieved = self.model.retrieve(masked_x, batch_x_mark, dec_inp, batch_y_mark, not_mask_indicator)
                        outputs = self.model.forecast(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    gts = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, gts)
                    
                    
                    # print(not_mask_indicator.shape)
                    # print(past_retrieved.shape)
                    # print(retrieve_target.shape)
                    loss_recons = criterion_past_retrieve(past_retrieved*(1-not_mask_indicator), retrieve_target*(1-not_mask_indicator))
                    
                    
                    train_loss.append(loss.item())
                    # loss_recons = torch.tensor(0.0)
                    train_recon_loss.append(loss_recons.item())
                    recon_loss = loss_recons.item()
                    pred_loss = loss.item()
                    loss = (loss + self.recon_loss_weight * loss_recons)/(1.0+self.recon_loss_weight)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | total loss: {2:.7f} | pred_loss={3:.7f}, recon_loss={4:.7f}".format(i + 1, epoch + 1, loss.item(),pred_loss, recon_loss))
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
            vali_loss = self.vali(vali_data, vali_loader, criterion_test)
            test_loss = self.vali(test_data, test_loader, criterion_test)

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

    def test(self, setting, test=0, ttt = False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            try:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            except:
                print("No trained model found. Using originally intialized model from", self.args.model_load_from)
                self.model.load_state_dict(torch.load(self.args.model_load_from))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            if i>100:
                break
            # if i > 10:
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
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

            pred = outputs
            true = batch_y

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

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

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