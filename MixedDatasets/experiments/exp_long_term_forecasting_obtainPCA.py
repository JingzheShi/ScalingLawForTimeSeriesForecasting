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
from sklearn import decomposition

warnings.filterwarnings('ignore')
def obtain_topk_normalized_vectors(input_traj, K):
    # input_traj: N*D torch.tensor
    # output: K*D torch.tensor
    
    # Ensure input is a numpy array for PCA compatibility
    input_traj_np = input_traj.numpy()
    
    # Apply PCA without explicitly centering data
    pca = decomposition.PCA(n_components=K)
    pca.fit(input_traj_np)
    
    # Extract the top K components and normalize them
    components = pca.components_  # Shape will be (K, D)
    
    # Convert the components back to a torch tensor
    output = torch.tensor(components, dtype=torch.float)
    
    return output
# def obtain_topk_normalized_vectors(input_traj, K):
#     # Center the trajectories by subtracting the mean
#     mean = torch.mean(input_traj, dim=0)
#     centered_traj = input_traj - mean
    
#     # Compute the PCA using torch.pca_lowrank
#     U, S, V = torch.pca_lowrank(centered_traj, q=K, niter=100)
    
#     # Normalize the vectors to have unit length
#     # V contains the right singular vectors, which correspond to the PCA directions
#     # They are already normalized if you use torch.pca_lowrank, but you can explicitly normalize if needed
#     V_normalized = V
    
#     return V_normalized.t()  # Return KxD tensor representing top K PCA directions

class Exp_Long_Term_Forecast_obtainPCA(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_obtainPCA, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.model_load_from is not None:
            try:
                model.load_state_dict(torch.load(self.args.model_load_from,map_location='cpu'),strict=False)
            except Exception as e:
                print("Loading Error! Now we do not load projector parameters.")
                new_dict = dict()
                old_dict = torch.load(self.args.model_load_from,map_location='cpu')
                for k, v in old_dict.items():
                    if not k.startswith('projector.'):
                        new_dict[k] = v
                model.load_state_dict(new_dict, strict=False)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print("Model size:", str(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)+'M')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # param_list = []
        # freeze_param_name_list = [
        #      'enc_embedding', 'encoder', #'projector',# 'mask_encoder','enc_embedding', 
        # ]
        # print("NOW WE FREEZE THE FOLLOWING PARAMETER:",freeze_param_name_list)
        # for name, param in self.model.named_parameters():
        #     if (not any(name.startswith(freeze_param_name + '.') for freeze_param_name in freeze_param_name_list)):
        #         param_list.append(param)
        # # for name, param in self.model.named_parameters():
        # #     if (not name.startswith('projector.')):
        # #         param_list.append(param)
        # model_optim = optim.Adam(param_list, lr=self.args.learning_rate, weight_decay=0.0)
        
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay = 1e-4)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            batch_y_list = list()
            from tqdm import tqdm
            if 1:
                with torch.no_grad():
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                        if i > 2000:
                            break
                        iter_count += 1
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)

                        batch_y = batch_y.float().to(self.device)
                        batch_x_mean = batch_x.mean(dim=-2,keepdim=False).unsqueeze(-2)
                        std_x = batch_x.std(dim=-2,keepdim=False).unsqueeze(-2)
                        to_append = (batch_y - batch_x_mean) / (std_x + 1e-1)
                        batch_y_list.append(to_append.detach().cpu())
                    ys = torch.cat(batch_y_list, dim=0)[...,0] # N,T,1->N,T
                    print("ys.shape==",ys.shape)
                    mean = ys.mean(dim=0) # T
                    deltas = ys-mean.unsqueeze(0)
                    pcas = obtain_topk_normalized_vectors(deltas, 200)
                    pcas_norm = (pcas**2).sum(dim=-1).sqrt()
                    pcas = pcas / pcas_norm.unsqueeze(-1)
                    for idx in range(10):
                        for idx2 in range(10):
                            if idx != idx2:
                                print("pcas[",idx,"] dot pcas[",idx2,"]==",torch.sum(pcas[idx]*pcas[idx2]))
                            else:
                                print("pcas[",idx,"] dot pcas[",idx2,"]==",torch.sum(pcas[idx]*pcas[idx2]))
                    # print("pcas[0] dot pcas[1]==",torch.sum(pcas[0]*pcas[1]))
                    print("mean==",mean)
                    torch.save(dict(
                        mean=mean,
                        pcas=pcas,
                    ),'/root/autodl-tmp/pcas_for_weather_336+0.pt')
                    assert False, 'generation Done!'
            else:
                mse = 0.0
                mae = 0.0
                TOTAL_ITER = 1000
                num_sample = 0
                with torch.no_grad():
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                        if i <= 2500:
                            continue
                        if i > 2500 + TOTAL_ITER:
                            break
                        batch_x = batch_x.float()
                        batch_y = batch_y.float()
                        num_sample += batch_y.shape[0]
                        mean_and_pcas_dict = torch.load('/root/autodl-tmp/pcas_for_weather_336+192.pt')
                        mean = mean_and_pcas_dict['mean'].float()
                        pcas = mean_and_pcas_dict['pcas'].float() # K,T. K = 100
                        batch_x_mean = batch_x.mean(dim=-2,keepdim=False).unsqueeze(-2)
                        std_x = batch_x.std(dim=-2,keepdim=False).unsqueeze(-2)
                        
                        y_norm = (batch_y - batch_x_mean) / (std_x + 1e-5)
                        # print(y_norm.shape)
                        # let pcas be identity matrix.
                        # pcas = torch.eye(y_norm.shape[-2],device=pcas.device)
                        y_deltas = (y_norm-mean.unsqueeze(0).unsqueeze(-1))[...,0]
                        masked_y_deltas = torch.zeros_like(y_deltas)
                        masked_y_deltas[:,:336] = y_deltas[:,:336]
                        # print(y_deltas.shape)
                        # B, T, 1 -> B, T
                        # use the top PCAs to project the deltas.
                        projecteted_deltas = torch.matmul(masked_y_deltas, pcas.t()) # B,T x T,K -> B,K
                        # print("projecteted_deltas.shape==",projecteted_deltas.shape)
                        # print("sum(projecteted_deltas**2)==",(projecteted_deltas**2).sum(dim=-1))
                        
                        reconstructed_y_delta = torch.matmul(projecteted_deltas, pcas) # B, T
                        # print(torch.sum((y_deltas - reconstructed_y_delta)**2))
                        # print("reconstructed_y_delta.shape==",reconstructed_y_delta.shape)
                        # print('batch_y.shape==',batch_y.shape)
                        # print(((y_deltas+mean.unsqueeze(0)-y_norm[...,0])**2).sum())
                        reconstructed_y = (reconstructed_y_delta.unsqueeze(-1)+mean.unsqueeze(0)) * (std_x + 1e-5) + batch_x_mean
                        
                        loss = reconstructed_y[...,-192:,0] - batch_y[...,-192:,0]
                        currentmse = (loss**2).mean().item()
                        currentmae = (loss.abs()).mean().item()
                        # print("Current mse, mae == ",currentmse,currentmae)
                        mse += currentmse
                        mae += currentmae
                print("mse==",mse/TOTAL_ITER)
                print("mae==",mae/TOTAL_ITER)
                assert False
                        
                
                
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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            try:
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            except:
                print("No trained model found. Using originally intialized model from", self.args.model_load_from)
                self.model.load_state_dict(torch.load(self.args.model_load_from,map_location='cpu'),strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
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

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # from tqdm import tqdm
        mae_sum = 0.0
        mse_sum = 0.0
        for idx in tqdm(range(len(preds))):
            mae_sum += np.mean(np.abs(preds[idx] - trues[idx]))
            mse_sum += np.mean((preds[idx] - trues[idx]) ** 2)
        mae = mae_sum / len(preds)
        mse = mse_sum / len(preds)
        print("")
        
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

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