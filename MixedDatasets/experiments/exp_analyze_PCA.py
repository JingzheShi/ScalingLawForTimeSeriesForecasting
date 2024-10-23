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

def obtain_topk_normalized_vectors_with_eigenvalues(input_traj, K):
    # input_traj: N*D torch.tensor
    # output: components: K*D torch.tensor, eigenvalues: K-element torch.tensor
    
    # Ensure input is a numpy array for PCA compatibility
    input_traj_np = input_traj.numpy()
    
    # Apply PCA without explicitly centering data
    pca = decomposition.PCA(n_components=K)
    pca.fit(input_traj_np)
    
    # Extract the top K components and normalize them
    components = pca.components_  # Shape will be (K, D)
    
    # Eigenvalues (variance explained by each PC) from the PCA
    eigenvalues = pca.explained_variance_  # Shape will be (K,)
    
    # Convert the components and eigenvalues back to a torch tensor
    output_components = torch.tensor(components, dtype=torch.float)
    output_eigenvalues = torch.tensor(eigenvalues, dtype=torch.float)
    
    return output_components, output_eigenvalues
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


class Exp_Long_Term_Forecast_AnalyzePCA(Exp_Basic):
    def __init__(self,args):
        super().__init__(args)
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
        print("Now we only use training dataset for PCA!!!")
        data_set, data_loader = data_provider(self.args, 'train')
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
                        batch_y_mean = batch_y.mean(dim=-2,keepdim=False).unsqueeze(-2)
                        std_x = batch_x.std(dim=-2,keepdim=False).unsqueeze(-2)
                        std_y = batch_y.std(dim=-2,keepdim=False).unsqueeze(-2)
                        # to_append = (batch_y - batch_y_mean) / (std_y + 1e-5)
                        to_append = batch_y
                        batch_y_list.append(to_append.detach().cpu())
                    ys = torch.cat(batch_y_list, dim=0)[...,0] # N,T,1->N,T
                    print("ys.shape==",ys.shape)
                    mean = ys.mean(dim=0) # T
                    deltas = ys-mean.unsqueeze(0)
                    pcas, eig_vals = obtain_topk_normalized_vectors_with_eigenvalues(deltas,1000)
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
                        eig_vals = eig_vals,
                    ),f'/root/autodl-tmp/pcas_for_weather_Hor=2000_numsamples={ys.shape[0]}.pt')
                    assert False, 'generation Done!'