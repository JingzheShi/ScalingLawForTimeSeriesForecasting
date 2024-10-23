from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar# Dataset_Solar, Dataset_PEMS, \
   # Dataset_Pred
from torch.utils.data import Dataset,DataLoader
import torch


from random import shuffle
import math

def create_square_wave(num, total_length, NUM_LEVELS = 10, T_min = 5, T_max = 1000):
    # Initialize the index tensor representing time steps.
    index = torch.arange(0, total_length)
    index = index.unsqueeze(0).unsqueeze(0).repeat(num, NUM_LEVELS, 1)  # (num,NUM_LEVELS, total_length)

    
    
    # Generate random periods for each wave within the specified range.
    T = torch.randint(low=T_min, high=T_max, size=(num, NUM_LEVELS))  # (num, NUM_LEVELS)
    T = T.unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    bias = torch.rand(num, NUM_LEVELS).unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    bias = (bias * (T.float())).long()  # (num, NUM_LEVELS, total_length)

    # Calculate the duty cycle (proportion of time the signal is high vs low).
    # This example assumes a 50% duty cycle for simplicity.
    duty_cycle = torch.rand(num, NUM_LEVELS)*0.925+0.05  # (num, NUM_LEVELS)
    duty_cycle = duty_cycle.unsqueeze(-1).repeat(1, 1, total_length)  # (num, NUM_LEVELS, total_length)
    # Initialize the square wave tensor.
    square_wave = torch.zeros(num, NUM_LEVELS, total_length)

    # Determine the high and low states of the square wave at each point in time.
    # for i in range(num):
    #     # Calculate the state (high or low) based on the current point in the period.
    #     square_wave[i] = torch.floor(((index[i]+bias[i]) % T[i]) / (T[i] * duty_cycle[i])) * 2 - 1  # Results in values of -1 or 1
    
    square_wave = torch.floor(((index+bias)%T)/(T*duty_cycle))*2-1
    
    weights = torch.rand(num,NUM_LEVELS) + 1e-4
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    square_wave = (square_wave * weights.unsqueeze(2)).sum(dim=1) # (num, total_length)
    
    return square_wave/2
def create_sinosoid(num, total_length, NUM_FREQ = 2, T_min = 5, T_max = 1000):
    # set random seed
    import time
    torch.manual_seed(int(time.time()))
    index = torch.arange(0, total_length).float()
    index = index.unsqueeze(0).unsqueeze(0).repeat(num, NUM_FREQ, 1) # (num, 1, total_length)
    T = torch.rand(num, NUM_FREQ) * (T_max - T_min) + T_min # (num, NUM_FREQ)
    freq = 2 * math.pi / T # (num, NUM_FREQ)
    bias = torch.rand(num, NUM_FREQ) * 2 * math.pi # (num, NUM_FREQ)
    sinosoid = torch.sin(index * freq.unsqueeze(2) + bias.unsqueeze(2)) # (num, NUM_FREQ, total_length)
    weights = torch.rand(num,NUM_FREQ) + 1e-4
    weights = weights / weights.sum(dim=1, keepdim=True)
    sinosoid = (sinosoid * weights.unsqueeze(2)).sum(dim=1) # (num, total_length)
    return sinosoid
    

def add_noise(tensor, noise_std = 0.02):
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise


    
FUNC_LIST = [create_sinosoid, create_square_wave]

    
def create_sequence(num, total_length, TYPE=2, noise_std=0.06):
    # returns a tensor of shape (num, total_length)
    weights = torch.rand(num, TYPE) + 1e-4
    weights = torch.nn.functional.softmax(weights*4, dim=1)
    seq = torch.zeros(num, TYPE, total_length).float()
    for idx in range(TYPE):
        seq[:,idx] = FUNC_LIST[idx](num, total_length)
    seq = (seq * weights.unsqueeze(2)).sum(dim=1)
    seq = add_noise(seq, noise_std)
    return seq
    




class Dataset_self_created(Dataset):
    def __init__(self,root_path,flag='train',size=None,
                 features='S',data_path=None,
                 target='OT',scale=True,timeenc=0,freq='h',all_data_length=75000,percent=100):
        super(Dataset_self_created, self).__init__()
        assert percent==100
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag = flag
        self.set_type = type_map[flag]
        

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.all_data_length = all_data_length
        self.__read_data__()
        
    def __read_data__(self):
        num_train = int(self.all_data_length*0.7)
        num_test = int(self.all_data_length*0.001)
        num_valid = self.all_data_length - num_train - num_test
        self.len_list = [num_train, num_valid, num_test]
        length = self.len_list[self.set_type]
        self.datas = create_sequence(length, self.seq_len + self.pred_len).unsqueeze(-1)
        
        
        
    def resample(self):
        print("Resample!")
        length = self.len_list[self.set_type]
        self.datas = create_sequence(length, self.seq_len + self.pred_len).unsqueeze(-1)
        
    def __getitem__(self,index):
        data = self.datas[index]
        if index == self.len_list[self.set_type] - 1:
            self.resample()
        seq_x = data[:self.seq_len]
        seqy = data[self.seq_len - self.label_len:self.seq_len + self.pred_len] 
        seq_x_mark = torch.zeros((seq_x.shape[0],1))
        seq_y_mark = torch.zeros((seqy.shape[0],1))
        return seq_x, seqy, seq_x_mark, seq_y_mark
        
        
    def __len__(self):
        return self.len_list[self.set_type]
        



data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    # 'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'self_created': Dataset_self_created,
}

class Dataset_Mix(Dataset):
    def __init__(self,datasetsDict_list=None,flag='test',batch_size=1,args=None,shrimp_aug=False,shrimp_coef=1.0):
        
        self.dataloaderIterator_list = []
        self.dataloader_list = []
        
        self.dataloader_length_list = []
        
        self.dataloader_idx_list_for_one_iter = []
        
        for (dataset_name, dataset_args) in datasetsDict_list:
            Data = data_dict[dataset_name]
            timeenc = 0 if dataset_args['embed'] != 'timeF' else 1
            if flag == 'test':
                shuffle_flag = False
                drop_last = False
                batch_size = batch_size
                freq = dataset_args['freq']
            elif flag == 'pred':
                shuffle_flag = False
                drop_last = False
                batch_size = batch_size
                freq = dataset_args['freq']
                Data = Dataset_Pred
            else:
                shuffle_flag = True
                drop_last = False
                batch_size = batch_size
                freq = dataset_args['freq']
            shrimp_aug = shrimp_aug and flag == 'train'
            if shrimp_aug:
                assert shrimp_coef >= 1
            self.shrimp_aug = shrimp_aug
            self.shrimp_coef = shrimp_coef
            
            self.true_seq_len = int(dataset_args['seq_len'] * shrimp_coef) if shrimp_aug else dataset_args['seq_len']
            self.true_label_len = int(dataset_args['label_len'] * shrimp_coef) if shrimp_aug else dataset_args['label_len']
            self.true_pred_len = int(dataset_args['pred_len'] * shrimp_coef) if shrimp_aug else dataset_args['pred_len']
            self.seq_len = dataset_args['seq_len']
            self.label_len = dataset_args['label_len']
            self.pred_len = dataset_args['pred_len']
            print("seq_len:{}, label_len:{}, pred_len:{}".format(self.true_seq_len, self.true_label_len, self.true_pred_len))
            
            
            data_set = Data(
                root_path=dataset_args['root_path'],
                data_path=dataset_args['data_path'],
                flag=flag,
                size=[self.true_seq_len, self.true_label_len, self.true_pred_len],
                features=dataset_args['features'],
                target=dataset_args['target'],
                timeenc=timeenc,
                freq=freq,
            )
            print(dataset_name, flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=dataset_args['num_workers'],
                drop_last=drop_last)
            
            dataloader_iterator = iter(data_loader)
                
            self.dataloader_list.append(data_loader)
            self.dataloaderIterator_list.append(dataloader_iterator)
            self.dataloader_length_list.append(len(dataloader_iterator))
        self.lenth = sum(self.dataloader_length_list)
        for dataset_idx, dataloader_length in enumerate(self.dataloader_length_list):
            self.dataloader_idx_list_for_one_iter.extend([dataset_idx]*dataloader_length)
            
        self.shuffle = shuffle_flag
        print("Total batch number: ", self.lenth)
    
        if self.shuffle:
            self.update_dataloader_idx_list_for_one_iter()
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = True
        
        
        
        
        
    
    def update_dataloader_idx_list_for_one_iter(self):
        shuffle(self.dataloader_idx_list_for_one_iter)
    
    def __len__(self):
        return self.lenth
    
    def interpolate_tensor(self,tensor,to_len):
        return torch.nn.functional.interpolate(tensor.transpose(-2,-1),size=to_len,mode='linear').transpose(-2,-1)
        
    
    
    def __getitem__(self, idx):
        
        dataset_idx = self.dataloader_idx_list_for_one_iter[idx]
        
        try:
            batch = next(self.dataloaderIterator_list[dataset_idx])
        except StopIteration:
            self.dataloaderIterator_list[dataset_idx] = iter(self.dataloader_list[dataset_idx])
            batch = next(self.dataloaderIterator_list[dataset_idx])
        
        # for item in batch:
        #     print(item.shape)
        # assert False
        
        
        if self.shrimp_aug:
            randtensor = self.shrimp_coef**(-2+2*torch.rand(1)) # 1/shrimp_coef**2 ~ 1
            randseqlength = (randtensor*self.true_seq_len).long() # 1/shrimp_coef*seq_len ~ seq_len
            randlabellength = (randtensor*self.true_label_len).long() # 1/shrimp_coef*label_len ~ label_len
            randpredlength = (randtensor*self.true_pred_len).long() # 1/shrimp_coef*pred_len ~ pred_len
            
            seq_x_cut = self.interpolate_tensor(batch[0][:, -randseqlength:,:],self.seq_len)
            seq_y_cut = self.interpolate_tensor(batch[1][:, self.true_label_len-randlabellength: self.true_label_len+randpredlength,:], self.label_len + self.pred_len)
            seq_x_mark_cut = self.interpolate_tensor(batch[2][:, -randseqlength:,:], self.seq_len)
            seq_y_mark_cut = self.interpolate_tensor(batch[3][:, self.true_label_len-randlabellength: self.true_label_len+randpredlength,:], self.label_len + self.pred_len)
            # print(seq_x_cut.shape, seq_y_cut.shape, seq_x_mark_cut.shape, seq_y_mark_cut.shape)
            batch = (seq_x_cut, seq_y_cut, seq_x_mark_cut, seq_y_mark_cut)
            
            
        
        return batch
            
            
            
            
        
        


def data_provider(args, flag):
    if args.data == 'Mix':
        Data = Dataset_Mix
        if flag not in  ['test','pred']:
            shuffle_flag = True
        else:
            shuffle_flag = False
        data_set = Data(
            datasetsDict_list=args.datasetsDict_list,
            flag=flag,
            batch_size=args.batch_size,
            args = args,
        )
        def dummy_collate(list):
            assert len(list) == 1, 'Mix dataset only support big batch_size=1, since each sample is composed of batch_size samples from one single dataset.'
            return list[0]
        data_loader = DataLoader(
            data_set,
            batch_size = 1,
            shuffle = shuffle_flag,
            num_workers=args.num_workers,
            collate_fn = dummy_collate,
            )
        return data_set, data_loader
        
    Data = data_dict[args.data]
    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    try:
        if args.data == 'custom':
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent = args.percent,
                starting_percent = args.starting_percent,
            )
        else:
            if args.five_percent_few_shot or args.ten_percent_few_shot:
                data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent = 5 if args.five_percent_few_shot else 10 if args.ten_percent_few_shot else 100,
                )
                print("Now we use percent = ", 5 if args.five_percent_few_shot else 10 if args.ten_percent_few_shot else 100)
            else:
                data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent = args.percent,
                )
                print("Now we use percent = ", args.percent)
    except Exception as e:
        print(e)
        data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
