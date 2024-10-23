import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, iTransformer_wMask, iTransformer_wFreqMask, DLinear, iMLP_wMask, iTransformer_wMask_adaptive, iMLP_wMask_adaptive, iMLP_res, iMLP_wMask_res, iMLP_res_gate, iMLP_res_gate_revin, PCALinear, iMLP_pure, iMLP_pyramid, iMLP_res_gate_patch,\
        iMLP_gate_narrow,iMLP_gate_narrow_patch,Linear_patch,Linear,iMLP_res_gate_patch_test,Linear_interpolate,MLP_interpolate,MLP_res_interpolate


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'iTransformer_wMask': iTransformer_wMask,
            'iTransformer_wMask_adaptive': iTransformer_wMask_adaptive,
            'iTransformer_wFreqMask': iTransformer_wFreqMask,
            'DLinear': DLinear,
            'iMLP_wMask': iMLP_wMask,
            'iMLP_wMask_adaptive': iMLP_wMask_adaptive,
            'iMLP_res': iMLP_res,
            'iMLP_wMask_res': iMLP_wMask_res,
            'iMLP_res_gate': iMLP_res_gate,
            'iMLP_res_gate_revin': iMLP_res_gate_revin,
            'PCALinear': PCALinear,
            'iMLP_pure': iMLP_pure,
            'iMLP_pyramid': iMLP_pyramid,
            'iMLP_res_gate_patch': iMLP_res_gate_patch,
            'iMLP_gate_narrow': iMLP_gate_narrow,
            'iMLP_gate_narrow_patch': iMLP_gate_narrow_patch,
            'Linear_patch': Linear_patch,
            'Linear':Linear,
            'iMLP_res_gate_patch_test':iMLP_res_gate_patch_test,
            'Linear_interpolate':Linear_interpolate,
            'MLP_interpolate':MLP_interpolate,
            'MLP_res_interpolate':MLP_res_interpolate
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
