"""
Centralized model class instantiation based on args - PyTorch Version
"""

from models import MTMFSeq2Seq, MTMFSeq2SeqPred
from helpers import _baseMFDP, MFDPOneStep, MFDPMultiStep
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ClsConstructor():

    def __init__(self, args):
        self.args = args
        self.supported_models = ['MTMFSeq2Seq',
                                 'MTMFSeq2One',
                                 'MLP',
                                 'GBM']
        assert args.model_type in self.supported_models
        
        
    def create_data_processor(self):
        
        args = self.args
        if hasattr(args, 'scaler_type') and args.scaler_type == 'standard':
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
        elif hasattr(args, 'scaler_type') and args.scaler_type == 'minmax':
            scaler_x, scaler_y = MinMaxScaler((-1,1)), MinMaxScaler((-1,1))
        else:
            # Default to minmax
            scaler_x, scaler_y = MinMaxScaler((-1,1)), MinMaxScaler((-1,1))
    
        if args.model_type in ['MTMFSeq2Seq']:
            dp = MFDPMultiStep(Lx = args.Lx,
                               Tx = args.Tx,
                               Ly = args.Ty - 1,  # Ly should be Ty - 1
                               Ty = args.Ty,
                               freq_ratio = args.freq_ratio,
                               scaler_x = scaler_x,
                               scaler_y = scaler_y,
                               zero_pad = getattr(args, 'zero_pad', True))
        elif args.model_type in ['MTMFSeq2One', 'MLP', 'GBM']:
            dp = MFDPOneStep(Lx = args.Lx,
                           Tx = args.Tx,
                           Ly = args.Ty - 1,
                           Ty = args.Ty,
                           freq_ratio = args.freq_ratio,
                           scaler_x = scaler_x,
                           scaler_y = scaler_y,
                           zero_pad = getattr(args, 'zero_pad', True))
        
        return dp

    def create_model(self):
        
        args = self.args
        
        if args.model_type == 'MTMFSeq2Seq':
            model = MTMFSeq2Seq(dim_x = args.dim_x,
                               dim_y = args.dim_y,
                               Lx = args.Lx,
                               Tx = args.Tx,
                               Ty = args.Ty,
                               n_a = args.n_a,
                               n_s = args.n_s,
                               n_align_x = args.n_align_x,
                               n_align_y = args.n_align_y,
                               fc_x = args.fc_x,
                               fc_y = args.fc_y,
                               dropout_rate = args.dropout_rate,
                               freq_ratio = args.freq_ratio,
                               bidirectional_encoder = getattr(args, 'bidirectional_encoder', False),
                               l1reg = getattr(args, 'l1reg', 1e-5),
                               l2reg = getattr(args, 'l2reg', 1e-4))
        else:
            raise NotImplementedError(f'Model {args.model_type} not implemented yet')
        
        return model

    def create_predictor(self, model, dp, apply_inv_scaler = True):
    
        args = self.args
        
        if args.model_type == 'MTMFSeq2Seq':
            predictor = MTMFSeq2SeqPred(model, dp.scaler_x, dp.scaler_y, apply_inv_scaler=apply_inv_scaler)
        else:
            raise NotImplementedError(f'Predictor for {args.model_type} not implemented yet')
        
        return predictor