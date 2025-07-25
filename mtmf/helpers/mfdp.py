"""
Data processor for parsing the input x (high-freq) and y (low-freq) series
PyTorch Version - Fixed for sample size synchronization
"""

import math
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class _baseMFDP():
    """Base class that hosts some helper functions for processing the data"""
    
    def __init__(self, Lx, Tx, Ly, Ty, freq_ratio, scaler_x=MinMaxScaler((-1,1)), scaler_y=MinMaxScaler((-1,1))):
        
        self.Lx = Lx
        self.Tx = Tx
        self.Ly = Ly
        self.Ty = Ty
        
        self.freq_ratio = freq_ratio
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        
    def find_length(self, target_sample_size, len_input, len_target=1, stride=1):
        """Find the length of the time points required given target_sample_size"""
        return (target_sample_size - 1) * stride + len_input + len_target
    
    def sample_size_calc(self, len_total, len_input, len_target, stride=1):
        """Calculate the number of available supervised samples"""
        return math.floor((len_total - len_input - len_target) / float(stride)) + 1
        
    def supervised_sample_helper(self, x, T_input, T_target, stride=1):
        """
        Helper function for generating samples in the input-target form
        
        Args:
            x: (np.array) features in cols and time steps in rows
            T_input: (int) length/timepoints of the input
            T_target: (int) length/timepoints of the target
            stride: int -- stride size
        
        Returns:
            input: np.array (sample_size, T_input, # of features)
            target: np.array (sample_size, T_target, # of features)
        """
        sample_size = self.sample_size_calc(x.shape[0], T_input, T_target, stride=stride)
        
        list_input, list_target = [], []
        for i in range(sample_size):
            input_start = int(i * stride)
            input_end = int(input_start + T_input) - 1
            target_start = int(input_start + T_input)
            target_end = target_start + T_target - 1
            
            list_input.append(x[input_start:(input_end+1), :])
            list_target.append(x[target_start:(target_end + 1), :])
            
        input_array, target_array = np.array(list_input), np.array(list_target)
        return input_array, target_array

    def supervised_sample_helper_with_size(self, x, T_input, T_target, stride, sample_size):
        """
        Helper function that generates EXACTLY sample_size samples (original TensorFlow approach)
        
        Args:
            x: (np.array) features in cols and time steps in rows
            T_input: (int) length/timepoints of the input
            T_target: (int) length/timepoints of the target
            stride: int -- stride size
            sample_size: int -- exact number of samples to generate
        
        Returns:
            input: np.array (sample_size, T_input, # of features)
            target: np.array (sample_size, T_target, # of features)
        """
        list_input, list_target = [], []
        for i in range(sample_size):  # Use the provided sample_size
            input_start = int(i * stride)
            input_end = int(input_start + T_input) - 1
            target_start = int(input_start + T_input)
            target_end = target_start + T_target - 1
            
            list_input.append(x[input_start:(input_end+1), :])
            list_target.append(x[target_start:(target_end + 1), :])
            
        input_array, target_array = np.array(list_input), np.array(list_target)
        return input_array, target_array

    def apply_scaler(self, scaler_name, x, inverse=False):
        """Apply scaling to data"""
        scaler = getattr(self, scaler_name)
        if len(x.shape) == 3:  # (batch_size, len, dim)
            xnew = np.zeros(x.shape)
            if not inverse:
                for i in range(x.shape[0]):
                    xnew[i, :, :] = scaler.transform(x[i, :, :])
            else:
                for i in range(x.shape[0]):
                    xnew[i, :, :] = scaler.inverse_transform(x[i, :, :])
        else:  # (sample_size, dim)
            if not inverse:
                xnew = scaler.transform(x)
            else:
                xnew = scaler.inverse_transform(x)
        return xnew

class MFDPMultiStep(_baseMFDP):
    """Multi-step data processor for seq2seq models"""

    def __init__(self, Lx, Tx, Ly, Ty, freq_ratio, scaler_x=MinMaxScaler((-1,1)), 
                 scaler_y=MinMaxScaler((-1,1)), zero_pad=True):
        super().__init__(Lx, Tx, Ly, Ty, freq_ratio, scaler_x, scaler_y)
        self.zero_pad = zero_pad
            
    def mf_sample_generator(self, x, y, update_scaler=False, apply_scaler=False, verbose=False):
        """
        Mixed frequency sample generator for training/validation
        
        Args:
            x: high-frequency data, shape (time_steps, dim_x)
            y: low-frequency data, shape (time_steps, dim_y)
            update_scaler: whether to fit the scalers on this data
            apply_scaler: whether to apply scaling to the data
            verbose: whether to print debug info
            
        Returns:
            inputs: [x_encoder_in, x_decoder_in, y_decoder_in]
            targets: [x_target, y_target]
        """
        # CRITICAL: Calculate minimum sample size (original TensorFlow approach)
        sample_size = min(
            self.sample_size_calc(x.shape[0], self.Lx, self.Tx, stride=self.freq_ratio),
            self.sample_size_calc(y.shape[0], self.Ly, self.Ty, stride=1)
        )
        
        if verbose:
            print(f'[mf_sample_generator]: sample size = {sample_size} {datetime.now()}')
        
        if update_scaler:
            self.scaler_x.fit(x)
            self.scaler_y.fit(y)
            if verbose:
                print(f' >>> scaler updated: scaler_x_n_feature = {self.scaler_x.n_features_in_}, scaler_y_n_feature = {self.scaler_y.n_features_in_}.')
        
        if apply_scaler:
            x, y = self.scaler_x.transform(x), self.scaler_y.transform(y)
            if verbose:
                print(f' >>> scaler applied.')
            
        # Convert to the format of input - target using the SAME sample_size
        x_input, x_target = self.supervised_sample_helper_with_size(x, self.Lx, self.Tx, self.freq_ratio, sample_size)
        y_input, y_target = self.supervised_sample_helper_with_size(y, self.Ly, self.Ty, 1, sample_size)
        
        # Further process so that the input are in a format compatible with decoder
        x_encoder_in = x_input
        x_decoder_in = np.concatenate([np.expand_dims(x_input[:, -1, :], axis=1), x_target[:, :-1, :]], axis=1)
        
        if self.zero_pad:
            y_decoder_in = np.pad(y_input, ((0, 0), (1, 0), (0, 0)), mode='constant')
        else:
            y_decoder_in = y_input
            
        if x_target.shape[1] == 1:
            x_target = np.squeeze(x_target, axis=1)
        
        # CRITICAL FIX: For MTMFSeq2Seq, Y decoder outputs single prediction, not sequence
        # Use only the LAST timestep of y_target as the target
        if y_target.shape[1] > 1:
            y_target = y_target[:, -1, :]  # Take last timestep: (batch_size, dim_y)
        # Keep y_target as (batch_size, dim_y) to match model output shape
        # Don't squeeze the last dimension
        
        return [x_encoder_in, x_decoder_in, y_decoder_in], [x_target, y_target]
    
    def create_one_forecast_sample(self, x, y, x_id, y_id, x_step, horizon=4, apply_scaler=False, verbose=False):
        """
        Create a single forecast sample for inference
        
        Args:
            x: high-frequency data
            y: low-frequency data  
            x_id: starting index for x
            y_id: starting index for y
            x_step: current step in x sequence
            horizon: forecast horizon
            apply_scaler: whether to apply scaling
            verbose: debug output
            
        Returns:
            inputs: [x_encoder_in, x_decoder_in, y_decoder_in]
            targets: [x_target, y_target]
        """
        assert x_id == self.freq_ratio * (y_id - 1)  # Assuming data alignment
        
        x_encoder_in = x[x_id:(x_id + self.Lx), :]
        x_decoder_in = x[(x_id + self.Lx - 1):(x_id + self.Lx + x_step), :]
        y_decoder_in = y[y_id:(y_id + self.Ly), :]
        
        x_target = x[(x_id + self.Lx):(x_id + self.Lx + horizon * self.freq_ratio), :]
        y_target = y[(y_id + self.Ly):(y_id + self.Ly + horizon), :]
        
        if apply_scaler:
            x_encoder_in = self.scaler_x.transform(x_encoder_in)
            x_decoder_in = self.scaler_x.transform(x_decoder_in)
            y_decoder_in = self.scaler_y.transform(y_decoder_in)
        
        # Zero pad y
        y_decoder_in = np.pad(y_decoder_in, ((1, 0), (0, 0)), mode='constant')
        
        # Expand so that there is also the batch dimension
        x_encoder_in = np.expand_dims(x_encoder_in, axis=0)
        x_decoder_in = np.expand_dims(x_decoder_in, axis=0)
        y_decoder_in = np.expand_dims(y_decoder_in, axis=0)
        
        if verbose:  # for debugging
            print(f'x_encoder_in indices: {list(range(x_id, x_id + self.Lx))}')
            print(f'x_decoder_in indices: {list(range(x_id + self.Lx - 1, x_id + self.Lx + x_step))}')
            print(f'y_decoder_in indices: {list(range(y_id, y_id + self.Ly))}')
            
            print(f'x_target indices: {list(range(x_id + self.Lx, x_id + self.Lx + horizon * self.freq_ratio))}')
            print(f'y_target indices: {list(range(y_id + self.Ly, y_id + self.Ly + horizon))}')
                        
        return [x_encoder_in, x_decoder_in, y_decoder_in], [x_target, y_target]

class MFDPOneStep(_baseMFDP):
    """Single-step data processor for seq2one models"""
    
    def __init__(self, Lx, Tx, Ly, Ty, freq_ratio, scaler_x=MinMaxScaler((-1,1)), 
                 scaler_y=MinMaxScaler((-1,1)), zero_pad=True):
        super().__init__(Lx, Tx, Ly, Ty, freq_ratio, scaler_x, scaler_y)
        self.zero_pad = zero_pad
        
    def mf_sample_generator(self, x, y, update_scaler=False, apply_scaler=False, verbose=False):
        """
        Mixed frequency sample generator for training/validation
        """
        # CRITICAL: Calculate minimum sample size (original TensorFlow approach)
        sample_size = min(
            self.sample_size_calc(x.shape[0], self.Lx, self.Tx, stride=self.freq_ratio),
            self.sample_size_calc(y.shape[0], self.Ly, self.Ty, stride=1)
        )
        
        if verbose:
            print(f'[mf_sample_generator]: sample size = {sample_size} {datetime.now()}')
            
        if update_scaler:
            self.scaler_x.fit(x)
            self.scaler_y.fit(y)
            if verbose:
                print(f' >>> scaler updated: scaler_x_n_feature = {self.scaler_x.n_features_in_}, scaler_y_n_feature = {self.scaler_y.n_features_in_}.')
                
        if apply_scaler:
            x, y = self.scaler_x.transform(x), self.scaler_y.transform(y)
            if verbose:
                print(f' >>> scaler applied.')
            
        # Use the same sample_size for both X and Y
        x_input, x_target = self.supervised_sample_helper_with_size(x, self.Lx, self.Tx, self.freq_ratio, sample_size)
        y_input, y_target = self.supervised_sample_helper_with_size(y, self.Ly, self.Ty, 1, sample_size)
        
        if self.zero_pad:  # Create dummy zeros for y as the first decoder step input
            y_input = np.pad(y_input, ((0, 0), (1, 0), (0, 0)), mode='constant')
        
        if x_target.shape[1] == 1:
            x_target = np.squeeze(x_target, axis=1)
        if y_target.shape[1] == 1:
            y_target = np.squeeze(y_target, axis=1)
        
        return (x_input, y_input), (x_target, y_target)

    def create_one_forecast_sample(self, x, y, x_id, y_id, x_step, horizon=4, apply_scaler=False, verbose=False):
        """Create a single forecast sample for inference"""
        
        assert x_id == self.freq_ratio * (y_id - 1)  # Assuming data alignment
        
        x_input = x[(x_id + x_step):(x_id + self.Lx + x_step), :]
        x_target = x[(x_id + self.Lx):(x_id + self.Lx + horizon * self.freq_ratio), :]
        
        y_input = y[y_id:(y_id + self.Ly), :]
        y_target = y[(y_id + self.Ly):(y_id + self.Ly + horizon), :]
        
        if apply_scaler:
            x_input = self.scaler_x.transform(x_input)
            y_input = self.scaler_y.transform(y_input)
        
        if self.zero_pad:
            y_input = np.pad(y_input, ((1, 0), (0, 0)), mode='constant')
        
        x_input = np.expand_dims(x_input, axis=0)
        y_input = np.expand_dims(y_input, axis=0)
        
        if verbose:  # for debugging
            print(f'x_input indices: {list(range(x_id + x_step, x_id + self.Lx + x_step))}')
            print(f'x_target indices: {list(range((x_id + self.Lx), (x_id + self.Lx + horizon * self.freq_ratio)))}')
            print(f'y_input indices: {list(range(y_id, (y_id + self.Ly)))}')
            print(f'y_target indices: {list(range((y_id + self.Ly), (y_id + self.Ly + horizon)))}')
        
        return [x_input, y_input], [x_target, y_target]