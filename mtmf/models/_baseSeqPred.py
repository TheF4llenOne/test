"""
Multi-task Mixed Frequency Model - PyTorch Version
(c) 2023 Jiahe Lin & George Michailidis
Converted to PyTorch
"""

import torch
import torch.nn as nn
import numpy as np

class _baseSeqPred:
    '''
    Base class for predictor that generate forecast/nowcasts, where the output is multi-step
    This is shared between seq2seq and transformer
    '''
    def __init__(self, model, scaler_x, scaler_y, apply_inv_scaler=True):
        
        self.model = model
        self.freq_ratio = self.model.freq_ratio
        self.Tx = self.model.Tx
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.apply_inv_scaler = apply_inv_scaler
        
    def _get_device(self):
        """Get the device of the model parameters"""
        return next(self.model.parameters()).device
        
    def _to_device(self, tensor):
        """Move tensor to model device"""
        return tensor.to(self._get_device())
    
    def forecast_hf_multi_step(self, inputs, num_of_steps):
        """
        Multi-step forecasting for high-frequency data
        
        Args:
            inputs: tuple of (x_encoder_in, x_decoder_in, y_decoder_in)
            num_of_steps: int, number of steps to forecast
            
        Returns:
            x_pred_vals: numpy array of shape (num_of_steps, dim_x)
        """
        x_encoder_in, x_decoder_in, y_decoder_in = inputs
        assert x_decoder_in.shape[1] - 1 + num_of_steps == self.freq_ratio
        
        # Convert to tensors if they're numpy arrays and move to device
        if isinstance(x_encoder_in, np.ndarray):
            x_encoder_in = self._to_device(torch.tensor(x_encoder_in, dtype=torch.float32))
        if isinstance(x_decoder_in, np.ndarray):
            x_decoder_in = self._to_device(torch.tensor(x_decoder_in, dtype=torch.float32))
        if isinstance(y_decoder_in, np.ndarray):
            y_decoder_in = self._to_device(torch.tensor(y_decoder_in, dtype=torch.float32))
        
        x_pred_vals = []
        
        with torch.no_grad():  # PyTorch equivalent of training=False
            for step in range(num_of_steps):
                # Forward pass through the model - FIX: Pass as list
                x_forecast_multi, _ = self.model([x_encoder_in, x_decoder_in, y_decoder_in])
                
                # Get the last timestep prediction
                x_forecast = x_forecast_multi[:, -1, :]  # Shape: (batch_size, dim_x)
                
                # Store prediction (convert to numpy and squeeze)
                x_pred_vals.append(x_forecast.squeeze().cpu().numpy())
                
                # Update decoder input by concatenating the new prediction
                x_decoder_in = torch.cat([x_decoder_in, x_forecast.unsqueeze(1)], dim=1)
        
        x_pred_vals = np.array(x_pred_vals)
        return x_pred_vals  # Shape: (num_of_steps, dim_x)
    
    def predict_system_one_cycle(self, inputs, x_step):
        """
        Predict one complete cycle (both HF and LF predictions)
        
        Args:
            inputs: tuple of (x_encoder_in, x_decoder_in, y_decoder_in)
            x_step: int, current step in the x sequence
            
        Returns:
            predictions: tuple of (x_pred, y_pred)
            updated_inputs: tuple of updated inputs for next cycle
        """
        x_encoder_in, x_decoder_in, y_decoder_in = inputs
        assert x_step == x_decoder_in.shape[1] - 1
        x_steps_to_forecast = self.freq_ratio - x_step
        
        # Convert inputs to tensors if needed and move to device
        if isinstance(x_encoder_in, np.ndarray):
            x_encoder_in = self._to_device(torch.tensor(x_encoder_in, dtype=torch.float32))
        if isinstance(x_decoder_in, np.ndarray):
            x_decoder_in = self._to_device(torch.tensor(x_decoder_in, dtype=torch.float32))
        if isinstance(y_decoder_in, np.ndarray):
            y_decoder_in = self._to_device(torch.tensor(y_decoder_in, dtype=torch.float32))
        
        if x_steps_to_forecast < 1:  # Directly make a nowcast
            x_decoder_in_aug = x_decoder_in
        else:
            # Convert back to numpy for forecast_hf_multi_step
            numpy_inputs = (
                x_encoder_in.cpu().numpy(),
                x_decoder_in.cpu().numpy(),
                y_decoder_in.cpu().numpy()
            )
            x_pred_vals = self.forecast_hf_multi_step(numpy_inputs, num_of_steps=x_steps_to_forecast)
            
            # Convert back to tensor and concatenate
            x_pred_vals_tensor = self._to_device(torch.tensor(x_pred_vals, dtype=torch.float32)).unsqueeze(0)  # Add batch dim
            x_decoder_in_aug = torch.cat([x_decoder_in, x_pred_vals_tensor], dim=1)
        
        # Collect x prediction
        x_pred = x_decoder_in_aug[0, -self.freq_ratio:, :].cpu().numpy()
        
        # Make y prediction - FIX: Pass as list
        with torch.no_grad():
            _, y_pred = self.model([x_encoder_in, x_decoder_in_aug, y_decoder_in])
        
        # Convert back to numpy
        y_pred = y_pred.cpu().numpy()
        
        # Update encoder and decoder inputs
        x_encoder_in = torch.cat([x_encoder_in[:, self.freq_ratio:, :], 
                                x_decoder_in_aug[:, -self.freq_ratio:, :]], dim=1)
        x_decoder_in = x_decoder_in_aug[:, -1:, :]  # Take last step as new decoder input
        
        # Update y decoder
        # y_pred is (batch_size, dim_y), we need to add sequence dimension to make it (batch_size, 1, dim_y)
        y_pred_expanded = self._to_device(torch.tensor(y_pred, dtype=torch.float32)).unsqueeze(1)
        y_decoder_in = torch.cat([y_decoder_in[:, 1:, :], y_pred_expanded], dim=1)
        
        # Convert back to numpy for return
        updated_inputs = (
            x_encoder_in.cpu().numpy(),
            x_decoder_in.cpu().numpy(),
            y_decoder_in.cpu().numpy()
        )
        
        return (x_pred, y_pred[0]), updated_inputs
        
    def __call__(self, inputs, x_step, horizon=4):
        """
        Main prediction method that handles multi-horizon forecasting
        
        Args:
            inputs: tuple of (x_encoder_in, x_decoder_in, y_decoder_in)
            x_step: int, current step in the x sequence
            horizon: int, number of forecast horizons
            
        Returns:
            x_pred: numpy array of shape (horizon * freq_ratio, dim_x)
            y_pred: numpy array of shape (horizon, dim_y)
        """
        assert x_step == inputs[1].shape[1] - 1
        assert x_step <= self.freq_ratio
        
        x_pred, y_pred = [], []
        
        for hz in range(horizon):
            predictions, inputs = self.predict_system_one_cycle(inputs, x_step=x_step)
            x_pred.append(predictions[0])
            y_pred.append(predictions[1])
            x_step = 0  # Reset x_step for subsequent cycles
            
        # Concatenate predictions
        x_pred = np.concatenate(x_pred, axis=0)  # Shape: (horizon * freq_ratio, dim_x)
        y_pred = np.stack(y_pred, axis=0)        # Shape: (horizon, dim_y)
        
        # Apply inverse scaling if needed
        if self.apply_inv_scaler:
            x_pred = self.scaler_x.inverse_transform(x_pred)
            y_pred = self.scaler_y.inverse_transform(y_pred)
            
        return x_pred, y_pred