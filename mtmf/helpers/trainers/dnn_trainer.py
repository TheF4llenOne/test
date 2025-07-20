"""
Trainer class for running DNN models on real data - PyTorch Version
(c) 2023 Jiahe Lin & George Michailidis
Converted to PyTorch
"""

import sys
import os
import shutil
import pickle
import math

import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helpers import ClsConstructor
from .utils_train import plot_loss_over_epoch, plot_fitted_val

class EarlyStopping:
    """Early stopping callback for PyTorch training"""
    def __init__(self, patience=10, monitor='val_loss', min_delta=0, mode='min'):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
class ReduceLROnPlateau:
    """Learning rate reduction callback for PyTorch"""
    def __init__(self, optimizer, monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, min_delta=0, mode='min'):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr < current_lr:
                self.optimizer.param_groups[0]['lr'] = new_lr
                print(f"Reducing learning rate to {new_lr}")
            self.counter = 0

class LossPrintCallback:
    """Loss printing callback for PyTorch training"""
    def __init__(self, every_n_epochs=100):
        self.every_n_epochs = every_n_epochs
        
    def __call__(self, epoch, logs):
        if logs.get('output_1_loss') is not None and logs.get('output_2_loss') is not None:
            if (epoch+1) % self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}; output_1_loss = {logs.get('output_1_loss'):.4f}, output_2_loss = {logs.get('output_2_loss'):.4f}.")
        else:
            if (epoch+1) % self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}.")

class nnTrainer:
    """Neural Network Trainer for PyTorch models"""
    
    def __init__(self, args, criterion=nn.MSELoss(), seed=411):
        """
        Initialize the trainer
        
        Args:
            args: configuration arguments
            criterion: loss function (default: MSELoss)
            seed: random seed
        """
        self.args = args
        self.criterion = criterion
        self.cls_constructor = ClsConstructor(self.args)
        self.seed = seed
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def set_seed(self, repickle_args=True):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        
        # Set deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        setattr(self.args, 'seed', self.seed)
        if repickle_args:
            with open(f"{self.args.output_folder}/args.pkl", "wb") as handle:
                pickle.dump(self.args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def source_data(self):
        """Load and prepare the data"""
        x = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx", 
                         index_col='timestamp', sheet_name='x')
        y = pd.read_excel(f"{self.args.data_folder}/{self.args.ds_name}.xlsx", 
                         index_col='timestamp', sheet_name='y')
        
        x.index, y.index = pd.to_datetime(x.index), pd.to_datetime(y.index)
        xdata, ydata = x.values, y.values
        
        self.df_info = {
            'x_index': list(x.index),
            'y_index': list(y.index),
            'x_columns': list(x.columns),
            'y_columns': list(y.columns),
            'x_total_obs': x.shape[0],
            'y_total_obs': y.shape[0]
        }
        self.raw_data = (xdata, ydata)
        
    def generate_train_val_datasets(self, x_train_end, y_train_end, n_val=None):
        """
        Generate training and validation datasets
        
        Args:
            x_train_end: ending index for x training data
            y_train_end: ending index for y training data  
            n_val: number of validation samples
            
        Returns:
            dp: data processor
            train_data: (train_inputs, train_targets)
            val_data: (val_inputs, val_targets) or None
        """
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        
        # Get training data
        x_train, y_train = xdata[:x_train_end, :], ydata[:y_train_end, :]
        train_inputs, train_targets = dp.mf_sample_generator(
            x_train, y_train,
            update_scaler=True if args.scale_data else False,
            apply_scaler=True if args.scale_data else False,
            verbose=True if args.verbose > 0 else False
        )
        
        print(f'[{len(train_inputs[0])} train samples generated] inputs dims: {[temp.shape for temp in train_inputs]}; target dims: {[temp.shape for temp in train_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Generate validation data if requested
        val_inputs, val_targets = None, None
        if n_val is not None:
            assert n_val < len(train_inputs[0])
            
            # Split training data for validation
            val_inputs = [inp[-n_val:] for inp in train_inputs]
            val_targets = [tgt[-n_val:] for tgt in train_targets]
            train_inputs = [inp[:-n_val] for inp in train_inputs]
            train_targets = [tgt[:-n_val] for tgt in train_targets]
            
            print(f'[{n_val} val samples reserved] inputs dims: {[temp.shape for temp in val_inputs]}; target dims: {[temp.shape for temp in val_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        train_data = (train_inputs, train_targets)
        val_data = (val_inputs, val_targets) if n_val is not None else None
        
        return dp, train_data, val_data
    
    def _create_data_loader(self, inputs, targets, batch_size, shuffle=True):
        """Convert numpy arrays to PyTorch DataLoader"""
        # Convert inputs and targets to tensors - now all arrays should have same batch size
        input_tensors = [torch.FloatTensor(inp) for inp in inputs]
        target_tensors = [torch.FloatTensor(tgt) for tgt in targets]
        
        # Create dataset
        dataset = TensorDataset(*input_tensors, *target_tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _compute_rmse(self, pred, target):
        """Compute RMSE metric"""
        return torch.sqrt(torch.mean((pred - target) ** 2))
            
    def config_and_train_model(self, train_data, val_data=None):
        """
        Configure and train the model
        
        Args:
            train_data: (train_inputs, train_targets)
            val_data: (val_inputs, val_targets) or None
            
        Returns:
            trained model
        """
        args = self.args
        model = self.cls_constructor.create_model()
        model = model.to(self.device)
        
        train_inputs, train_targets = train_data
        
        # Save model summary
        with open(f'{args.output_folder}/model_summary.txt', 'w') as f:
            f.write(str(model))
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Set up callbacks
        callbacks = {}
        
        if hasattr(args, 'reduce_LR_monitor') and args.reduce_LR_monitor and len(args.reduce_LR_monitor) > 0:
            reduce_lr = ReduceLROnPlateau(
                optimizer, 
                monitor=args.reduce_LR_monitor,
                factor=args.reduce_LR_factor,
                patience=args.reduce_LR_patience,
                min_lr=0.000001,
                min_delta=0,
                mode='min'
            )
            callbacks['reduce_lr'] = reduce_lr
            
        if hasattr(args, 'ES_patience') and args.ES_patience is not None:
            early_stopping = EarlyStopping(
                patience=args.ES_patience,
                monitor='val_loss',
                min_delta=0,
                mode='min'
            )
            callbacks['early_stopping'] = early_stopping
            
        if args.verbose > 0:
            loss_printer = LossPrintCallback(args.verbose)
            callbacks['loss_printer'] = loss_printer
        
        # Create data loaders
        shuffle_train = getattr(args, 'shuffle', True)  # Default to True if not specified
        train_loader = self._create_data_loader(train_inputs, train_targets, args.batch_size, shuffle_train)
        if val_data is not None:
            val_inputs, val_targets = val_data
            val_loader = self._create_data_loader(val_inputs, val_targets, args.batch_size, False)
        else:
            val_loader = None
        
        print(f'[{args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Manual training loop
        history = {
            'loss': [], 'output_1_loss': [], 'output_2_loss': [], 
            'val_loss': [], 'val_output_1_loss': [], 'val_output_2_loss': []
        }
        
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_output_1_loss = 0.0
            train_output_2_loss = 0.0
            
            for batch_data in train_loader:
                # Split batch into inputs and targets
                num_inputs = len(train_inputs)
                batch_inputs = batch_data[:num_inputs]
                batch_targets = batch_data[num_inputs:]
                
                # Move to device
                batch_inputs = [inp.to(self.device) for inp in batch_inputs]
                batch_targets = [tgt.to(self.device) for tgt in batch_targets]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_inputs)
                
                # Compute losses
                loss_x = self.criterion(outputs[0], batch_targets[0])
                loss_y = self.criterion(outputs[1], batch_targets[1])
                total_loss = loss_x + loss_y
                
                # Add L1/L2 regularization if specified
                if hasattr(args, 'l1reg') and args.l1reg > 0:
                    l1_penalty = sum(p.abs().sum() for p in model.parameters())
                    total_loss += args.l1reg * l1_penalty
                    
                if hasattr(args, 'l2reg') and args.l2reg > 0:
                    l2_penalty = sum((p**2).sum() for p in model.parameters())
                    total_loss += args.l2reg * l2_penalty
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_output_1_loss += loss_x.item()
                train_output_2_loss += loss_y.item()
            
            # Average losses
            train_loss /= len(train_loader)
            train_output_1_loss /= len(train_loader)
            train_output_2_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            val_output_1_loss = 0.0
            val_output_2_loss = 0.0
            
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch_data in val_loader:
                        num_inputs = len(train_inputs)
                        batch_inputs = batch_data[:num_inputs]
                        batch_targets = batch_data[num_inputs:]
                        
                        batch_inputs = [inp.to(self.device) for inp in batch_inputs]
                        batch_targets = [tgt.to(self.device) for tgt in batch_targets]
                        
                        outputs = model(batch_inputs)
                        
                        loss_x = self.criterion(outputs[0], batch_targets[0])
                        loss_y = self.criterion(outputs[1], batch_targets[1])
                        total_loss = loss_x + loss_y
                        
                        val_loss += total_loss.item()
                        val_output_1_loss += loss_x.item()
                        val_output_2_loss += loss_y.item()
                
                val_loss /= len(val_loader)
                val_output_1_loss /= len(val_loader)
                val_output_2_loss /= len(val_loader)
            
            # Store history
            history['loss'].append(train_loss)
            history['output_1_loss'].append(train_output_1_loss)
            history['output_2_loss'].append(train_output_2_loss)
            if val_loader is not None:
                history['val_loss'].append(val_loss)
                history['val_output_1_loss'].append(val_output_1_loss)
                history['val_output_2_loss'].append(val_output_2_loss)
            
            # Execute callbacks
            logs = {
                'loss': train_loss,
                'output_1_loss': train_output_1_loss,
                'output_2_loss': train_output_2_loss
            }
            if val_loader is not None:
                logs.update({
                    'val_loss': val_loss,
                    'val_output_1_loss': val_output_1_loss,
                    'val_output_2_loss': val_output_2_loss
                })
            
            # Apply callbacks
            if 'loss_printer' in callbacks:
                callbacks['loss_printer'](epoch, logs)
                
            if 'reduce_lr' in callbacks:
                callbacks['reduce_lr'](val_loss if val_loader is not None else train_loss)
                
            if 'early_stopping' in callbacks:
                callbacks['early_stopping'](val_loss if val_loader is not None else train_loss)
                if callbacks['early_stopping'].early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print(f'[{args.model_type} model training ends] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Save model
        torch.save(model.state_dict(), f'{args.output_folder}/model_weights.pth')
        torch.save(model, f'{args.output_folder}/model_complete.pth')
        
        # Plot loss
        plot_loss_over_epoch(history, args, save_as_file=f'{args.output_folder}/loss_over_epoch.png')
        return model
            
    def eval_train(self, model, dp, train_data):
        """Evaluate the model on training data"""
        args = self.args
        train_inputs, train_targets = train_data
        
        # PyTorch inference
        model.eval()
        with torch.no_grad():
            # Convert to tensors and move to device
            input_tensors = [torch.FloatTensor(inp).to(self.device) for inp in train_inputs]
            
            # Get predictions
            outputs = model(input_tensors)
            x_is_pred = outputs[0].cpu().numpy()
            y_is_pred = outputs[1].cpu().numpy()
        
        if args.scale_data:
            x_is_pred = dp.apply_scaler('scaler_x', x_is_pred, inverse=True)
            y_is_pred = dp.apply_scaler('scaler_y', y_is_pred, inverse=True)
            x_is_truth = dp.apply_scaler('scaler_x', train_targets[0], inverse=True)
            y_is_truth = dp.apply_scaler('scaler_y', train_targets[1], inverse=True)
        else:
            x_is_truth, y_is_truth = train_targets[0], train_targets[1]
            
        plot_fitted_val(
            args, (x_is_pred, y_is_pred), (x_is_truth, y_is_truth), 
            x_col=self.df_info['x_columns'].index(args.X_COLNAME), 
            y_col=self.df_info['y_columns'].index(args.Y_COLNAME), 
            time_steps=None, 
            save_as_file=f'{args.output_folder}/train_fit_static.png'
        )
    
    def config_predictor(self, model, dp):
        """Configure the predictor for inference"""
        args = self.args
        predictor = self.cls_constructor.create_predictor(model, dp, apply_inv_scaler=args.scale_data)
        return predictor
    
    def add_prediction_to_collector(self, predictions_by_vintage, targets_by_vintage, T_datestamp, x_PRED_collector=[], y_PRED_collector=[]):
        """
        Add predictions AND ground truth to collectors in original format
        
        Args:
            predictions_by_vintage: Dict with format {vintage: (x_pred, y_pred)}
            targets_by_vintage: Dict with format {vintage: (x_target, y_target)}
            T_datestamp: Current timestamp 
            x_PRED_collector: List to collect X predictions
            y_PRED_collector: List to collect Y predictions
        """
        args = self.args
        
        # Create step column names for predictions and targets
        x_pred_col_keys = [f'pred_step_{i+1}' for i in range(args.freq_ratio * args.horizon)]  # 24 steps for X
        x_true_col_keys = [f'true_step_{i+1}' for i in range(args.freq_ratio * args.horizon)]  # 24 steps for X
        y_pred_col_keys = [f'pred_step_{i+1}' for i in range(args.horizon)]  # 4 steps for Y
        y_true_col_keys = [f'true_step_{i+1}' for i in range(args.horizon)]  # 4 steps for Y
        
        # Extract predictions and targets for each vintage
        for vintage in predictions_by_vintage.keys():
            x_pred, y_pred = predictions_by_vintage[vintage]
            x_target, y_target = targets_by_vintage[vintage]
            
            # Process X predictions (high-frequency)
            for col_id, variable_name in enumerate(self.df_info['x_columns']):
                temp = {
                    'prev_timestamp': T_datestamp,  # Previous reference timestamp
                    'tag': vintage,                 # F, N1, N2, N3, N4, N5, N6
                    'variable_name': variable_name, # Name of the X variable
                    'frequency': 'high'             # Mark as high-frequency
                }
                # Add individual step predictions and targets
                temp.update(dict(zip(x_pred_col_keys, list(x_pred[:, col_id]))))
                temp.update(dict(zip(x_true_col_keys, list(x_target[:, col_id]))))
                x_PRED_collector.append(temp)
            
            # Process Y predictions (low-frequency)
            for col_id, variable_name in enumerate(self.df_info['y_columns']):
                temp = {
                    'prev_timestamp': T_datestamp,  # Previous reference timestamp
                    'tag': vintage,                 # F, N1, N2, N3, N4, N5, N6
                    'variable_name': variable_name, # Name of the Y variable
                    'frequency': 'low'              # Mark as low-frequency
                }
                # Add individual step predictions and targets - handle both 1D and 2D arrays
                if len(y_pred.shape) > 1:
                    temp.update(dict(zip(y_pred_col_keys, list(y_pred[:, col_id]))))
                    temp.update(dict(zip(y_true_col_keys, list(y_target[:, col_id]))))
                else:
                    # For 1D arrays
                    y_pred_vals = [y_pred[col_id]] * args.horizon if y_pred.ndim == 1 else list(y_pred[:, col_id])
                    y_true_vals = [y_target[col_id]] * args.horizon if y_target.ndim == 1 else list(y_target[:, col_id])
                    temp.update(dict(zip(y_pred_col_keys, y_pred_vals)))
                    temp.update(dict(zip(y_true_col_keys, y_true_vals)))
                y_PRED_collector.append(temp)
    
    def run_forecast_one_set(self, predictor, dp, y_start_id, x_start_id):
        """
        Run one set of forecasts: F, N1, N2, N3, N4, N5, N6 (updated format)
        
        Args:
            predictor: configured predictor
            dp: data processor
            y_start_id: starting y index
            x_start_id: starting x index
            
        Returns:
            predictions_by_vintage: Dict with format {vintage: (x_pred, y_pred)}
            targets_by_vintage: Dict with format {vintage: (x_target, y_target)}
        """
        args = self.args
        xdata, ydata = self.raw_data
        
        assert x_start_id == args.freq_ratio * (y_start_id - 1)
        predictions_by_vintage = {}
        targets_by_vintage = {}
        
        for x_step in range(args.freq_ratio + 1):  # 0 to 6 = F, N1, N2, N3, N4, N5, N6
            experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
            
            # Get forecast inputs and targets
            inputs, targets = dp.create_one_forecast_sample(
                xdata, ydata, x_start_id, y_start_id, x_step,
                horizon=getattr(args, 'horizon', 4),
                apply_scaler=True if args.scale_data else False,
                verbose=False
            )
            
            # Get predictions
            x_pred, y_pred = predictor(inputs, x_step=x_step, horizon=getattr(args, 'horizon', 4))
            
            # Store predictions and targets as tuples
            predictions_by_vintage[experiment_tag] = (x_pred, y_pred)
            targets_by_vintage[experiment_tag] = targets  # (x_target, y_target)
            
        return predictions_by_vintage, targets_by_vintage
    
    def run_forecast_comprehensive(self):
        """
        Comprehensive rolling forecast implementation (adapted from original)
        """
        args = self.args
        
        print(f'[Starting comprehensive forecasting for {args.model_type}] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Initialize collectors
        x_PRED_collector, y_PRED_collector = [], []
        
        # Calculate training end points (80% of data for training)
        x_total = self.df_info['x_total_obs']
        y_total = self.df_info['y_total_obs']
        x_train_end = int(x_total * 0.8)
        y_train_end = int(y_total * 0.8)
        
        # Handle validation samples
        n_val = None
        if hasattr(args, 'n_val') and args.n_val is not None and args.n_val > 0:
            if args.n_val < 1.0:
                n_val = int(min(x_train_end, y_train_end) * args.n_val)
            else:
                n_val = int(args.n_val)
        
        # Train model once (static mode)
        print(f'[Setting up and training model]')
        dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val=n_val)
        model = self.config_and_train_model(train_data, val_data)
        self.eval_train(model, dp, train_data)
        predictor = self.config_predictor(model, dp)
        
        # Calculate rolling forecast parameters
        y_start_id = y_train_end - args.Ty + 1  # Start after training data
        x_start_id = args.freq_ratio * (y_start_id - 1)
        
        # Calculate how many rolling forecasts we can do
        test_size = min(
            y_total - (y_start_id + args.horizon),  # Don't go beyond available Y data
            (x_total - (x_start_id + args.freq_ratio * args.horizon)) // args.freq_ratio  # Don't go beyond available X data
        )
        
        print(f'[Rolling forecasts] Starting from y_id={y_start_id}, x_id={x_start_id}, test_size={test_size}')
        
        # Run rolling forecasts
        print(f'[forecast experiments start] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        for experiment_id in range(test_size):
            
            # Get current timestamp for this experiment
            T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
            x_range = [self.df_info['x_index'][x_start_id], self.df_info['x_index'][x_start_id + args.Lx - 1]]
            y_range = [self.df_info['y_index'][y_start_id], self.df_info['y_index'][y_start_id + args.Ty - 2]]
            
            print(f" >> experiment {experiment_id+1}/{test_size}: ref_timestamp = {T_datestamp}")
            print(f"    x_input_range = [{x_range[0]} to {x_range[1]}]")
            print(f"    y_input_range = [{y_range[0]} to {y_range[1]}]")
            
            # Run forecast for all vintages (F, N1-N6)
            try:
                predictions_by_vintage, targets_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                
                # Add to collectors (now with targets)
                self.add_prediction_to_collector(predictions_by_vintage, targets_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
                
                print(f"    ✓ Generated predictions for vintages: {list(predictions_by_vintage.keys())}")
                
            except Exception as e:
                print(f"    ✗ Error in experiment {experiment_id+1}: {e}")
                break
            
            # Move to next time period
            x_start_id += args.freq_ratio
            y_start_id += 1
        
        print(f'[forecast experiments end] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Convert to DataFrames
        x_PRED_df = pd.DataFrame(x_PRED_collector)
        y_PRED_df = pd.DataFrame(y_PRED_collector)
        
        print(f'[Results Summary]')
        print(f'  X predictions: {len(x_PRED_df)} records')
        print(f'  Y predictions: {len(y_PRED_df)} records')
        print(f'  Experiments completed: {experiment_id+1}')
        print(f'  Vintages per experiment: {len(predictions_by_vintage)}')
        
        # Save comprehensive results
        output_file = f"{args.output_folder}/comprehensive_forecasts.xlsx"
        
        with pd.ExcelWriter(output_file) as writer:
            x_PRED_df.to_excel(writer, sheet_name='x_prediction', index=False)
            y_PRED_df.to_excel(writer, sheet_name='y_prediction', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Experiments', 'X Records', 'Y Records', 'Vintages per Experiment', 'X Variables', 'Y Variables'],
                'Value': [experiment_id+1, len(x_PRED_df), len(y_PRED_df), len(predictions_by_vintage), 
                         len(self.df_info['x_columns']), len(self.df_info['y_columns'])]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='summary', index=False)
        
        print(f'[Comprehensive forecasts saved to: {output_file}]')
        
        # Also save the simple CSV for compatibility
        simple_file = f"{args.output_folder}/forecast_predictions.csv"
        self._save_predictions_simple(predictions_by_vintage, simple_file)
        
        # NEW: Add evaluation step (missing from original PyTorch conversion)
        from .utils_train import evaluate_forecast_results
        
        print(f'[Starting evaluation] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        evaluate_forecast_results(x_PRED_df, y_PRED_df, args.output_folder)
        
        return x_PRED_df, y_PRED_df
    
    def run_forecast(self):
        """
        Main forecast method - now calls comprehensive version
        """
        return self.run_forecast_comprehensive()
    
    def _save_predictions_simple(self, predictions_by_vintage, filename):
        """Simple CSV export for predictions (compatibility)"""
        results = {}
        
        for vintage, (x_pred, y_pred) in predictions_by_vintage.items():
            # Save y predictions (main target)
            if len(y_pred.shape) > 1:
                for i in range(y_pred.shape[1]):
                    results[f'{vintage}_y_pred_dim{i}'] = y_pred[:, i]
            else:
                results[f'{vintage}_y_pred'] = y_pred.flatten()
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Simple predictions saved to {filename}")