"""
Trainer class for running DNN models on real data - PyTorch Version
FIXED VERSION with all corrections applied
"""

import sys
import os
import shutil
import pickle

import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helpers import ClsConstructor
from .utils_train import *

class LossPrintCallback:
    def __init__(self, every_n_epochs=100):
        self.every_n_epochs = every_n_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('output_1_loss') is not None and logs.get('output_2_loss') is not None:
            if (epoch+1) % self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}; output_1_loss = {logs.get('output_1_loss'):.4f}, output_2_loss = {logs.get('output_2_loss'):.4f}.")
        else:
            if (epoch+1) % self.every_n_epochs == 0:
                print(f"  >> epoch = {epoch+1}; loss = {logs.get('loss'):.4f}.")

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, logs):
        val_loss = logs.get(self.monitor)
        if val_loss is None:
            return
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class ReduceLROnPlateau:
    def __init__(self, optimizer, monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001, min_delta=0, mode='min'):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        
    def step(self, metrics):
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return
            
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == 'min':
                improved = current_score < self.best_score - self.min_delta
            else:
                improved = current_score > self.best_score + self.min_delta
                
            if improved:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self._reduce_lr()
                    self.counter = 0
    
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr

class nnTrainer:

    def __init__(self, args, criterion=nn.MSELoss(), seed=411):
        
        self.args = args
        self.criterion = criterion
        self.cls_constructor = ClsConstructor(self.args)
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_seed(self, repickle_args=True):
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        
        setattr(self.args, 'seed', self.seed)
        with open(f"{self.args.output_folder}/args.pickle", "wb") as handle:
            pickle.dump(self.args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def source_data(self):
        """Handles hour-based medical data properly"""
        # Read parquet files for medical data using config settings
        vitals_file = getattr(self.args, 'vitals_file', 'vitals_forward_filled_cleaned.parquet')
        labs_file = getattr(self.args, 'labs_file', 'labs_filtered_cleaned.parquet')
        
        x = pd.read_parquet(f"{self.args.data_folder}/{vitals_file}")
        y = pd.read_parquet(f"{self.args.data_folder}/{labs_file}")
        
        # Get time column names from config
        vitals_time_col = getattr(self.args, 'vitals_time_col', 'hour_bin')
        labs_time_col = getattr(self.args, 'labs_time_col', 'bin_start_hour')
        
        # Set timestamp columns as index
        x = x.set_index(vitals_time_col)
        y = y.set_index(labs_time_col)
        
        # Convert hour indices to datetime for framework compatibility
        base_date = pd.Timestamp('2020-01-01')
        x.index = base_date + pd.to_timedelta(x.index, unit='h')
        y.index = base_date + pd.to_timedelta(y.index, unit='h')
        
        # FIX: Add the same rounding as configurator to ensure exact matches
        x.index = x.index.round('H')
        y.index = y.index.round('H')
        
        # Remove patient ID columns if present (not features for modeling)
        if 'unique_patient_id' in x.columns:
            x = x.drop('unique_patient_id', axis=1)
        if 'unique_patient_id' in y.columns:
            y = y.drop('unique_patient_id', axis=1)
        
        xdata, ydata = x.values, y.values
        
        self.df_info = {'x_index': list(x.index),
                        'y_index': list(y.index),
                        'x_columns': list(x.columns),
                        'y_columns': list(y.columns),
                        'x_total_obs': x.shape[0],
                        'y_total_obs': y.shape[0]
                    }
        self.raw_data = (xdata, ydata)
        
        print(f"Data loaded successfully:")
        print(f"  Vitals (X): {xdata.shape} - {list(x.columns)}")
        print(f"  Labs (Y): {ydata.shape} - {list(y.columns)}")
        print(f"  Time range: {x.index.min()} to {x.index.max()}")
        
        # Debug: Print some timestamps to verify rounding
        print(f"  First few vitals timestamps: {x.index[:3].tolist()}")
        print(f"  First few labs timestamps: {y.index[:3].tolist()}")
        
    def generate_train_val_datasets(self, x_train_end, y_train_end, n_val=None):
        """
        helper function for generating train/val dataset; the reason for not adding them as attributes is out of
        consideration for dynamic run
        Argv:
        - x_train_end, y_train_end: the ending index, resp for x and y
        """
        args = self.args
        dp = self.cls_constructor.create_data_processor()
        
        xdata, ydata = self.raw_data
        
        ## get training data
        x_train, y_train = xdata[:x_train_end,:], ydata[:y_train_end,:]
        train_inputs, train_targets = dp.mf_sample_generator(x_train,
                                                            y_train,
                                                            update_scaler=True if args.scale_data else False,
                                                            apply_scaler=True if args.scale_data else False)
        
        print(f'[train samples generated] inputs dims: {[temp.shape for temp in train_inputs]}; target dims: {[temp.shape for temp in train_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # FIX: Synchronize sample counts for mixed-frequency data
        # Find the minimum number of samples across all inputs/targets
        all_sample_counts = [inp.shape[0] for inp in train_inputs] + [tgt.shape[0] for tgt in train_targets]
        min_samples = min(all_sample_counts)
        
        print(f'🔧 SYNC FIX: Sample counts before sync: {all_sample_counts}')
        print(f'🔧 SYNC FIX: Using minimum sample count: {min_samples}')
        
        # Truncate all arrays to the minimum sample count
        train_inputs = [inp[:min_samples] for inp in train_inputs]
        train_targets = [tgt[:min_samples] for tgt in train_targets]
        
        print(f'🔧 SYNC FIX: Sample counts after sync: {[inp.shape[0] for inp in train_inputs + train_targets]}')
        
        if n_val is not None:
            if isinstance(n_val, float):
                n_val = round(min_samples * n_val)  # Use min_samples instead of train_inputs[0].shape[0]
            
            train_inputs, val_inputs = [x[:-n_val] for x in train_inputs], [x[-n_val:] for x in train_inputs]
            train_targets, val_targets = [x[:-n_val] for x in train_targets], [x[-n_val:] for x in train_targets]
            print(f'[{n_val} val samples reserved] inputs dims: {[temp.shape for temp in val_inputs]}; target dims: {[temp.shape for temp in val_targets]} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        train_data = (train_inputs, train_targets)
        val_data = (val_inputs, val_targets) if n_val is not None else None
        
        return dp, train_data, val_data
    
    def _create_data_loader(self, inputs, targets, batch_size, shuffle=True):
        """Convert numpy arrays to PyTorch DataLoader"""
        # Convert inputs and targets to tensors
        input_tensors = [torch.FloatTensor(inp) for inp in inputs]
        target_tensors = [torch.FloatTensor(tgt) for tgt in targets]
        
        # Create dataset
        dataset = TensorDataset(*input_tensors, *target_tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _compute_rmse(self, pred, target):
        """Compute RMSE metric"""
        return torch.sqrt(torch.mean((pred - target) ** 2))
            
    def config_and_train_model(self, train_data, val_data=None):
    
        args = self.args
        model = self.cls_constructor.create_model()
        model = model.to(self.device)
        
        train_inputs, train_targets = train_data
        
        # Save model summary (PyTorch equivalent)
        with open(f'{args.output_folder}/model_summary.txt', 'w') as f:
            f.write(str(model))
        
        # Set up optimizer (PyTorch manual setup)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # FIXED: Set up callbacks (PyTorch custom classes)
        callbacks = {}
        
        if hasattr(args, 'reduce_LR_monitor') and args.reduce_LR_monitor:
            reduce_lr = ReduceLROnPlateau(optimizer, 
                                        monitor=args.reduce_LR_monitor,
                                        factor=args.reduce_LR_factor,
                                        patience=args.reduce_LR_patience,
                                        min_lr=0.000001,
                                        min_delta=0,
                                        mode='min')
            callbacks['reduce_lr'] = reduce_lr
            
        if args.ES_patience is not None:
            early_stopping = EarlyStopping(patience=args.ES_patience,
                                        monitor='val_loss',
                                        min_delta=0,
                                        mode='min')
            callbacks['early_stopping'] = early_stopping
            
        if args.verbose > 0:
            loss_printer = LossPrintCallback(args.verbose)
            callbacks['loss_printer'] = loss_printer
        
        # Create data loaders (PyTorch requirement)
        train_loader = self._create_data_loader(train_inputs, train_targets, args.batch_size, args.shuffle)
        if val_data is not None:
            val_inputs, val_targets = val_data
            val_loader = self._create_data_loader(val_inputs, val_targets, args.batch_size, False)
        else:
            val_loader = None
        
        print(f'[{args.model_type} model training starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Manual training loop (PyTorch requirement)
        history = {'loss': [], 'output_1_loss': [], 'output_2_loss': [], 
                  'val_loss': [], 'val_output_1_loss': [], 'val_output_2_loss': []}  # Initialize all keys
        
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
                
                # FIXED: Forward pass
                outputs = model(batch_inputs, training=True)  # Added training=True
                
                # Compute losses
                loss_x = self.criterion(outputs[0], batch_targets[0])
                loss_y = self.criterion(outputs[1], batch_targets[1])
                total_loss = loss_x + loss_y
                
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
            
            # FIXED: Validation phase with proper loss tracking
            val_loss = 0.0
            val_output_1_loss = 0.0
            val_output_2_loss = 0.0
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch_data in val_loader:
                        num_inputs = len(val_inputs)
                        batch_inputs = batch_data[:num_inputs]
                        batch_targets = batch_data[num_inputs:]
                        
                        batch_inputs = [inp.to(self.device) for inp in batch_inputs]
                        batch_targets = [tgt.to(self.device) for tgt in batch_targets]
                        
                        outputs = model(batch_inputs, training=False)  # FIXED: Added training=False
                        loss_x = self.criterion(outputs[0], batch_targets[0])
                        loss_y = self.criterion(outputs[1], batch_targets[1])
                        total_loss = loss_x + loss_y
                        
                        val_loss += total_loss.item()
                        val_output_1_loss += loss_x.item()
                        val_output_2_loss += loss_y.item()
                
                val_loss /= len(val_loader)
                val_output_1_loss /= len(val_loader)
                val_output_2_loss /= len(val_loader)
            
            # FIXED: Store history with validation losses
            history['loss'].append(train_loss)
            history['output_1_loss'].append(train_output_1_loss)
            history['output_2_loss'].append(train_output_2_loss)
            if val_loader is not None:
                history['val_loss'].append(val_loss)
                history['val_output_1_loss'].append(val_output_1_loss)
                history['val_output_2_loss'].append(val_output_2_loss)
            
            # Apply callbacks
            logs = {'loss': train_loss, 'output_1_loss': train_output_1_loss, 'output_2_loss': train_output_2_loss}
            if val_loader is not None:
                logs['val_loss'] = val_loss
            
            if 'loss_printer' in callbacks:
                callbacks['loss_printer'].on_epoch_end(epoch, logs)
            
            if 'reduce_lr' in callbacks:
                callbacks['reduce_lr'].step(logs)
            
            if 'early_stopping' in callbacks:
                callbacks['early_stopping'](logs)
                if callbacks['early_stopping'].early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                            
        print(f'[{args.model_type} model training ends] epoch = {len(history["loss"])} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Plot loss (same as TensorFlow)
        plot_loss_over_epoch(history, args, save_as_file=f'{args.output_folder}/loss_over_epoch.png')
        return model
            
    def eval_train(self, model, dp, train_data):
    
        args = self.args
        train_inputs, train_targets = train_data
        
        # FIXED: PyTorch inference (replaces model.predict)
        model.eval()
        with torch.no_grad():
            # Convert to tensors and move to device
            input_tensors = [torch.FloatTensor(inp).to(self.device) for inp in train_inputs]
            
            # Get predictions
            outputs = model(input_tensors, training=False)  # FIXED: Added training=False
            x_is_pred = outputs[0].cpu().numpy()
            y_is_pred = outputs[1].cpu().numpy()
        
        if args.scale_data:
            x_is_pred = dp.apply_scaler('scaler_x', x_is_pred, inverse=True)
            y_is_pred = dp.apply_scaler('scaler_y', y_is_pred, inverse=True)
            x_is_truth = dp.apply_scaler('scaler_x', train_targets[0], inverse=True)
            y_is_truth = dp.apply_scaler('scaler_y', train_targets[1], inverse=True)
        else:
            x_is_truth, y_is_truth = train_targets[0], train_targets[1]
            
        plot_fitted_val(args, (x_is_pred, y_is_pred), (x_is_truth, y_is_truth), 
                    x_col=self.df_info['x_columns'].index(args.X_COLNAME), 
                    y_col=self.df_info['y_columns'].index(args.Y_COLNAME), 
                    time_steps=None, 
                    save_as_file=f'{args.output_folder}/train_fit_static.png')
    
    def config_predictor(self, model, dp):
        
        args = self.args
        predictor = self.cls_constructor.create_predictor(model, dp, apply_inv_scaler=args.scale_data)
        
        return predictor
    
    def run_forecast_one_set(self, predictor, dp, y_start_id, x_start_id):
        
        """ helper function for running one set of forecast: F, N1, N2, N3 """
        args = self.args
        xdata, ydata = self.raw_data
        
        assert x_start_id == args.freq_ratio * (y_start_id - 1)
        predictions_by_vintage = {}
        
        for x_step in range(args.freq_ratio+1):
            experiment_tag = 'F' if x_step == 0 else f'N{x_step}'
            inputs, targets = dp.create_one_forecast_sample(xdata,
                                                            ydata,
                                                            x_start_id,
                                                            y_start_id,
                                                            x_step=x_step,
                                                            horizon=args.horizon,
                                                            apply_scaler=True if args.scale_data else False,
                                                            verbose=False)
                                                            
            x_pred, y_pred = predictor(inputs, x_step=x_step, horizon=args.horizon)
            predictions_by_vintage[experiment_tag] = {'x_pred': x_pred, 'y_pred': y_pred}
        return predictions_by_vintage
    
    def add_prediction_to_collector(self, predictions_by_vintage, T_datestamp, x_PRED_collector=[], y_PRED_collector=[]):
        
        args = self.args
        
        ## initialization for recording the forecast
        x_numeric_col_keys = [f'step_{i+1}' for i in range(args.freq_ratio * args.horizon)]
        y_numeric_col_keys = [f'step_{i+1}' for i in range(args.horizon)]
        
        ## extract prediction
        for vintage, predictions in predictions_by_vintage.items():
            x_pred, y_pred = predictions['x_pred'], predictions['y_pred']
            for col_id, variable_name in enumerate(self.df_info['x_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(x_numeric_col_keys, list(x_pred[:,col_id]))))
                x_PRED_collector.append(temp)
            for col_id, variable_name in enumerate(self.df_info['y_columns']):
                temp = {'prev_QE': T_datestamp, 'tag': vintage, 'variable_name': variable_name}
                temp.update(dict(zip(y_numeric_col_keys, list(y_pred[:,col_id]))))
                y_PRED_collector.append(temp)
        
    def find_closest_index_with_tolerance(self, timestamp_list, target_timestamp, tolerance_hours=4):
        """
        Find the closest timestamp within tolerance hours, return its index
        """
        import pandas as pd
        
        timestamps = pd.Series(timestamp_list)
        time_diffs = (timestamps - target_timestamp).abs()
        
        # Filter to only timestamps within tolerance
        tolerance_td = pd.Timedelta(hours=tolerance_hours)
        within_tolerance = time_diffs <= tolerance_td
        
        if not within_tolerance.any():
            return None  # No timestamps within tolerance
        
        # Find the closest timestamp within tolerance
        closest_idx = time_diffs[within_tolerance].argmin()
        # Get the actual index in the original list
        actual_idx = timestamps[within_tolerance].index[closest_idx]
        
        return actual_idx

    def run_forecast(self):
        """ main function """
    
        args = self.args
        
        # FINAL FIX: Ensure first_prediction_date has same precision as data
        if hasattr(args, 'first_prediction_date'):
            original_date = args.first_prediction_date
            args.first_prediction_date = args.first_prediction_date.round('H')
            print(f"🔧 FINAL FIX: Rounded first_prediction_date from {original_date} to {args.first_prediction_date}")
        
        # Check timestamp availability with tolerance
        print(f"🔍 Looking for {args.first_prediction_date} with ±4 hour tolerance...")
        
        # For vitals: exact match (hourly data)
        if args.first_prediction_date in self.df_info['x_index']:
            x_prediction_idx = self.df_info['x_index'].index(args.first_prediction_date)
            print(f"✅ Found exact vitals timestamp at index {x_prediction_idx}")
        else:
            print(f"❌ Exact vitals timestamp not found")
            return
        
        # For labs: tolerance-based match (every 12 hours)
        y_prediction_idx = self.find_closest_index_with_tolerance(
            self.df_info['y_index'], 
            args.first_prediction_date, 
            tolerance_hours=4
        )
        
        if y_prediction_idx is not None:
            actual_y_timestamp = self.df_info['y_index'][y_prediction_idx]
            time_diff = abs((actual_y_timestamp - args.first_prediction_date).total_seconds() / 3600)
            print(f"✅ Found labs timestamp at index {y_prediction_idx}: {actual_y_timestamp}")
            print(f"   Time difference: {time_diff:.1f} hours (within 4-hour tolerance)")
        else:
            print(f"❌ No labs timestamp found within 4-hour tolerance")
            return
        
        ## initialization for recording the forecast
        x_PRED_collector, y_PRED_collector = [], []
        
        if args.mode == 'static':
            
            # Use the found indices with tolerance
            x_train_end = x_prediction_idx - args.freq_ratio + 1 ## +1 to ensure the index is inclusive
            y_train_end = y_prediction_idx - 2 + 1 ## +1 to ensure the index is inclusive
            
            print(f"📊 Training data endpoints: x_train_end={x_train_end}, y_train_end={y_train_end}")
            print(f"   Using vitals up to: {self.df_info['x_index'][x_prediction_idx]}")
            print(f"   Using labs up to: {self.df_info['y_index'][y_prediction_idx]}")
            
            ## set up and train model
            dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val=args.n_val)
            model = self.config_and_train_model(train_data, val_data=val_data)
            self.eval_train(model, dp, train_data)
            predictor = self.config_predictor(model, dp)
            
            ## run rolling forecast based on the trained model
            y_start_id = y_prediction_idx - (args.Ty - 1)
            x_start_id = args.freq_ratio * (y_start_id - 1)
            test_size = self.df_info['y_total_obs'] - y_prediction_idx
            
            print(f'[forecast starts] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            for experiment_id in range(test_size):
                
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                x_range = [self.df_info['x_index'][x_start_id], self.df_info['x_index'][x_start_id + args.Lx - 1]]
                y_range = [self.df_info['y_index'][y_start_id], self.df_info['y_index'][y_start_id + args.Ty - 2]] ## -2 since Ty = Ly+1
                
                print(f" >> id = {experiment_id+1}/{test_size}: prev timestamp = {T_datestamp}; x_input_range = {x_range}, y_input_range = {y_range}")
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
            
                # update x_start_id and y_start_id
                x_start_id += args.freq_ratio
                y_start_id += 1
                
            print(f'[forecast ends] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
        elif args.mode == 'dynamic':
            
            # For dynamic mode, apply tolerance to each prediction date
            offset = y_prediction_idx
            test_size = self.df_info['y_total_obs'] - offset
            
            for experiment_id in range(test_size):
                
                pred_date = self.df_info['y_index'][offset + experiment_id]
                
                print(f' >> id = {experiment_id+1}/{test_size}: next QE = {pred_date.strftime("%Y-%m-%d")}')
                
                # Find corresponding x index with tolerance
                x_pred_idx = self.find_closest_index_with_tolerance(
                    self.df_info['x_index'], pred_date, tolerance_hours=4
                )
                
                if x_pred_idx is None:
                    print(f"   Skipping - no vitals data within tolerance")
                    continue
                
                ## set up and train model
                x_train_end = x_pred_idx - args.freq_ratio + 1
                y_train_end = (offset + experiment_id) - 2 + 1
                
                dp, train_data, val_data = self.generate_train_val_datasets(x_train_end, y_train_end, n_val=args.n_val)
                model = self.config_and_train_model(train_data, val_data=val_data)
                self.eval_train(model, dp, train_data)
                predictor = self.config_predictor(model, dp)
                
                ## run forecast
                y_start_id = (offset + experiment_id) - (args.Ty - 1)
                x_start_id = args.freq_ratio * (y_start_id - 1)
                T_datestamp = self.df_info['x_index'][x_start_id + args.Lx - 1]
                
                predictions_by_vintage = self.run_forecast_one_set(predictor, dp, y_start_id, x_start_id)
                self.add_prediction_to_collector(predictions_by_vintage, T_datestamp, x_PRED_collector, y_PRED_collector)
                
                del predictor
                del model
                # Clear GPU memory if using PyTorch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        x_PRED_df, y_PRED_df = pd.DataFrame(x_PRED_collector), pd.DataFrame(y_PRED_collector)
        
        # Export results based on config settings
        if hasattr(args, 'export_to_parquet') and args.export_to_parquet:
            print(f'[exporting predictions to parquet] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            x_PRED_df.to_parquet(f'{args.output_folder}/x_predictions.parquet', index=False)
            y_PRED_df.to_parquet(f'{args.output_folder}/y_predictions.parquet', index=False)
        
        if hasattr(args, 'export_to_csv') and args.export_to_csv:
            print(f'[exporting predictions to CSV] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            x_PRED_df.to_csv(f'{args.output_folder}/x_predictions.csv', index=False)
            y_PRED_df.to_csv(f'{args.output_folder}/y_predictions.csv', index=False)
        
        # Default Excel export
        with pd.ExcelWriter(args.output_filename) as writer:
            x_PRED_df.to_excel(writer, sheet_name='x_prediction', index=False)
            y_PRED_df.to_excel(writer, sheet_name='y_prediction', index=False)
        
        print(f'[predictions saved] {args.output_filename} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'✅ FORECAST COMPLETED SUCCESSFULLY!')
        print(f'   Generated {len(x_PRED_df)} vitals predictions and {len(y_PRED_df)} labs predictions')