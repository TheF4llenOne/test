"""
Classes for configuring the environment, including loading configs, creating directories etc.
"""

import sys
import yaml
import os
import shutil
import pickle
import pandas as pd
import numpy as np

class SimEnvConfigurator():
    # Keep your existing SimEnvConfigurator as-is
    def __init__(self, args, data_folder = 'data_sim', output_meta_folder = 'output_sim'):
        """
        initialization
        args = parser.parse_args() comes from parsing cmd line input
        """
        self.data_folder = data_folder
        self.output_meta_folder = output_meta_folder
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if config.get('includes') is not None:
            for item in config.get('includes'):
                with open(item) as handle:
                    config_this_item = yaml.safe_load(handle)
                for key in config_this_item:
                    for k, v in config_this_item[key].items():
                        setattr(args, k, v)
            del config['includes']
        
        if config['setup']['model_type'] not in ['GBM','DeepAR','NHiTS','ARIMA']:
            for key in config:
                for k, v in config[key].items():
                    setattr(args, k, v)
        else:
            for key in config:
                if key in ['hyper_params','hyper_params_x', 'hyper_params_y']:
                    setattr(args, key, config[key])
                else:
                    for k, v in config[key].items():
                        setattr(args, k, v)
                    
        ## ensure cmd line input is of the correct type
        setattr(args,'train_size',int(args.train_size))
        if hasattr(args, 'use_ckpt'):
            setattr(args,'use_ckpt',int(args.use_ckpt))
        else:
            setattr(args,'use_ckpt',0)
        if hasattr(args, 'verbose'):
            setattr(args,'verbose',int(args.verbose))
        
        self.args = args
    
    def config_directory_and_add_to_args(self, delete_existing=False):
    
        setattr(self.args,'data_folder',self.data_folder)
        
        setattr(self.args,'output_parent_folder', f"{self.output_meta_folder}/{self.args.ds_name}")
        if hasattr(self.args, 'output_folder_override') and len(self.args.output_folder_override):
            setattr(self.args,'output_folder', f"{self.args.output_parent_folder}/{self.args.output_folder_override}")
        else:
            setattr(self.args,'output_folder', f"{self.args.output_parent_folder}/{self.args.model_type}_{self.args.train_size}")
        
        if self.args.use_ckpt:
            setattr(self.args,'ckpt_folder',f'{self.args.output_folder}/ckpt')
            
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
            print(f'folder {self.args.output_folder}/ created')
        else:
            if delete_existing:
                print(f'folder {self.args.output_folder}/ exists; deleted and recreated to ensure there is no stale output for this run')
                shutil.rmtree(self.args.output_folder)
                os.mkdir(self.args.output_folder)
            else:
                print(f'folder {self.args.output_folder}/ exists; no action needed')
                pass
            
    def config_args(self, pickle_args = True):
        """Handles hour-based medical data properly"""
        
        # Read parquet files for medical data using config settings
        vitals_file = getattr(self.args, 'vitals_file', 'vitals_forward_filled_cleaned.parquet')
        labs_file = getattr(self.args, 'labs_file', 'labs_filtered_cleaned.parquet')
        
        x = pd.read_parquet(f"{self.args.data_folder}/{vitals_file}")
        y = pd.read_parquet(f"{self.args.data_folder}/{labs_file}")
        
        # Get time column names from config or use defaults
        vitals_time_col = getattr(self.args, 'vitals_time_col', 'hour_bin')
        labs_time_col = getattr(self.args, 'labs_time_col', 'bin_start_hour')
        
        # Set timestamp columns as index
        x = x.set_index(vitals_time_col)
        y = y.set_index(labs_time_col)
        
        # Convert hour indices to datetime for framework compatibility
        base_date = pd.Timestamp('2020-01-01')
        x.index = base_date + pd.to_timedelta(x.index, unit='h')
        y.index = base_date + pd.to_timedelta(y.index, unit='h')
        
        # Round timestamps to hour precision to avoid nanosecond precision issues
        x.index = x.index.round('H')
        y.index = y.index.round('H')
        
        # FIX: Now handle first_prediction_date correctly using the same base_date
        if self.raw_first_prediction_date is not None:
            if isinstance(self.raw_first_prediction_date, (int, float)):
                # It's an hour index - convert it using the same method as the data
                prediction_datetime = base_date + pd.to_timedelta(self.raw_first_prediction_date, unit='h')
                # IMPORTANT: Round to same precision as data
                prediction_datetime = prediction_datetime.round('H')
                setattr(self.args, 'first_prediction_date', prediction_datetime)
                print(f"✅ FIXED: Converted hour index {self.raw_first_prediction_date} to datetime {prediction_datetime}")
            else:
                # It's already a datetime string - parse it normally and round
                prediction_datetime = pd.to_datetime(self.raw_first_prediction_date).round('H')
                setattr(self.args, 'first_prediction_date', prediction_datetime)
                print(f"✅ Using datetime string: {prediction_datetime}")
        else:
            # Find a reasonable split point (e.g., 80% for training)
            total_y_points = len(y)
            split_point_idx = int(0.8 * total_y_points)
            split_y_time = y.index.sort_values()[split_point_idx]
            setattr(self.args, 'first_prediction_date', split_y_time)
            print(f"✅ Auto-calculated prediction date (80% through data): {split_y_time}")
        
        # Remove patient ID columns if present (not features for modeling)
        if 'unique_patient_id' in x.columns:
            x = x.drop('unique_patient_id', axis=1)
        if 'unique_patient_id' in y.columns:
            y = y.drop('unique_patient_id', axis=1)
            
        xdata, ydata = x.values, y.values
        
        ## set dimension based on actual data
        setattr(self.args, 'dim_x', xdata.shape[1])
        setattr(self.args, 'dim_y', ydata.shape[1])
        
        print(f"Data dimensions: X (vitals) = {xdata.shape}, Y (labs) = {ydata.shape}")
        print(f"X features ({len(x.columns)}): {list(x.columns)}")
        print(f"Y features ({len(y.columns)}): {list(y.columns)}")
        print(f"Time range: {x.index.min()} to {x.index.max()}")
    
        ## setup for benchmark models where the output for high freq is not a sequence
        if self.args.model_type in ['MTMFSeq2One','MLP','RNN','GBM']:
            setattr(self.args, 'Tx', 1)
            
        if self.args.model_type in ['MLP','GBM','DeepAR','NHiTS','ARIMA']:
            setattr(self.args, 'zero_pad', False)
        else:
            setattr(self.args, 'zero_pad', True)
        
        if pickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        return self.args

class EnvConfigurator():
    """FIXED VERSION for medical hour-based data"""
    def __init__(self, args, data_folder = 'data_medical', output_meta_folder = 'output_medical'):
        """
        initialization
        args = parser.parse_args() comes from parsing cmd line input
        """
        self.data_folder = data_folder
        self.output_meta_folder = output_meta_folder
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        if config.get('includes') is not None:
            for item in config.get('includes'):
                with open(item) as handle:
                    config_this_item = yaml.safe_load(handle)
                for key in config_this_item:
                    for k, v in config_this_item[key].items():
                        setattr(args, k, v)
            del config['includes']
        
        if config['setup']['model_type'] not in ['GBM','DeepAR','NHiTS','ARIMA']:
            for key in config:
                for k, v in config[key].items():
                    setattr(args, k, v)
        else:
            for key in config:
                if key in ['hyper_params','hyper_params_x', 'hyper_params_y']:
                    setattr(args, key, config[key])
                else:
                    for k, v in config[key].items():
                        setattr(args, k, v)
        
        # FIX: Don't convert first_prediction_date here - do it later when we know the base_date
        # Store the raw value for now
        if hasattr(args, 'first_prediction_date'):
            # Keep the original value (could be hour index or datetime string)
            self.raw_first_prediction_date = args.first_prediction_date
        else:
            self.raw_first_prediction_date = None
        
        ## ensure cmd line input is of the correct type
        if hasattr(args,'verbose'):
            setattr(args,'verbose',int(args.verbose))
        
        self.args = args
    
    def config_directory_and_add_to_args(self, delete_existing=False):
    
        setattr(self.args,'data_folder',self.data_folder)
        if hasattr(self.args, 'output_folder_override') and len(self.args.output_folder_override):
            setattr(self.args,'output_folder', f"{self.output_meta_folder}/{self.args.output_folder_override}")
        else:
            setattr(self.args,'output_folder', f"{self.output_meta_folder}/{self.args.model_type}_{self.args.mode}")
        
        # Support multiple output formats
        if hasattr(self.args, 'output_format') and self.args.output_format == 'csv':
            setattr(self.args,'output_filename', f"{self.args.output_folder}/predictions.csv")
        elif hasattr(self.args, 'output_format') and self.args.output_format == 'parquet':
            setattr(self.args,'output_filename', f"{self.args.output_folder}/predictions.parquet")
        else:
            setattr(self.args,'output_filename', f"{self.args.output_folder}/predictions.xlsx")
            
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
            print(f'folder {self.args.output_folder}/ created')
        else:
            if delete_existing:
                print(f'folder {self.args.output_folder}/ exists; deleted and recreated to ensure there is no stale output for this run')
                shutil.rmtree(self.args.output_folder)
                os.mkdir(self.args.output_folder)
            else:
                print(f'folder {self.args.output_folder}/ exists; no action needed')
                pass
                
    def config_args(self, pickle_args = True):
        """FIXED VERSION: Handles hour-based medical data properly"""
        
        # Read parquet files for medical data using config settings
        vitals_file = getattr(self.args, 'vitals_file', 'vitals_forward_filled_cleaned.parquet')
        labs_file = getattr(self.args, 'labs_file', 'labs_filtered_cleaned.parquet')
        
        x = pd.read_parquet(f"{self.args.data_folder}/{vitals_file}")
        y = pd.read_parquet(f"{self.args.data_folder}/{labs_file}")
        
        # Get time column names from config or use defaults
        vitals_time_col = getattr(self.args, 'vitals_time_col', 'hour_bin')
        labs_time_col = getattr(self.args, 'labs_time_col', 'bin_start_hour')
        
        # Set timestamp columns as index
        x = x.set_index(vitals_time_col)
        y = y.set_index(labs_time_col)
        
        # Convert hour indices to datetime for framework compatibility
        base_date = pd.Timestamp('2020-01-01')
        x.index = base_date + pd.to_timedelta(x.index, unit='h')
        y.index = base_date + pd.to_timedelta(y.index, unit='h')
        
        # FIX: Now handle first_prediction_date correctly using the same base_date
        if self.raw_first_prediction_date is not None:
            if isinstance(self.raw_first_prediction_date, (int, float)):
                # It's an hour index - convert it using the same method as the data
                prediction_datetime = base_date + pd.to_timedelta(self.raw_first_prediction_date, unit='h')
                setattr(self.args, 'first_prediction_date', prediction_datetime)
                print(f"✅ FIXED: Converted hour index {self.raw_first_prediction_date} to datetime {prediction_datetime}")
            else:
                # It's already a datetime string - parse it normally
                setattr(self.args, 'first_prediction_date', pd.to_datetime(self.raw_first_prediction_date))
                print(f"✅ Using datetime string: {self.args.first_prediction_date}")
        else:
            # Find a reasonable split point (e.g., 80% for training)
            total_y_points = len(y)
            split_point_idx = int(0.8 * total_y_points)
            split_y_time = y.index.sort_values()[split_point_idx]
            setattr(self.args, 'first_prediction_date', split_y_time)
            print(f"✅ Auto-calculated prediction date (80% through data): {split_y_time}")
        
        # Remove patient ID columns if present (not features for modeling)
        if 'unique_patient_id' in x.columns:
            x = x.drop('unique_patient_id', axis=1)
        if 'unique_patient_id' in y.columns:
            y = y.drop('unique_patient_id', axis=1)
            
        xdata, ydata = x.values, y.values
        
        ## set dimension based on actual data
        setattr(self.args, 'dim_x', xdata.shape[1])
        setattr(self.args, 'dim_y', ydata.shape[1])
        
        print(f"Data dimensions: X (vitals) = {xdata.shape}, Y (labs) = {ydata.shape}")
        print(f"X features ({len(x.columns)}): {list(x.columns)}")
        print(f"Y features ({len(y.columns)}): {list(y.columns)}")
        print(f"Time range: {x.index.min()} to {x.index.max()}")
    
        ## setup for benchmark models where the output for high freq is not a sequence
        if self.args.model_type in ['MTMFSeq2One','MLP','RNN','GBM']:
            setattr(self.args, 'Tx', 1)
            
        if self.args.model_type in ['MLP','GBM','DeepAR','NHiTS','ARIMA']:
            setattr(self.args, 'zero_pad', False)
        else:
            setattr(self.args, 'zero_pad', True)
        
        if pickle_args:
            with open(f"{self.args.output_folder}/args.pickle","wb") as handle:
                pickle.dump(self.args, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        return self.args