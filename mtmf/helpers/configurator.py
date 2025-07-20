"""
Classes for configuring the environment, including loading configs, creating directories etc.
PyTorch Version
"""

import sys
import yaml
import os
import shutil
import pickle
import pandas as pd
import numpy as np

class EnvConfigurator():
    """
    Environment configurator for real data experiments
    Handles YAML config loading, directory setup, and argument configuration
    """
    
    def __init__(self, args, data_folder='data_electricity', output_meta_folder='output_electricity'):
        """
        Initialize environment configurator
        
        Args:
            args: parser.parse_args() from command line input
            data_folder: folder containing the data files
            output_meta_folder: parent folder for outputs
        """
        self.data_folder = data_folder
        self.output_meta_folder = output_meta_folder
        
        # Load main config file
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        # Handle includes (like defaults.yaml)
        if config.get('includes') is not None:
            for item in config.get('includes'):
                with open(item) as handle:
                    config_this_item = yaml.safe_load(handle)
                for key in config_this_item:
                    for k, v in config_this_item[key].items():
                        setattr(args, k, v)
            del config['includes']
        
        # Apply config settings to args
        if config['setup']['model_type'] not in ['GBM','DeepAR','NHiTS','ARIMA']:
            # For neural network models (our case)
            for key in config:
                for k, v in config[key].items():
                    setattr(args, k, v)
        else:
            # For other model types with special parameter handling
            for key in config:
                if key in ['hyper_params','hyper_params_x', 'hyper_params_y']:
                    setattr(args, key, config[key])
                else:
                    for k, v in config[key].items():
                        setattr(args, k, v)
        
        # Ensure command line inputs have correct types
        if hasattr(args, 'verbose'):
            setattr(args, 'verbose', int(args.verbose))
        
        self.args = args
    
    def config_directory_and_add_to_args(self, delete_existing=False):
        """
        Configure output directories and add paths to args
        
        Args:
            delete_existing: whether to delete existing output folder
        """
        # Set data folder
        setattr(self.args, 'data_folder', self.data_folder)
        
        # Set output folder structure
        setattr(self.args, 'output_parent_folder', f"{self.output_meta_folder}")
        
        if hasattr(self.args, 'output_folder_override') and len(self.args.output_folder_override):
            setattr(self.args, 'output_folder', f"{self.args.output_parent_folder}/{self.args.output_folder_override}")
        else:
            # Default output folder naming
            setattr(self.args, 'output_folder', f"{self.args.output_parent_folder}/{self.args.model_type}")
        
        # Create output directory
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
            print(f'Folder {self.args.output_folder}/ created')
        else:
            if delete_existing:
                print(f'Folder {self.args.output_folder}/ exists; deleted and recreated')
                shutil.rmtree(self.args.output_folder)
                os.makedirs(self.args.output_folder)
            else:
                print(f'Folder {self.args.output_folder}/ exists; no action needed')
    
    def config_args(self, pickle_args=True):
        """
        Configure arguments based on data dimensions and model requirements
        
        Args:
            pickle_args: whether to save args to pickle file
            
        Returns:
            configured args object
        """
        # Read data to determine dimensions
        try:
            # Try to read electricity.xlsx first (our processed data)
            x_data = pd.read_excel(f"{self.data_folder}/electricity.xlsx", 
                                  index_col='timestamp', sheet_name='x')
            y_data = pd.read_excel(f"{self.data_folder}/electricity.xlsx", 
                                  index_col='timestamp', sheet_name='y')
        except FileNotFoundError:
            # Fallback to any Excel file in the folder
            excel_files = [f for f in os.listdir(self.data_folder) if f.endswith('.xlsx')]
            if excel_files:
                filepath = f"{self.data_folder}/{excel_files[0]}"
                x_data = pd.read_excel(filepath, index_col='timestamp', sheet_name='x')
                y_data = pd.read_excel(filepath, index_col='timestamp', sheet_name='y')
            else:
                raise FileNotFoundError(f"No Excel data files found in {self.data_folder}")
        
        xdata, ydata = x_data.values, y_data.values
        
        # Set data dimensions
        setattr(self.args, 'dim_x', xdata.shape[1])
        setattr(self.args, 'dim_y', ydata.shape[1])
        
        print(f"Data dimensions: dim_x={self.args.dim_x}, dim_y={self.args.dim_y}")
        print(f"Data shapes: X={xdata.shape}, Y={ydata.shape}")
        
        # Model-specific configurations
        if self.args.model_type in ['MTMFSeq2One', 'MLP', 'RNN', 'GBM']:
            setattr(self.args, 'Tx', 1)  # Single-step output
        
        if self.args.model_type in ['MLP', 'GBM', 'DeepAR', 'NHiTS', 'ARIMA']:
            setattr(self.args, 'zero_pad', False)
        else:
            setattr(self.args, 'zero_pad', True)
        
        # Save configuration
        if pickle_args:
            with open(f"{self.args.output_folder}/args.pkl", 'wb') as f:
                pickle.dump(self.args, f)
            print(f"Arguments saved to {self.args.output_folder}/args.pkl")
        
        return self.args