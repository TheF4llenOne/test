# Simple run script for medical data
import sys
import argparse
import os
from datetime import datetime
import pandas as pd
import torch
from helpers import EnvConfigurator
from helpers import nnTrainer

parser = argparse.ArgumentParser(description='train model on mixed frequency medical data')
parser.add_argument('--config', default='configs/medical/seq2seq.yaml')
parser.add_argument('--mode', default='static', type=str, help='training mode (static/dynamic)')
parser.add_argument('--verbose', default=1, type=int, help='verbose interval')
parser.add_argument('--output_folder_override', default='', type=str, help='override for output_folder; leave blank if default is used')

def main():
    raw_args = parser.parse_args()
    setattr(raw_args, 'verbose', int(raw_args.verbose))
    
    dataset_name = 'medical'
    actual_data_folder = 'data_medical'
    
    if not os.path.exists(actual_data_folder):
        print(f"ERROR: Data folder '{actual_data_folder}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    env_configurator = EnvConfigurator(raw_args,
                                       data_folder=actual_data_folder,
                                       output_meta_folder=f'output_{dataset_name}')
                                       
    env_configurator.config_directory_and_add_to_args()
    args = env_configurator.config_args(pickle_args=True)
    
    print(f"Using data folder: {args.data_folder}")
    
    vitals_file = f"{args.data_folder}/{getattr(args, 'vitals_file', 'vitals_forward_filled_cleaned.parquet')}"
    labs_file = f"{args.data_folder}/{getattr(args, 'labs_file', 'labs_filtered_cleaned.parquet')}"
    print(f"  Vitals file: {vitals_file} - {'EXISTS' if os.path.exists(vitals_file) else 'MISSING'}")
    print(f"  Labs file: {labs_file} - {'EXISTS' if os.path.exists(labs_file) else 'MISSING'}")
    
    if not os.path.exists(vitals_file) or not os.path.exists(labs_file):
        print("ERROR: Data files are missing!")
        return
    
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        print(vars(args), file=f)
    
    print("="*60)
    print("MEDICAL MIXED-FREQUENCY FORECASTING")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Forecasting Goals:")
    print(f"  - Vitals: Predict {args.Tx} hour(s) ahead")
    print(f"  - Labs: Predict {args.horizon} period(s) ahead ({args.horizon * args.freq_ratio} hours)")
    print(f"Architecture: {args.n_a} hidden units, {args.n_s} context dim")
    print("="*60)
    
    trainer = nnTrainer(args=args, seed=411)
    trainer.set_seed()
    trainer.source_data()
    trainer.run_forecast()

if __name__ == "__main__":
    print('=============================================================')
    print(f'>>> PyTorch version: {torch.__version__}; CUDA available: {torch.cuda.is_available()}')
    print(f'>>> Current working directory: {os.getcwd()}')
    print(f'>>> {sys.argv[0]} started execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')