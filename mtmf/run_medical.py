"""
Running MTMFSeq2Seq on medical parquet data
Forecasts:
- Vitals: 1 hour ahead  
- Labs: 12 hours ahead (1 lab period)

To run:
python run_medical.py --config=configs/medical/seq2seq.yaml --verbose=1

"""

import sys
import yaml
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
    """ main function for running medical data experiments """
    raw_args = parser.parse_args()
    setattr(raw_args, 'verbose', int(raw_args.verbose))
    
    dataset_name = 'medical'
    
    env_configurator = EnvConfigurator(raw_args,
                                       data_folder = f'data_{dataset_name}',
                                       output_meta_folder = f'output_{dataset_name}')
                                       
    env_configurator.config_directory_and_add_to_args()
    args = env_configurator.config_args(pickle_args=True)
    
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
    
    # Use PyTorch trainer
    trainer = nnTrainer(args=args, seed=411)
    
    trainer.set_seed()
    trainer.source_data()
    trainer.run_forecast()
    
if __name__ == "__main__":
    print('=============================================================')
    print(f'>>> PyTorch version: {torch.__version__}; CUDA available: {torch.cuda.is_available()}')
    print(f'>>> {sys.argv[0]} started execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    main()
    print(f'>>> {sys.argv[0]} finished execution on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=============================================================')