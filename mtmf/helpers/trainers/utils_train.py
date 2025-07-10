"""
utility functions for additional recording during training - PyTorch Compatible
"""
import numpy as np
import pandas as pd
import math
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.gridspec import GridSpec

def plot_loss_over_epoch(history, args, save_as_file=""):

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[:,0])
    # PyTorch modification: history is a dict, not history.history
    ax0.plot(history['loss'], label='train',color='red')
    if history.get('val_loss') is not None:
        ax0.plot(history['val_loss'], label='validation',color='green')
    ax0.set_title('Total Loss')
    ax0.legend(loc='upper right')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(history['output_1_loss'], label='train',color='red')
    if history.get('val_output_1_loss') is not None:
        ax1.plot(history['val_output_1_loss'], label='validation',color='green')
    ax1.set_title('output_1: x')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(history['output_2_loss'], label='train',color='red')
    if history.get('val_output_2_loss') is not None:
        ax2.plot(history['val_output_2_loss'], label='validation',color='green')
    ax2.set_title('output_2: y')
    ax2.legend(loc='upper right')
    
    if len(save_as_file) > 0:
        fig.savefig(save_as_file)

def plot_fitted_val(args, pred, truth, x_col = None, y_col = None, time_steps = 100, save_as_file = True):
    
    x_pred, y_pred = pred
    x_truth, y_truth = truth
    
    if x_col is None:
        x_col = np.random.choice(list(range(args.dim_x)))
    if y_col is None:
        y_col = np.random.choice(list(range(args.dim_y)))
    
    if time_steps is not None: ## subset
        ticks = np.arange(time_steps)
        # PyTorch modification: Add bounds checking to avoid errors
        max_samples = min(time_steps, x_pred.shape[0])
        sample_ids = np.random.choice(a=list(range(x_pred.shape[0])), size=max_samples, replace=False)
        x_pred, x_truth = x_pred[sample_ids], x_truth[sample_ids]
        y_pred, y_truth = y_pred[sample_ids], y_truth[sample_ids]
        ticks = np.arange(max_samples)
    else:
        ticks = np.arange(x_pred.shape[0])
    
    if args.Tx > 1:
        num_cols = math.ceil((args.Tx+1)/2.)
        fig, axs = plt.subplots(2, int(num_cols), figsize=(12,6), constrained_layout=True)
        for step_id in range(args.Tx):
            r, c = step_id//int(num_cols), step_id%int(num_cols)
            # PyTorch modification: Handle case where num_cols might be 1
            if num_cols == 1:
                curr_ax = axs[r]
            else:
                curr_ax = axs[r,c]
            curr_ax.plot(ticks,x_pred[:, step_id, x_col],label='model',color='red')
            curr_ax.plot(ticks,x_truth[:, step_id, x_col],label='truth',color='black')
            curr_ax.legend(loc='lower right')
            curr_ax.set_title(f'step_id={step_id+1}, x_col={x_col}, model vs. truth')
        
        # Handle y plot
        if num_cols == 1:
            axs[-1].plot(ticks,y_pred[:,y_col],label='model',color='red')
            axs[-1].plot(ticks,y_truth[:,y_col],label='truth',color='black')
            axs[-1].legend()
            axs[-1].set_title(f'y_col={y_col}, model vs. truth')
        else:
            axs[-1,-1].plot(ticks,y_pred[:,y_col],label='model',color='red')
            axs[-1,-1].plot(ticks,y_truth[:,y_col],label='truth',color='black')
            axs[-1,-1].legend()
            axs[-1,-1].set_title(f'y_col={y_col}, model vs. truth')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12,3), constrained_layout=True)
        axs[0].plot(ticks,x_pred[:,x_col],label='model',color='red')
        axs[0].plot(ticks,x_truth[:,x_col],label='truth',color='black')
        axs[0].legend(loc='lower right')
        axs[0].set_title(f'step_id=1, x_col={x_col}, model vs. truth')
        axs[1].plot(ticks,y_pred[:,y_col],label='model',color='red')
        axs[1].plot(ticks,y_truth[:,y_col],label='truth',color='black')
        axs[1].legend()
        axs[1].set_title(f'y_col={y_col}, model vs. truth')
    
    if isinstance(save_as_file, str) and len(save_as_file) > 0:
        fig.savefig(save_as_file)
    plt.close()
    
def export_error_to_excel(df_x_err, df_y_err, x_err_collect, y_err_collect, args):
    
    print('######################################')
    print('## rmse summary for x (step_4):')
    for tag in x_err_collect.keys():
        median = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'median'), tag].values[0]
        mean = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'mean'), tag].values[0]
        std = df_x_err.loc[(df_x_err.index=='step_4') & (df_x_err['metric'] == 'std'), tag].values[0]
        print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
    print('## rmse summary for y (step_1):')
    for tag in y_err_collect.keys():
        median = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'median'), tag].values[0]
        mean = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'mean'), tag].values[0]
        std = df_y_err.loc[(df_y_err.index=='step_1') & (df_y_err['metric'] == 'std'), tag].values[0]
        print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
    print('######################################')
    
    output_folder = args.output_folder
    
    # Export summary DataFrames
    if hasattr(args, 'export_to_csv') and args.export_to_csv:
        print(f'[exporting error summary to CSV] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        df_x_err.to_csv(f'{output_folder}/summary_x_err.csv', index=True)
        df_y_err.to_csv(f'{output_folder}/summary_y_err.csv', index=True)
        
        for experiment_tag, arr in x_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_csv(f'{output_folder}/x_{experiment_tag}.csv', index=True)
        for experiment_tag, arr in y_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_csv(f'{output_folder}/y_{experiment_tag}.csv', index=True)
    
    # Export to parquet if requested
    if hasattr(args, 'export_to_parquet') and args.export_to_parquet:
        print(f'[exporting error summary to parquet] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        df_x_err.to_parquet(f'{output_folder}/summary_x_err.parquet', index=True)
        df_y_err.to_parquet(f'{output_folder}/summary_y_err.parquet', index=True)
        
        for experiment_tag, arr in x_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_parquet(f'{output_folder}/x_{experiment_tag}.parquet', index=True)
        for experiment_tag, arr in y_err_collect.items():
            df = pd.DataFrame(data=arr,columns=[f'step{i+1}' for i in range(arr.shape[1])],index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_parquet(f'{output_folder}/y_{experiment_tag}.parquet', index=True)