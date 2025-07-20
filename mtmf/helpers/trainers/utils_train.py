"""
utility functions for additional recording during training - PyTorch Version
"""
import numpy as np
import pandas as pd
import math
from datetime import datetime

import matplotlib.pyplot as plt
# Handle different matplotlib versions gracefully
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        # Fallback to default matplotlib grid style
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
from matplotlib.gridspec import GridSpec

def plot_loss_over_epoch(history, args, save_as_file=""):
    """
    Plot training loss over epochs - matches original TensorFlow version exactly
    
    Args:
        history: dict with 'loss', 'val_loss', 'output_1_loss', etc. (PyTorch format)
        args: configuration arguments
        save_as_file: path to save file (empty string means don't save)
    """
    # Convert PyTorch history format to TensorFlow history.history format
    class HistoryWrapper:
        def __init__(self, history_dict):
            self.history = history_dict
    
    history = HistoryWrapper(history)

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax0 = fig.add_subplot(gs[:,0])
    ax0.plot(history.history['loss'], label='train',color='red')
    if history.history.get('val_loss') is not None:
        ax0.plot(history.history['val_loss'], label='validation',color='green')
    ax0.legend(loc='upper right')

    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(history.history['output_1_loss'], label='train',color='red')
    if history.history.get('val_output_1_loss') is not None:
        ax1.plot(history.history['val_output_1_loss'], label='validation',color='green')
    ax1.set_title('output_1: x')
    ax1.legend(loc='upper right')

    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(history.history['output_2_loss'], label='train',color='red')
    if history.history.get('val_output_2_loss') is not None:
        ax2.plot(history.history['val_output_2_loss'], label='validation',color='green')
    ax2.set_title('output_2: y')
    ax2.legend(loc='upper right')
    
    if len(save_as_file) > 0:
        fig.savefig(save_as_file)

def plot_fitted_val(args, pred, truth, x_col=None, y_col=None, time_steps=100, save_as_file=True):
    """
    Plot fitted values vs truth - matches original TensorFlow version exactly
    
    Args:
        args: configuration arguments
        pred: tuple of (x_pred, y_pred)
        truth: tuple of (x_truth, y_truth)
        x_col: column index for x data (None for random)
        y_col: column index for y data (None for random)
        time_steps: number of time steps to plot (None for all)
        save_as_file: path to save file (True uses default behavior)
    """
    x_pred, y_pred = pred
    x_truth, y_truth = truth
    
    if x_col is None:
        x_col = np.random.choice(list(range(args.dim_x)))
    if y_col is None:
        y_col = np.random.choice(list(range(args.dim_y)))
    
    if time_steps is not None: ## subset
        ticks = np.arange(time_steps)
        sample_ids = np.random.choice(a=list(range(x_pred.shape[0])),size=time_steps)
        x_pred, x_truth = x_pred[sample_ids], x_truth[sample_ids]
        y_pred, y_truth = y_pred[sample_ids], y_truth[sample_ids]
    else:
        ticks = np.arange(x_pred.shape[0])
    
    if args.Tx > 1:
        num_cols = math.ceil((args.Tx+1)/2.)
        fig, axs = plt.subplots(2, num_cols, figsize=(12,6), constrained_layout=True)
        for step_id in range(args.Tx):
            r, c = step_id//num_cols, step_id%num_cols
            axs[r,c].plot(ticks,x_pred[:, step_id, x_col],label='model',color='red')
            axs[r,c].plot(ticks,x_truth[:, step_id, x_col],label='truth',color='black')
            axs[r,c].legend(loc='lower right')
            axs[r,c].set_title(f'step_id={step_id+1}, x_col={x_col}, model vs. truth')
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
    if len(save_as_file) > 0:
        fig.savefig(save_as_file)
    plt.close()

def evaluate_forecast_results(x_PRED_df, y_PRED_df, output_folder):
    """
    Evaluate forecast results and export to Excel - FIXED VERSION
    
    Args:
        x_PRED_df: DataFrame with HF predictions and targets
        y_PRED_df: DataFrame with LF predictions and targets
        output_folder: output directory path
    """
    from helpers import Evaluator
    
    evaluator = Evaluator()
    vintages = ['F', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']
    
    # Collect errors for each vintage
    x_err_collect = {}
    y_err_collect = {}
    
    for vintage in vintages:
        # X errors (high-frequency)
        x_data = x_PRED_df[x_PRED_df['tag'] == vintage]
        if len(x_data) > 0:
            x_errors = []
            for _, row in x_data.iterrows():
                pred_cols = [col for col in row.index if col.startswith('pred_step_')]
                true_cols = [col for col in row.index if col.startswith('true_step_')]
                
                pred_vals = np.array([row[col] for col in pred_cols])
                true_vals = np.array([row[col] for col in true_cols])
                
                # RMSE by step for this sample
                rmse_by_step = evaluator.rmse_by_step(
                    true_vals.reshape(-1, 1), 
                    pred_vals.reshape(-1, 1)
                )
                x_errors.append(rmse_by_step)
            
            x_err_collect[vintage] = np.array(x_errors)
        
        # Y errors (low-frequency)  
        y_data = y_PRED_df[y_PRED_df['tag'] == vintage]
        if len(y_data) > 0:
            y_errors = []
            for _, row in y_data.iterrows():
                pred_cols = [col for col in row.index if col.startswith('pred_step_')]
                true_cols = [col for col in row.index if col.startswith('true_step_')]
                
                pred_vals = np.array([row[col] for col in pred_cols])
                true_vals = np.array([row[col] for col in true_cols])
                
                rmse_by_step = evaluator.rmse_by_step(
                    true_vals.reshape(-1, 1), 
                    pred_vals.reshape(-1, 1)
                )
                y_errors.append(rmse_by_step)
            
            y_err_collect[vintage] = np.array(y_errors)
    
    # FIXED: Create summary DataFrames using the original TensorFlow pattern
    
    # X error summary
    if x_err_collect:
        # Determine the number of steps from the first vintage's error array
        first_vintage = list(x_err_collect.keys())[0]
        x_num_steps = x_err_collect[first_vintage].shape[1] if len(x_err_collect[first_vintage].shape) > 1 else 1
        x_indices = [f'step_{i+1}' for i in range(x_num_steps)]
        
        # Create separate DataFrames for each metric, then concatenate
        x_err_median = pd.DataFrame(
            data=np.stack([np.median(err, axis=0) for err in x_err_collect.values()], axis=1),
            columns=list(x_err_collect.keys()), 
            index=x_indices
        )
        x_err_median['metric'] = 'median'
        
        x_err_mean = pd.DataFrame(
            data=np.stack([np.mean(err, axis=0) for err in x_err_collect.values()], axis=1),
            columns=list(x_err_collect.keys()), 
            index=x_indices
        )
        x_err_mean['metric'] = 'mean'
        
        x_err_std = pd.DataFrame(
            data=np.stack([np.std(err, axis=0) for err in x_err_collect.values()], axis=1),
            columns=list(x_err_collect.keys()), 
            index=x_indices
        )
        x_err_std['metric'] = 'std'
        
        df_x_err = pd.concat([x_err_median, x_err_mean, x_err_std])
    else:
        df_x_err = pd.DataFrame()
    
    # Y error summary
    if y_err_collect:
        # Determine the number of steps from the first vintage's error array
        first_vintage = list(y_err_collect.keys())[0]
        y_num_steps = y_err_collect[first_vintage].shape[1] if len(y_err_collect[first_vintage].shape) > 1 else 1
        y_indices = [f'step_{i+1}' for i in range(y_num_steps)]
        
        # Create separate DataFrames for each metric, then concatenate
        y_err_median = pd.DataFrame(
            data=np.stack([np.median(err, axis=0) for err in y_err_collect.values()], axis=1),
            columns=list(y_err_collect.keys()), 
            index=y_indices
        )
        y_err_median['metric'] = 'median'
        
        y_err_mean = pd.DataFrame(
            data=np.stack([np.mean(err, axis=0) for err in y_err_collect.values()], axis=1),
            columns=list(y_err_collect.keys()), 
            index=y_indices
        )
        y_err_mean['metric'] = 'mean'
        
        y_err_std = pd.DataFrame(
            data=np.stack([np.std(err, axis=0) for err in y_err_collect.values()], axis=1),
            columns=list(y_err_collect.keys()), 
            index=y_indices
        )
        y_err_std['metric'] = 'std'
        
        df_y_err = pd.concat([y_err_median, y_err_mean, y_err_std])
    else:
        df_y_err = pd.DataFrame()
    
    # Call the existing export function
    export_error_to_excel(df_x_err, df_y_err, x_err_collect, y_err_collect, output_folder)
    
    return df_x_err, df_y_err, x_err_collect, y_err_collect

def export_error_to_excel(df_x_err, df_y_err, x_err_collect, y_err_collect, output_folder):
    """
    Export error analysis to Excel - FIXED VERSION
    
    Args:
        df_x_err: DataFrame with x error summary
        df_y_err: DataFrame with y error summary  
        x_err_collect: dict of x error arrays by experiment tag
        y_err_collect: dict of y error arrays by experiment tag
        output_folder: output directory path
    """
    print('######################################')
    print('## rmse summary for x (step_4):')
    for tag in x_err_collect.keys():
        if len(x_err_collect[tag].shape) > 1 and x_err_collect[tag].shape[1] >= 4:
            try:
                # FIXED: Use the new DataFrame structure where 'metric' is a column
                median_row = df_x_err[(df_x_err.index == 'step_4') & (df_x_err['metric'] == 'median')]
                mean_row = df_x_err[(df_x_err.index == 'step_4') & (df_x_err['metric'] == 'mean')]
                std_row = df_x_err[(df_x_err.index == 'step_4') & (df_x_err['metric'] == 'std')]
                
                if len(median_row) > 0 and tag in median_row.columns:
                    median = median_row[tag].values[0]
                    mean = mean_row[tag].values[0]
                    std = std_row[tag].values[0]
                    print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
                else:
                    print(f'#    tag = {tag}; step_4 data not available')
            except (KeyError, IndexError) as e:
                print(f'#    tag = {tag}; step_4 data not available - {e}')
    
    print('## rmse summary for y (step_1):')
    for tag in y_err_collect.keys():
        if len(y_err_collect[tag].shape) > 1 and y_err_collect[tag].shape[1] >= 1:
            try:
                # FIXED: Use the new DataFrame structure where 'metric' is a column
                median_row = df_y_err[(df_y_err.index == 'step_1') & (df_y_err['metric'] == 'median')]
                mean_row = df_y_err[(df_y_err.index == 'step_1') & (df_y_err['metric'] == 'mean')]
                std_row = df_y_err[(df_y_err.index == 'step_1') & (df_y_err['metric'] == 'std')]
                
                if len(median_row) > 0 and tag in median_row.columns:
                    median = median_row[tag].values[0]
                    mean = mean_row[tag].values[0]
                    std = std_row[tag].values[0]
                    print(f'#    tag = {tag}; median = {median:.2f}, mean = {mean:.2f}, std = {std:.3f}')
                else:
                    print(f'#    tag = {tag}; step_1 data not available')
            except (KeyError, IndexError) as e:
                print(f'#    tag = {tag}; step_1 data not available - {e}')
    print('######################################')
    
    print(f'[exporting error summary to excel] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    with pd.ExcelWriter(f'{output_folder}/forecast_err.xlsx') as writer:
        df_x_err.to_excel(writer, sheet_name=f'summary_x_err', index=True)
        df_y_err.to_excel(writer, sheet_name=f'summary_y_err', index=True)
        
        for experiment_tag, arr in x_err_collect.items():
            df = pd.DataFrame(data=arr, columns=[f'step{i+1}' for i in range(arr.shape[1])], index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_excel(writer, sheet_name=f'x_{experiment_tag}', index=True)
        for experiment_tag, arr in y_err_collect.items():
            df = pd.DataFrame(data=arr, columns=[f'step{i+1}' for i in range(arr.shape[1])], index=range(arr.shape[0]))
            df.index.name = 'experiment_id'
            df.to_excel(writer, sheet_name=f'y_{experiment_tag}', index=True)