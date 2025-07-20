"""
Multi-task Mixed Frequency Model based on a seq2seq architecture - PyTorch Version
(c) 2023 Jiahe Lin & George Michailidis
Converted to PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import _baseSeqPred

class PreAttnEncoder(nn.Module):
    """Pre-attention Encoder module"""
    def __init__(self, dim_x, n_a, dropout_rate=0.2, bidirectional_encoder=False):
        """
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        n_a: (int) hidden state dimension of the pre-attention LSTM
        dropout_rate: (float) dropout rate
        bidirectional_encoder: (bool) whether to use bidirectional LSTM
        """
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        
        if self.bidirectional_encoder:
            self.lstm = nn.LSTM(input_size=dim_x, hidden_size=n_a, 
                               batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=dim_x, hidden_size=n_a, 
                               batch_first=True, bidirectional=False)
            
    def forward(self, x):
        """
        Forward pass for the encoder
        Args:
            x: (tensor) shape (batch_size, Lx, dim_x)
        Return:
            a: (tensor) sequence of LSTM hidden states, (batch_size, Lx, n_a) or (batch_size, Lx, 2*n_a) if bidirectional
        """
        # LSTM returns (output, (h_n, c_n))
        # output shape: (batch_size, seq_len, hidden_size * num_directions)
        a, _ = self.lstm(x)
        return a

class OneStepAttn(nn.Module):
    """Attention alignment module"""
    def __init__(self, input_dim, n_align):
        """
        Args:
            input_dim: (int) input dimension (encoder_dim + decoder_dim)
            n_align: (int) hidden unit of the alignment model
        """
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(input_dim, n_align)  # Now with correct input size
        self.densor2 = nn.Linear(n_align, 1)
        # Note: activations (tanh, relu, softmax) are applied in forward() method
        
    def forward(self, attn_input):
        """
        Forward pass: performs one step customized attention that outputs a context vector 
        computed as a dot product of the attention weights "alphas" and the hidden states "a" 
        from the LSTM encoder.
        
        Args:
            attn_input: list of [a, s_prev] where:
                a: (tensor) LSTM hidden states, shape (batch_size, Tx, n_a)
                s_prev: (tensor) previous decoder hidden state, shape (batch_size, n_s)
        
        Returns:
            context: (tensor) context vector, shape (batch_size, 1, n_a)
        """
        a, s_prev = attn_input
        
        # Repeat s_prev to match the sequence length of a
        # s_prev: (batch_size, n_s) -> (batch_size, Tx, n_s)
        s_prev_repeated = s_prev.unsqueeze(1).repeat(1, a.shape[1], 1)
        
        # Concatenate a and s_prev_repeated along the last dimension
        # concat_input: (batch_size, Tx, n_a + n_s)
        concat_input = torch.cat([a, s_prev_repeated], dim=-1)
        
        # First layer with tanh activation
        e_intermediate = torch.tanh(self.densor1(concat_input))
        
        # Second layer with relu activation
        e = F.relu(self.densor2(e_intermediate))  # Shape: (batch_size, Tx, 1)
        
        # Apply softmax to get attention weights - using PyTorch's optimized implementation
        alphas = F.softmax(e, dim=1)  # Normalize over sequence dimension (Tx)
        
        # Compute context vector as weighted sum of encoder hidden states
        # context: (batch_size, 1, n_a)
        context = torch.sum(alphas * a, dim=1, keepdim=True)
        
        return context

class MTMFSeq2Seq(nn.Module):
    """Multi-task Mixed Frequency Seq2Seq Model"""
    def __init__(self, dim_x, dim_y, Lx, Tx, Ty, n_a, n_s, n_align_x, n_align_y, 
                 fc_x, fc_y, dropout_rate=0.2, l1reg=1e-4, l2reg=1e-4, 
                 freq_ratio=6, bidirectional_encoder=False):
        """
        Args:
            dim_x: (int) dimension of high-frequency input
            dim_y: (int) dimension of low-frequency input  
            Lx: (int) length of encoder sequence
            Tx: (int) length of high-frequency decoder sequence
            Ty: (int) length of low-frequency decoder sequence
            n_a: (int) hidden state dimension of pre-attention LSTM encoder
            n_s: (int) hidden state dimension of post-attention LSTM decoder
            n_align_x: (int) hidden unit of alignment model for x
            n_align_y: (int) hidden unit of alignment model for y
            fc_x: (int) hidden state dimension of dense layer before dropout for x
            fc_y: (int) hidden state dimension of dense layer before dropout for y
            dropout_rate: (float) dropout rate for the layer before output
            l1reg: (float) L1 regularization coefficient
            l2reg: (float) L2 regularization coefficient
            freq_ratio: (int) frequency ratio
            bidirectional_encoder: (bool) whether the encoder should be bidirectional
        """
        super(MTMFSeq2Seq, self).__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.Lx = Lx
        self.Tx = Tx
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.n_align_x = n_align_x
        self.n_align_y = n_align_y
        self.fc_x = fc_x
        self.fc_y = fc_y
        self.freq_ratio = freq_ratio
        self.bidirectional_encoder = bidirectional_encoder
        self.l1reg = l1reg
        self.l2reg = l2reg
        
        # Pre-attention encoder
        self.pre_attn = PreAttnEncoder(dim_x, n_a, dropout_rate=dropout_rate, 
                                      bidirectional_encoder=bidirectional_encoder)
        
        # Attention mechanisms
        # Input size for attention is encoder_output + decoder_hidden_state
        encoder_output_dim = 2 * n_a if bidirectional_encoder else n_a
        
        self.one_step_attention_x = OneStepAttn(encoder_output_dim + n_a, n_align_x)
        self.one_step_attention_y = OneStepAttn(encoder_output_dim + n_s, n_align_y)
        
        # X decoder (high-frequency)
        self.post_attn_x = nn.LSTMCell(encoder_output_dim + dim_x, n_a)
        self.ffn1_x = nn.Linear(n_a, fc_x)
        self.dropout_fn_x = nn.Dropout(dropout_rate)
        self.ffn2_x = nn.Linear(fc_x, dim_x)
        
        # Y decoder (low-frequency)  
        self.post_attn_y = nn.LSTMCell(encoder_output_dim + dim_y, n_s)
        self.ffn1_y = nn.Linear(n_s, fc_y)
        self.dropout_fn_y = nn.Dropout(dropout_rate)
        self.ffn2_y = nn.Linear(fc_y, dim_y)
        
        # Apply L1/L2 regularization to output layers if needed
        # Note: PyTorch handles this differently - typically done in optimizer or loss function
        
    def initialize_state(self, batch_size, dim, device):
        """Initialize hidden states for LSTM cells"""
        if batch_size is not None:
            return torch.zeros(batch_size, dim, device=device)
        else:
            return torch.empty(0, dim, device=device)
    
    def forward(self, batch_inputs):
        """
        Forward pass of the MTMFSeq2Seq model
        
        Args:
            batch_inputs: list of [x_encoder_in, x_decoder_in, y_decoder_in]
                x_encoder_in: (tensor) shape (batch_size, Lx, dim_x)
                x_decoder_in: (tensor) shape (batch_size, Tx, dim_x)  
                y_decoder_in: (tensor) shape (batch_size, Ty, dim_y)
                
        Returns:
            [x_pred, y_pred]: list of predictions
                x_pred: (tensor) shape (batch_size, Tx, dim_x)
                y_pred: (tensor) shape (batch_size, dim_y)
        """
        x_encoder_in, x_decoder_in, y_decoder_in = batch_inputs
        batch_size = x_encoder_in.shape[0]
        device = x_encoder_in.device
        
        ##############################################
        ## Stage 1: Pre-attention encoding
        ##############################################
        a = self.pre_attn(x_encoder_in)  # Shape: (batch_size, Lx, n_a) or (batch_size, Lx, 2*n_a)
        
        ##############################################
        ## Stage 2.1: x -> y decoding (low-frequency)
        ##############################################
        s_y = self.initialize_state(batch_size, self.n_s, device)
        c_y = self.initialize_state(batch_size, self.n_s, device)
        
        for t in range(self.Ty):
            # Select attention window based on frequency ratio
            a_idx = int((t + 1) * self.freq_ratio - 1)
            a_to_attend = a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
            
            # Compute attention context
            context = self.one_step_attention_y([a_to_attend, s_y])
            
            # Concatenate context and decoder input
            post_attn_input = torch.cat([context, y_decoder_in[:, t, :].unsqueeze(1)], dim=-1)
            
            # LSTM cell forward pass
            s_y, c_y = self.post_attn_y(post_attn_input.squeeze(1), (s_y, c_y))
        
        # Final y prediction
        y_pred = self.ffn1_y(s_y)
        y_pred = self.dropout_fn_y(y_pred)
        y_pred = self.ffn2_y(y_pred)
        
        ##############################################
        ## Stage 2.2: x -> x decoding (high-frequency)
        ##############################################
        # Initialize with last encoder hidden state (take only first n_a dimensions if bidirectional)
        s_x = a[:, -1, :self.n_a]
        c_x = self.initialize_state(batch_size, self.n_a, device)
        
        x_pred_by_step = []
        for t in range(x_decoder_in.shape[1]):
            # Compute attention context
            context = self.one_step_attention_x([a, s_x])
            
            # Concatenate context and decoder input
            post_attn_input = torch.cat([context, x_decoder_in[:, t, :].unsqueeze(1)], dim=-1)
            
            # LSTM cell forward pass
            s_x, c_x = self.post_attn_x(post_attn_input.squeeze(1), (s_x, c_x))
            
            # Prediction for this timestep
            x_pred_curr = self.ffn1_x(s_x)
            x_pred_curr = self.dropout_fn_x(x_pred_curr)
            x_pred_curr = self.ffn2_x(x_pred_curr)
            x_pred_by_step.append(x_pred_curr)
            
        # Stack all timestep predictions
        x_pred = torch.stack(x_pred_by_step, dim=1)  # Shape: (batch_size, Tx, dim_x)
        
        return [x_pred, y_pred]

class MTMFSeq2SeqPred(_baseSeqPred):
    """Prediction wrapper for MTMFSeq2Seq model"""
    
    def __init__(self, model, scaler_x, scaler_y, apply_inv_scaler=True):
        """
        Initialize the prediction wrapper
        
        Args:
            model: MTMFSeq2Seq model instance
            scaler_x: scaler for high-frequency data
            scaler_y: scaler for low-frequency data  
            apply_inv_scaler: whether to apply inverse scaling to predictions
        """
        super().__init__(model, scaler_x, scaler_y, apply_inv_scaler)