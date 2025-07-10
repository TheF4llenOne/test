"""
Multi-task Mixed Frequency Model based on a seq2seq architecture
(c) 2023 Jiahe Lin & George Michailidis
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
        """
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        if self.bidirectional_encoder:
            self.lstm = nn.LSTM(input_size=dim_x, hidden_size=n_a, batch_first=True, 
                               bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=dim_x, hidden_size=n_a, batch_first=True)
            
    def forward(self, x):
        """
        forward pass for the encoder
        Argv:
            x: (tensor) shape (batch_size, Lx, dim_x)
        Return:
            a: (tensor) sequence of LSTM hidden states, (batch_size, Lx, n_a)
        """
        a, _ = self.lstm(x)
        return a

class OneStepAttn(nn.Module):
    """Attention alignment module"""
    def __init__(self, input_size, n_align):
        """
        input_size: (int) size of concatenated input (n_a + n_s)
        n_align: (int) hidden unit of the alignment model
        """
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(input_size, n_align)
        self.densor2 = nn.Linear(n_align, 1)
    
    def _softmax(self, x, dim=1):
        """
        Customized softmax function that is suitable for getting attention
        Argv:
            x : Tensor.
            dim: Integer, dimension along which the softmax normalization is applied.
        Returns
            Tensor, output of softmax transformation.
        """
        if x.dim() == 2: 
            return F.softmax(x, dim=dim)
        elif x.dim() > 2:
            # Match TensorFlow's manual implementation exactly
            e = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
            s = torch.sum(e, dim=dim, keepdim=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')
    
    def forward(self, attn_input):
        """
        forward pass: performs one step customized attention that outputs a context vector computed as a dot product of the attention weights "alphas" and the hidden states "a" from the LSTM encoder.
        Argv:
            a: hidden state from the pre-attention LSTM, shape = (m, *, n_a)
            s_prev: previous hidden state of the (post-attn) LSTM, shape = (m, n_s)
        Returns:
            context: context vector, input of the next (post-attention) LSTM cell
        """
        a, s_prev = attn_input
        # Repeat s_prev to match sequence length of a
        s_prev = s_prev.unsqueeze(1).repeat(1, a.shape[1], 1)  # (m, a.shape[1], n_s)
        concat = torch.cat([a, s_prev], dim=-1)
        e = torch.tanh(self.densor1(concat))
        energies = F.relu(self.densor2(e))
        alphas = self._softmax(energies, dim=1)
        context = torch.sum(alphas * a, dim=1, keepdim=True)
        return context

class MTMFSeq2Seq(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        Lx,
        Tx,
        Ty,
        n_a,
        n_s,
        n_align_x,
        n_align_y,
        fc_x,
        fc_y,
        dropout_rate,
        freq_ratio=3,
        bidirectional_encoder=False,
        l1reg=1e-5,
        l2reg=1e-4
    ):
        """
        dim_x: (int) dimension of the (encoder) input high-frequency sequence
        dim_y: (int) dimension of the (decoder) input/output low-frequency sequence
        Lx: (int) length of the input high-frequency sequence (encoder)
        Tx: (int) length of the target high-frequency sequence (decoder)
        Ty: (int) length of the output low-frequency sequence (decoder)
        n_a: (int) hidden state dimension of the pre-attention/post-atten LSTM for x
        n_s: (int) hidden state dimension of the post-attention LSTM
        n_align_{x,y}: (int) hidden state dimension of the dense layer in the alignment model
        fc_{x,y}: (int) hidden state dimension of the dense layer before the dropout layer
        dropout_rate: (float) dropout rate for the layer before output
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
        
        ## for the encoder
        self.pre_attn = PreAttnEncoder(dim_x, n_a, dropout_rate=dropout_rate, 
                                      bidirectional_encoder=bidirectional_encoder)
        ## for the attention alignment model
        self.one_step_attention_x = OneStepAttn(n_a + n_a)  # concat size
        self.one_step_attention_y = OneStepAttn(n_a + n_s)  # concat size
        
        ## for the x decoder
        self.post_attn_x = nn.LSTMCell(n_a + dim_x, n_a)
        self.ffn1_x = nn.Linear(n_a, fc_x)
        self.dropout_fn_x = nn.Dropout(dropout_rate)
        self.ffn2_x = nn.Linear(fc_x, dim_x)
        
        ## for the y decoder
        self.post_attn_y = nn.LSTMCell(n_a + dim_y, n_s)
        self.ffn1_y = nn.Linear(n_s, fc_y)
        self.dropout_fn_y = nn.Dropout(dropout_rate)
        self.ffn2_y = nn.Linear(fc_y, dim_y)
        
    def initialize_state(self, batch_size, dim, device):
        if batch_size is not None:
            return torch.zeros(batch_size, dim, device=device)
        else:
            # Create empty tensor that can be resized later
            return torch.empty(0, dim, device=device)
    
    def forward(self, batch_inputs, training=False):
        
        x_encoder_in, x_decoder_in, y_decoder_in = batch_inputs
        batch_size = x_encoder_in.shape[0]
        device = x_encoder_in.device
        
        # Set training mode for dropout layers
        if training:
            self.train()
        else:
            self.eval()
        
        ##############################################
        ## stage 1: pre-attn encoding
        ##############################################
        a = self.pre_attn(x_encoder_in)
        
        ##############################################
        ## stage 2.1: x -> y decoding
        ##############################################
        s_y = self.initialize_state(batch_size, self.n_s, device)
        c_y = self.initialize_state(batch_size, self.n_s, device)
        
        for t in range(self.Ty):
            a_idx = int((t+1)*self.freq_ratio-1)
            a_to_attend = a[:,(a_idx-self.freq_ratio+1):(a_idx+1),:]
            context = self.one_step_attention_y([a_to_attend, s_y])
            # Match TF's expand_dims behavior
            post_attn_input = torch.cat([context, y_decoder_in[:,t,:].unsqueeze(1)], dim=-1)
            s_y, c_y = self.post_attn_y(post_attn_input.squeeze(1), (s_y, c_y))
        
        y_pred = self.ffn1_y(s_y)
        y_pred = self.dropout_fn_y(y_pred)
        y_pred = self.ffn2_y(y_pred)
        
        ##############################################
        ## stage 2.2: x -> x decoding
        ##############################################
        # For bidirectional, we only take the forward direction's hidden size
        s_x = a[:,-1,:self.n_a] 
        c_x = self.initialize_state(batch_size, self.n_a, device)
        
        x_pred_by_step = []
        for t in range(x_decoder_in.shape[1]):
            context = self.one_step_attention_x([a, s_x])
            # Match TF's expand_dims behavior
            post_attn_input = torch.cat([context, x_decoder_in[:,t,:].unsqueeze(1)], dim=-1)
            s_x, c_x = self.post_attn_x(post_attn_input.squeeze(1), (s_x, c_x))
            x_pred_curr = self.ffn1_x(s_x)
            x_pred_curr = self.dropout_fn_x(x_pred_curr)
            x_pred_curr = self.ffn2_x(x_pred_curr)
            x_pred_by_step.append(x_pred_curr)
        x_pred = torch.stack(x_pred_by_step, dim=1)
        
        return [x_pred, y_pred]

class MTMFSeq2SeqPred(_baseSeqPred):
    
    def __init__(
        self,
        model,
        scaler_x,
        scaler_y,
        apply_inv_scaler=True
    ):
        super().__init__(model, scaler_x, scaler_y, apply_inv_scaler)