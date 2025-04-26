# implementation from
# https://github.com/myscience/mamba/blob/main
# which uses Gu & Dao (2023) Mamba model

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einops import rearrange

from torch import Tensor
from typing import TypeVar, Tuple
from torch.nn.functional import silu
from torch.nn.functional import softplus

from helpers.RMSNorm import RMSNorm
from helpers.pscan import pscan

T = TypeVar('T')
D = TypeVar('D')
Cache = Tuple[Tensor, Tensor] | None
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var


class Mamba(nn.Module):
    '''
    State-space sequence-model backbone from Gu & Dao (2023).

    Args:
        num_layers (int):
            Number of alternating MambaBlocks + RMSNorm layers.
        d_input (int):
            Dimension of the input embeddings (and output of each block).
        d_model (int):
            Internal state dimension inside each MambaBlock.
        d_state (int, optional):
            Dimension of the continuous state in the SSM stage.
            Defaults to 16.
        d_discr (int | None, optional):
            Rank of the discrete gating subspace (low-rank for D).
            If None, set to d_model // 16. Defaults to None.
        ker_size (int, optional):
            Kernel size for the depth-wise convolution. Defaults to 4.
        parallel (bool, optional):
            If True, use a parallel scan for SSM (“pscan”) rather than
            a Python loop. Defaults to False.
    '''
    
    def __init__(
        self,
        num_layers : int,
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
    ) -> None:
        super().__init__()
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.layers = nn.ModuleList([
            nn.ModuleList(
                [
                    MambaBlock(d_input, d_model, d_state, d_discr, ker_size, parallel),
                    RMSNorm(d_input)
                ]
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, seq : Tensor, cache : Cache = None) -> Tuple[Tensor, Cache]:
        '''
        Inputs and outputs a tensor of shape (batch_size, seq_len, d_seq).
        '''
        
        for mamba, norm in self.layers: # type: ignore
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        return seq, cache
        
class MambaBlock(nn.Module):
    '''
    Class representing the MambaBlock as introduced in Gu & Dao (2023).
    '''
    def __init__(
        self, 
        d_input : int,
        d_model : int,
        d_state : int = 16,
        d_discr : int | None = None,
        ker_size : int = 4,
        parallel : bool = False,
    ) -> None:
        super().__init__()
        
        d_discr = default(d_discr, d_model // 16)
        
        # Projection matrices from the input sequence space to the
        # model state space (of dimension d_model) and back.
        # NOTE: The in_proj matrix has a factor of 2 because it is
        #       used to split the input sequence into two branches
        self.in_proj  = nn.Linear(d_input, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_input, bias=False)
        
        # Projection matrices for endowing the SSM stage with
        # context-dependent capability (i.e. input dependence)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(d_model, d_discr, bias=False), # Fixing matrix rank to d_disc
            nn.Linear(d_discr, d_model, bias=False),
        )
        
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=ker_size,
            padding=ker_size - 1,
            groups=d_model,
            bias=True,
        )
        
        # Parameters for the SSM. Follows the S4 initialization
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model, dtype=torch.float))
        
        # Whether to use or not the parallel scan for the SSM
        self.parallel = parallel
        
    def forward(self, seq : Tensor, cache : Cache = None) -> Tuple[Tensor, Cache]:
        '''
        Forward pass of the MambaBlock.
        Inputs and outputs a tensor of shape (batch_size, seq_len, d_model).
        '''
        b, l, d = seq.shape
        
        (prev_hid, prev_inp) = default(cache, (None, None))
        
        # Project the input sequence from d_seq to d_model and into two
        # distinct branches, one for the SSM and the residual branch
        # (see Fig. 3 of the Mamba paper). The resulting shapes are:
        # a: (batch_size, seq_len, d_model), b: (batch_size, seq_len, d_model)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        
        # * The SSM branch
        # Apply the convolutional layer to the SSM branch
        # NOTE: We need to move the channel dimension to the second dimension
        #       for the convolution to work properly, hence the rearrange
        x = rearrange(a, 'b l d -> b d l')

        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l] # Crop the output to the original length
        a = rearrange(a, 'b d l -> b l d')
        
        # Apply the SSM
        a = silu(a)
        a, hid = self.ssm(a, prev_hid=prev_hid) 
        
        # * The residual branch
        b = silu(b)
        
        # Combine the two branches
        out = a * b
        out =  self.out_proj(out)
        
        # Update the cache for next call if provided
        if cache:
            # Drop the first element of the hidden input states and attach
            # the newly computed results from the convolutions
            cache = (hid.squeeze(), x[..., 1:]) # type: ignore
        
        return out, cache
    
    def ssm(self, seq : Tensor, prev_hid : Tensor | None) -> Tuple[Tensor, Tensor]:
        '''
        State Space Model (SSM) of the MambaBlock.
        
        Args:
            seq (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        '''
        
        # Compute the context-dependent projections
        A = -self.A # shape: (d_model, d_state)
        D = +self.D # shape: (d_model, )
        
        B = self.s_B(seq)               # shape: (batch_size, seq_len, d_state)
        C = self.s_C(seq)               # shape: (batch_size, seq_len, d_state)
        Δ = softplus(D + self.s_D(seq)) # shape: (batch_size, seq_len, d_model)
        
        # Discretize the A and B parameters using Δ
        A_bar = einsum(torch.exp(A), Δ, 'd s,   b l d -> b l d s')
        B_bar = einsum(          B,  Δ, 'b l s, b l d -> b l d s')
        
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        
        # Compute the state space hidden states
        # NOTE: This can be done either sequentially (slow) or with
        # a parallel scan (fast)
        hid = self._hid_states(
            A_bar,
            X_bar,
            parallel=self.parallel,
            prev_hid=prev_hid,    
        )
        
        # Compute the output based on the hidden states
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
    
        out = out + D * seq
        
        return out, hid
    
    def _hid_states(
        self,
        A : Tensor,
        X : Tensor,
        parallel : bool = False,
        prev_hid : Tensor | None = None,
    ) -> Tensor:
        '''
        Calculate the hidden states of the SSM.

        Args:
            A (Tensor): The tensor representing A_bar.
            X (Tensor): The tensor representing X.
            parallel (bool): Whether to use parallel scan or 
                sequential computation (slower).

        Returns:
            Tensor: The tensor representing the hidden states.
        '''
        b, l, d, s = A.shape
        
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        
        if prev_hid is not None:
            # If we have a previous hidden state it means we are running the
            # efficient auto-regressive inference, so we expect both A and X
            # to have a trivial length of 1, we just drop it when returning
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
        
        h = None if parallel else torch.zeros(b, d, s, device=self.device)
        
        return pscan(A, X) if parallel else torch.stack([
            h := A_t * h + X_t
            for A_t, X_t in zip(A, X)
        ], dim=1)

    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        '''
        return next(self.parameters()).device