from typing import Optional, Any, Union, Callable
import copy

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, MultiheadAttention, Linear, ModuleList, Module
from torch.nn import functional as F
import torch.nn as nn

#attention layer
from model.attn import ProbAttention, FullAttention, AttentionLayer

#%% ENCODER AND DECODER

class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
    
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask

        if (src_key_padding_mask is not None):
            convert_to_nested = True
            output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
            src_key_padding_mask_for_layers = None

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

#%% LAYERS
class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, 
                 attn='prob', factor =5, output_attention = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()

        # Implementation of the Conv layer - INFORMER 2020
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=d_model,
                                  out_channels=d_model,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular', **factory_kwargs)
        self.norm = nn.BatchNorm1d(d_model, **factory_kwargs)
        self.activation2 = nn.ELU()
        self.dropout = Dropout(dropout)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) #reduce the dimension to the half
        self.downConv2 = nn.Conv1d(in_channels=d_model,
                                  out_channels=d_model,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular', **factory_kwargs)
        self.upConv = nn.Conv1d(in_channels=int(d_model/2),
                                out_channels=d_model, kernel_size=3,padding=padding, padding_mode='zeros', **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation


        ## NEW ATTENTION LAYER -- based on the informer
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.attn = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention, **factory_kwargs), 
                                d_model, nhead, mix=False, **factory_kwargs)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*4, kernel_size=1, **factory_kwargs)
        self.conv2 = nn.Conv1d(in_channels=d_model*4, out_channels=d_model, kernel_size=1, **factory_kwargs)
        self.norm_enc1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm_enc2 = nn.LayerNorm(d_model, **factory_kwargs)
        


    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        try:
            x, attn = self.sa_informer(x)
        except:
            x = self.sa_informer(x)
            
        x = self.conv_informer(x)

        return x
    
    #self attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def sa_informer(self, x:Tensor):
        try :
            x_n, attn = self.attn(x, x, x, attn_mask=None)
        except:
            x_n = self.attn(x, x, x, attn_mask=None)

        x = x + self.dropout(x_n)
        y = x = self.norm_enc1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm_enc2(x+y)


    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def conv_informer(self, x:Tensor):
        x = self.downConv(x.permute(0,2,1)) # --> [len, dim, batch]
        x = self.norm(x)
        x = self.activation2(x)

        #new 
        # x = self.dropout(x)
        # x = self.downConv2(x)
        # # x = self.maxPool(x.permute(1,0,2)) #probar
        # x = self.maxPool(x.permute(0,2,1)) #change 256 to 128 ( the half ) --> [batch, seq, dim] we want to reduce the dimension of the features
        # x = self.upConv(x.permute(0,2,1)) #change again the distribution --> [batch, dim, seq] we want to upsize the dimension to its original
        # x = x.permute(0,2,1)

        #remove        
        x = self.maxPool(x.permute(1,2,0)) #--> [dim, batch, len]
        x = x.permute(2,1,0) # esto modifica el de la secuencia
        # x = x.transpose(1,2)

        return x


class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None,
                 attn='prob', factor =5, output_attention = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        

        ## NEW ATTENTION LAYER -- based on the informer
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.attn = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention, **factory_kwargs), 
                                d_model, nhead, mix=False, **factory_kwargs)
        self.attn2 = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False, **factory_kwargs),
                                d_model, nhead, mix=False, **factory_kwargs)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*4, kernel_size=1, **factory_kwargs)
        self.conv2 = nn.Conv1d(in_channels=d_model*4, out_channels=d_model, kernel_size=1, **factory_kwargs)
        self.norm_enc1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm_enc2 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm_enc3 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout = Dropout(dropout)
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt

        x = self.sa_informer(x, memory)

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def sa_informer(self, x:Tensor, x_enc:Tensor):
        try :
            x_n, attn = self.attn(x, x, x, attn_mask=None)
        except:
            x_n = self.attn(x, x, x, attn_mask=None)
        x = x + self.dropout(x_n)
        x = self.norm_enc1(x)

        #add the full attention layer
        try:
            x, attn = self.attn2(x, x_enc, x_enc, attn_mask = None)
        except:
            x = self.attn2(x, x_enc, x_enc, attn_mask = None)

        y = x = self.norm_enc2(self.dropout(x))
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x = self.norm_enc3(x+y)

        return x

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout3(x)
    


#%% OTHER FUNCTIONS

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu