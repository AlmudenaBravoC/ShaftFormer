from typing import Optional, Any, Union, Callable
import copy

import torch
from torch import Tensor
from torch.nn import Dropout, ModuleList, Module
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
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
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
        Attn1 = ProbAttention 
        Attn2 = FullAttention
        self.attn = AttentionLayer(Attn1(True, factor, attention_dropout=dropout, output_attention=output_attention, **factory_kwargs), 
                                d_model, nhead, mix=False, **factory_kwargs)
        self.cross = AttentionLayer(Attn2(False, factor, attention_dropout=dropout, output_attention=output_attention, **factory_kwargs), 
                                d_model, nhead, mix=False, **factory_kwargs)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*4, kernel_size=1, **factory_kwargs)
        self.conv2 = nn.Conv1d(in_channels=d_model*4, out_channels=d_model, kernel_size=1, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = False,
        memory_is_causal: bool = False,) -> Tensor:
        x = x + self.dropout(self.attn(
            x, x, x,
            attn_mask=tgt_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross(
            x, memory, memory,
            attn_mask=tgt_key_padding_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)





#%% OTHER FUNCTIONS

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class MLP_simple(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=70, hidden_dim1=50, hidden_dim2=20, output_dim=1):
        super(MLP_simple, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.h1 = nn.Linear(hidden_dim, hidden_dim1)
        self.h2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Permute the input tensor to shape [b, s, d]
        x = x.permute(1, 0, 2)
        
        # Flatten the sequence dimension
        x = x.reshape(x.shape[0], -1)
        
        # Apply the MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.h2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x