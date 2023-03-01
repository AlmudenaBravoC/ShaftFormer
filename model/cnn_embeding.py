import torch
import torch.nn.functional as F
import torch.nn as nn
import math

#based on the transformer_time_series (https://github.com/mlpotter/Transformer_Time_Series/blob/master/Transformer_Decoder_nologsparse.ipynb)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x)
        return torch.tanh(x)

class PositionalEmbedding(nn.Module): #basado en: informer2020 --> (https://github.com/zhouhaoyi/Informer2020/blob/main/models/embed.py) 
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionalEncoding(nn.Module): # basado en attention transformer --> https://github.com/vatsalsaglani/Attention-Transformer-/blob/Attention_main/package/AttentionTransformer/AttentionTransformer/PositionalEncoding.py

    def __init__(self, dim_hid, num_pos):

        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(num_pos, dim_hid))

    def _get_sinusoid_encoding_table(self, num_pos, dim_hid):

        def get_position_angle_vec(position):

            return [
                position / np.power(10000, 2 * (hid_j // 2) / dim_hid) for hid_j in range(dim_hid)
            ]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_pos)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def custom_embedding(args, x): 
    """
        INPUT EMBEDDING
    in_channels: the number of features per time point
    out_channels: the number of features outputted per time point
    kernel_size: k is the width of the 1-D sliding kernel
    """
    input_embedding = context_embedding(in_channels = args.inchannels, embedding_size = args.outchannels, k = args.kernel ) #output = [sequence len, embedding size, batch]
    positional_embedding = torch.nn.Embedding(args.inchannels*args.outchannels, args.outchannels)
    # positional_encoding = PositionalEncoding(args.outchannels, )


    z = x.unsqueeze(1)

    #we want the output to be --> [sequence len, batch, embedding size]
    z_embedding = input_embedding(z).permute(0,2,1)
    if torch.max(x.type(torch.long)) >= args.inchannels*args.outchannels:
        raise ValueError(f"Index out of range: change values of the Embedding:: input {torch.max(x.type(torch.long))} >= weights {args.inchannels*args.outchannels}")

    # positional_embeddings = positional_embedding(x.type(torch.long)) #.permute(1,0,2)
    # print(positional_embeddings)

    # input_embedding = z_embedding+positional_embeddings

    # return input_embedding
    return z_embedding