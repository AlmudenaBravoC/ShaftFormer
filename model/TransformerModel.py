
import torch
import torch.nn as nn
from torch.nn import Transformer
from utils.myTools import dotdict
import numpy as np

from model.cnn_embeding import context_embedding, PositionalEmbedding
from model.model_layers import TransformerEncoderLayer, TransformerEncoder


class ShaftFormer(nn.Module):
    def __init__(self, args:dotdict, device) -> None:
        super().__init__()
        self.args = args
        self.device = device

        ##EMBEDDING
        if args.conf_cnn:
            self.conv1 = context_embedding(in_channels = self.args.inchannels, embedding_size = self.args.outchannels-self.args.outchannels_conf, k = self.args.kernel ) 
            self.conv2 = context_embedding(in_channels = self.args.inchannels_conf, embedding_size = self.args.outchannels_conf, k = self.args.kernel ) #output = [1, embedding size, batch]
        else:
            self.conv1 = context_embedding(in_channels=self.args.inchannels, embedding_size=self.args.outchannels, k=self.args.kernel)
        self.position_embedding = PositionalEmbedding(d_model=self.args.outchannels)
        
        ## ENCODER
        encoder_layer = TransformerEncoderLayer(d_model= self.args.outchannels, nhead= self.args.heads , dropout=self.args.dropout, device= device)
        encoder = TransformerEncoder(encoder_layer, num_layers=self.args.nencoder)


        ## MODEL
        if self.args.model_type == "forecasting":
            print("model for forecasting")
            
            ## DECODER
            if args.two_linear and args.conf_cnn:
                self.linear = torch.nn.Linear(args.outchannels, self.args.outchannels-self.args.outchannels_conf)
                self.linear2 = torch.nn.Linear(self.args.outchannels-self.args.outchannels_conf, args.inchannels)
            else:
                self.linear = torch.nn.Linear(args.outchannels, args.inchannels)
            
            self.deconv = context_embedding(in_channels=self.args.outchannels, embedding_size=self.args.inchannels, k=self.args.kernel)

            if args.linear_initialization == 'Xavier':
                nn.init.xavier_uniform_(self.linear.weight)
            elif args.linear_initialization == 'He':
                nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            elif args.linear_initialization == 'Uniform':
                nn.init.uniform_(self.linear.weight, a=args.a, b=args.b)

            self.model = Transformer(d_model = self.args.outchannels, nhead=self.args.heads, custom_encoder=encoder, device=device, norm_first=True) #d_model must be divisible by nhead and d_model should be the same as the number of features of the data
            if self.args.use_multi_gpu and self.args.use_gpu:
                print('\t Parallelization of the model')
                self.model = nn.DataParallel(self.model.cpu(), device_ids=self.args.device_ids, dim=1) #dim = 1 that is where the signal is --> [len, batch, dim]
                self.model = self.model.to(self.device)
            
            if self.args.use_gpu:
                self.linear.to(device)
                if self.args.two_linear: self.linear2.to(device)
                self.conv1.to(device)
                if self.args.conf_cnn: self.conv2.to(device)
                self.deconv.to(device)

        else:
            print("model for classification")
            decoder = nn.Linear(1, 1) #we are not going to use this
            self.simple_mlp = MLP_simple(input_dim= self.args.outchannels, hidden_dim=64, output_dim=self.args.inchannels)

            transformer = Transformer(d_model = self.args.outchannels, nhead=self.args.heads, custom_encoder=encoder, custom_decoder=decoder, device=device, norm_first=True) #d_model must be divisible by nhead and d_model should be the same as the number of features of the data
            self.encoder_transformer = transformer.encoder()

            raise Exception
    
    
    def forward(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False):
        return self._process_one_batch(x, feat, target_len, test)
    
    def _process_one_batch(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False):

        if test and self.args.model_type == "forecasting": 
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.linear.to(self.device)
            if self.args.two_linear: self.linear2.to(self.device)
            self.conv1.to(self.device)
            if self.args.conf_cnn: self.conv2.to(self.device)
            self.deconv.to(self.device)

        x = x.to(self.device)
        feat = feat.to(self.device)
        z = x.unsqueeze(1)
        #output and input = [sequence len, embedding size, batch]
            #we want the output to be --> [sequence len, batch, embedding size]
        z_embedding = self.conv1(z).permute(0,2,1)
        # positional = self.position_embedding(x)

        #target / source 
        len_seq = z_embedding.shape[0]
        len_src = int(len_seq * (1-target_len))

        #create the feature matrix before passing through the conv2
        if self.args.conf_cnn:
            # f_complete = np.repeat(feat.cpu(), len_seq, axis=0) #repeat each configuration 
            # f = f_complete.reshape(len_seq, feat.shape[0], feat.shape[1]) #so we obtain a matrix of shape [seq_len, batch, dim]
            # f = f.to(self.device)
            # f_embedding = self.conv2(f.permute(0,2,1)).permute(0,2,1)

            #feat is [batch, dim] UPDATEE
            f_conv = self.conv2(feat.reshape(feat.shape[0], feat.shape[1], 1)) #we pass the feat through the conv --> [batch, dim(32), 1]
            f = np.repeat(f_conv.cpu().detach().numpy(), len_seq, axis=0) #repeat each configuration and convert to numpy
            f_embedding = f.reshape(len_seq, f_conv.shape[0], f_conv.shape[1]) #obtain the matrix [seq, batch, dim]
            f_embedding = torch.tensor(f_embedding, dtype=torch.float32, requires_grad=True).to(self.device)


            #concat all the embeddings
            tot_emb = torch.cat((z_embedding, f_embedding), dim=2) #concatenate along the third dimension
        else:
            tot_emb = z_embedding

        src = tot_emb[:len_src, :, :]
        tgt = tot_emb[len_src:, :, :]
        trues = x[len_src:, :]

        if self.args.model_type == "forecasting":
            if test:
                self.model.module.eval()
                if self.args.use_multi_gpu and self.args.use_gpu: memory = self.model.module.encoder(src)
                else: memory = self.model.encoder(src)

                out = torch.zeros((tgt.shape[0]-1, tgt.shape[1], tgt.shape[2])) #shape of the output with the same size as the tgt
                point = tgt[0, :, :] #we use the first
                out[0, :, :] = point 
                
                #we will be doing auto-regressive decoding
                for i in range(1, out.shape[0]): #for every point in the sequence
                    if self.args.use_multi_gpu and self.args.use_gpu: point = self.model.module.decoder(point.reshape(1, point.shape[0], point.shape[1]), memory) #obtain the next point in the sequence (that we will be using as tgt)
                    else: point = self.model.decoder(point.reshape(1, point.shape[0], point.shape[1]), memory) 
                    point = point.reshape(point.shape[1], point.shape[2]) #[1, 10, 96] --> [10, 96]
                    out[i, :, :] = point

                    if i % 10 ==0: point = tgt[i, :, :] #use the good one
                    if i % 100 == 0:print(i)
                
                self.model.module.train()
            
            else:
                #we need to shift the tgt one to the right. The model needs to be learn to predict the next point.
                    #the first of the trues shold be the second of the tgt. 
                    # all except the last point --> then the trues will be all except the first
                out = self.model(src,tgt[:-1, :, :])

            out = self.linear(out)
            if self.args.two_linear: out = self.linear2(out)
            return out, trues[1:, :]


class MLP_simple(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=64, output_dim=1):
        super(MLP_simple, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

        