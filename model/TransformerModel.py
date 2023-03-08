
import torch
import torch.nn as nn
from torch.nn import Transformer
from utils.myTools import dotdict
import numpy as np

from model.cnn_embeding import context_embedding, PositionalEmbedding
from model.model_layers import TransformerEncoderLayer, TransformerEncoder, MLP_simple


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
                self.deconv.to(device)

        else:
            print("model for classification")
            decoder = nn.Linear(1, 1) #we are not going to use this
            transformer = Transformer(d_model = self.args.outchannels, nhead=self.args.heads, custom_encoder=encoder, custom_decoder=decoder, device=device, norm_first=True) #d_model must be divisible by nhead and d_model should be the same as the number of features of the data
            self.encoder_transformer = transformer.encoder

            self.simple_mlp = MLP_simple(input_dim= self.args.outchannels*32, output_dim=self.args.num_class)

            if self.args.use_gpu:
                self.encoder_transformer.to(self.device)
                self.simple_mlp.to(self.device)
        
        #both models
        self.conv1.to(device)
        if self.args.conf_cnn: self.conv2.to(device)
    
    def forward(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False):
        return self._process_one_batch(x, feat, target_len, test)
    
    def _process_one_batch(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False):

        if test and self.args.model_type == "forecasting": 
            self.device = torch.device("cpu")
            self.model.to(self.device).eval()
            self.linear.to(self.device).eval()
            if self.args.two_linear: self.linear2.to(self.device).eval()
            self.deconv.to(self.device).eval()
            self.conv1.to(self.device).eval()
            if self.args.conf_cnn: self.conv2.to(self.device).eval()
        
        #target / source 
        len_seq = x.shape[0]
        len_src = int(len_seq * (1-target_len))

        x = x.to(self.device)
        feat = feat.to(self.device)

        if not test:
            z = x.unsqueeze(1)
            #output and input = [sequence len, embedding size, batch]
                #we want the output to be --> [sequence len, batch, embedding size]
            z_embedding = self.conv1(z).permute(0,2,1)
            # positional = self.position_embedding(x)

            #create the feature matrix before passing through the conv2
            if self.args.conf_cnn:
                # f_complete = np.repeat(feat.cpu(), len_seq, axis=0) #repeat each configuration 
                # f = f_complete.reshape(len_seq, feat.shape[0], feat.shape[1]) #so we obtain a matrix of shape [seq_len, batch, dim]
                # f = f.to(self.device)
                # f_embedding = self.conv2(f.permute(0,2,1)).permute(0,2,1)

                #feat is [batch, dim] 
                f_conv = self.conv2(feat.reshape(feat.shape[0], feat.shape[1], 1)) #we pass the feat through the conv --> [batch, dim(32), 1]
                f = np.repeat(f_conv.cpu().detach().numpy(), len_seq, axis=0) #repeat each configuration and convert to numpy
                f_embedding = f.reshape(len_seq, f_conv.shape[0], f_conv.shape[1]) #obtain the matrix [seq, batch, dim]
                f_embedding = torch.tensor(f_embedding, dtype=torch.float32, requires_grad=True).to(self.device)


                #concat all the embeddings
                tot_emb = torch.cat((z_embedding, f_embedding), dim=2) #concatenate along the third dimension
            else:
                tot_emb = z_embedding

            if self.args.model_type == "forecasting":
                src = tot_emb[:len_src, :, :]
                tgt = tot_emb[len_src:, :, :]
                trues = x[len_src:, :]

                #we need to shift the tgt one to the right. The model needs to be learn to predict the next point.
                    #the first of the trues shold be the second of the tgt. 
                    # all except the last point --> then the trues will be all except the first
                # out = self.model(src,tgt[:-1, :, :])
                out = self.model(src,torch.roll(tgt, 1, dims=0))
    
                out = self.linear(out)
                if self.args.two_linear: out = self.linear2(out)
                
                return out, trues
            
            else: #classification model
                memory = self.encoder_transformer(tot_emb)
                
                pred_class = self.simple_mlp.forward(memory) #now we use the memory from the encoder in the MLP
                return pred_class


        if test:
            #we will be doing auto-regressive decoding
            #we need to make a for to process all the values (including the process of the embeddings)

            x_src = x[:len_src, :]
            x_tgt = x[len_src:, :]

            z_src = x_src.unsqueeze(1)
            z_src_embedding = self.conv1(z_src).permute(0,2,1)

            f_conv = self.conv2(feat.reshape(feat.shape[0], feat.shape[1], 1)) #we pass the feat through the conv --> [batch, dim(32), 1]
            f = np.repeat(f_conv.cpu().detach().numpy(), len_seq, axis=0) #repeat each configuration and convert to numpy
            f_embedding = f.reshape(len_seq, f_conv.shape[0], f_conv.shape[1]) #obtain the matrix [seq, batch, dim]
            f_embedding = torch.tensor(f_embedding, dtype=torch.float32, requires_grad=True).to(self.device)

            #concat all the embeddings
            src = torch.cat((z_src_embedding, f_embedding[:len_src, :, :]), dim=2)

            if self.args.use_multi_gpu and self.args.use_gpu: 
                self.model.module.eval()
                memory = self.model.module.encoder(src)
            else: memory = self.model.encoder(src)

            #x --> [seq, batch]
            w = 50 #window size
            trues = x[len_src:, :]
            new_x = x_tgt[:w, :]
            f_tgt = f_embedding[len_src:, :, :] 
            out = torch.ones((trues.shape[0], trues.shape[1], 1)) #shape of the output --> [seq_len, batch]
            
            t = x_tgt.shape[0]-w+1
            t=w
            for i in range(t): #for every point in the signals
                if i % 100 == 0: print(i)

                z = new_x.unsqueeze(1)
                z_embedding = self.conv1(z).permute(0,2,1)
                f_ = f_tgt[i:i+w, :, :]  

                point = torch.cat((z_embedding, f_), dim=2)

                if self.args.use_multi_gpu and self.args.use_gpu: point = self.model.module.decoder(point, memory) #obtain the next point in the sequence (that we will be using as tgt)
                else: point = self.model.decoder(point, memory)  #[50, 10, 96] Result will be the next point of every point

                pred_points = self.linear(point)
                if i != t-1 : out[i, :, :] = pred_points[0, :, :] #save only the first point (rest was only context) -> [seq, batch, dim=1]
                else: out[i:i+w, :, :] = pred_points
                
                new_x = pred_points.view(w,-1)
            
            return out[:-1, :, :], trues[1:, :]