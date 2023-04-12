
import torch
import torch.nn as nn
from torch.nn import Transformer
from utils.myTools import dotdict
import numpy as np

from model.cnn_embeding import context_embedding, PositionalEmbedding
from model.model_layers import TransformerEncoderLayer, TransformerEncoder, MLP_simple, TransformerDecoder, TransformerDecoderLayer, SimpleDecoder
from data.procesamiento import postprocesing


class ShaftFormer(nn.Module):
    def __init__(self, args:dotdict, device) -> None:
        super().__init__()
        self.args = args
        self.device = device

        self.sigma = torch.tensor(0.5).to(device)

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

        ## DECODER
        # decoder_layer = TransformerDecoderLayer(d_model= self.args.outchannels, nhead= self.args.heads , dropout=self.args.dropout, device= device)
        # decoder = TransformerDecoder(decoder_layer, num_layers=self.args.nencoder)
        decoder = SimpleDecoder(self.args.outchannels, 1, self.args.dropout, device = self.device)


        ## MODEL
        if self.args.model_type == "forecasting":
            print("model for forecasting")
            
            ## DECODER
            if args.two_linear and args.conf_cnn:
                self.linear = torch.nn.Linear(args.outchannels, self.args.outchannels-self.args.outchannels_conf)
                self.linear2 = torch.nn.Linear(self.args.outchannels-self.args.outchannels_conf, args.inchannels)
            else:
                ##not using the residual connection
                self.linear = torch.nn.Linear(args.outchannels, args.inchannels)
            
            self.deconv = context_embedding(in_channels=self.args.outchannels, embedding_size=self.args.inchannels, k=self.args.kernel)

            ### LAYERS FOR RESIDUAL CONNECTIONS
                # One linear layer to estimate mean
            self.fc1 = nn.Linear(args.outchannels, args.inchannels)        
                 # One linear layer to estimate log-variance
            self.fc2 = nn.Linear(args.outchannels, args.inchannels) 

            if args.linear_initialization == 'Xavier':
                nn.init.xavier_uniform_(self.linear.weight)
            elif args.linear_initialization == 'He':
                nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            elif args.linear_initialization == 'Uniform':
                nn.init.uniform_(self.linear.weight, a=args.a, b=args.b)

            self.model = Transformer(d_model = self.args.outchannels, nhead=self.args.heads, custom_encoder=encoder, device=device, norm_first=True) #d_model must be divisible by nhead and d_model should be the same as the number of features of the data
            
            #GPU ________________________________________________
            if self.args.use_multi_gpu and self.args.use_gpu:
                print('\t Parallelization of the model')
                self.model = nn.DataParallel(self.model.cpu(), device_ids=self.args.device_ids, dim=1) #dim = 1 that is where the signal is --> [len, batch, dim]
                self.model = self.model.to(self.device)
            
            if self.args.use_gpu:
                self.linear.to(device)
                if self.args.two_linear: self.linear2.to(device)
                self.deconv.to(device)
                self.fc1.to(device)
                self.fc2.to(device)

        else:
            print("new layer for classification")

            self.simple_mlp = MLP_simple(input_dim= self.args.outchannels*250, output_dim=self.args.num_class)

            if self.args.use_gpu:
                self.simple_mlp.to(self.device)
        
        #both models
        self.conv1.to(device)
        if self.args.conf_cnn: self.conv2.to(device)
    
    def forward(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False, idx=None):
        return self._process_one_batch(x, feat, target_len, test, idx)
    
    def _process_one_batch(self, x: torch.Tensor, feat: torch.Tensor, target_len=0.3, test=False, idx=None):

        if test and self.args.model_type == "forecasting": 
            # self.device = torch.device("cpu")
            self.model.to(self.device).eval()
            self.linear.to(self.device).eval()
            if self.args.two_linear: self.linear2.to(self.device).eval()
            self.deconv.to(self.device).eval()
            self.conv1.to(self.device).eval()
            if self.args.conf_cnn: self.conv2.to(self.device).eval()
                    
        # if self.args.model_type=="classification":
        #     #we need to freeze the model part until the decoder (which we have change)
        #     self.conv1.requires_grad_ = False
        #     self.conv2.requires_grad_=False
        #     self.model.encoder.requires_grad_=False
        
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

                attention_mask = self._get_attention_mask(batch_size = tgt.shape[1], seq_len= tgt.shape[0]-1)
                attention_mask = attention_mask.to(self.device)

                #we need to shift the tgt one to the right. The model needs to be learn to predict the next point.
                    #the first of the trues shold be the second of the tgt. 
                    # all except the last point --> then the trues will be all except the first
                out = self.model(src,tgt[:-1, :, :], tgt_mask= attention_mask)

                #out = self.linear(out) # not using the residual connection
                
                # RESIDUAL CONNECTIONS
                mean = self.fc1(out) # We compute the mean
                variance = torch.exp(self.fc2(out))+torch.ones(mean.shape, device = self.device)*self.sigma # We compute the variance
                noise = torch.randn_like(mean)*torch.sqrt(variance) # We generate noise of the adecuate variance
                out = mean+noise # this is the sample (final value)

                if self.args.two_linear: out = self.linear2(out)
                
                return out, trues[1:, :]
        
            else: #classification model
                memory = self.model.encoder(tot_emb)

                #aply here the min-max scaler in reverse (preprocessing that is done to the signal)
                memory_new = postprocesing(memory, idx)
                memory_new= torch.tensor(memory_new).to(self.device)
                
                pred_class = self.simple_mlp.forward(memory_new) #now we use the memory from the encoder in the MLP
                return pred_class


        if test:
            self.model.eval()
            #we will be doing auto-regressive decoding
            #we need to make a for to process all the values (including the process of the embeddings)

            x_src = x[:len_src, :]
            trues = x[len_src:, :] #future values we want to predict knwoing x_src

            #conf embedding --> WE ALWAYS HAVE THIS, IS THE INFORMATION OF THE SIGNALS
            f_conv = self.conv2(feat.reshape(feat.shape[0], feat.shape[1], 1)) #we pass the feat through the conv --> [batch, dim(32), 1]
            f = np.repeat(f_conv.cpu().detach().numpy(), len_seq, axis=0) #repeat each configuration and convert to numpy
            f_embedding = f.reshape(len_seq, f_conv.shape[0], f_conv.shape[1]) #obtain the matrix [seq, batch, dim]
            f_embedding = torch.tensor(f_embedding, dtype=torch.float32, requires_grad=True).to(self.device)

            ##SRC
            z_src = x_src.unsqueeze(1)
            z_embedding = self.conv1(z_src).permute(0,2,1)
            points = torch.cat((z_embedding, f_embedding[:z_embedding.shape[0], :, :]), dim=2)
            memory = self.model.encoder(points)

            #x --> [seq, batch]
            out = torch.ones((trues.shape[0], trues.shape[1], 1)) #shape of the output --> [seq_len, batch, 1]

            w = 100
            last_points = trues[:w, :]

            
            for i in range(w, trues.shape[0]-1): #for every point in the signals

                if i % 100 == 0: print(i)
                
                z = last_points.unsqueeze(1)
                # z= test_points.unsqueeze(1)
                z_embedding = self.conv1(z).permute(0,2,1)
                points = torch.cat((z_embedding, f_embedding[:len(last_points), :, :]), dim=2) #ANTES len(last_points)

                attn_mask = self._get_attention_mask(points.shape[1], points.shape[0]) 
                attn_mask = attn_mask.to(self.device)
                future_point = self.model.decoder(points, memory, tgt_mask=attn_mask)  #[600, 10, 96] Result will be the next point  (matriz with zeros)
                pred_point = self.linear(future_point) 
                # pred_point = pred_point[-1:, :, :]

                #update the signal to have the new information
                last_points = torch.cat((last_points, pred_point[-1, :, 0].detach().view(1,-1)), 0)
                # last_points = pred_point[:i+1, :, 0]
                # test_points[i+1, :] = pred_point[i, :, 0].detach()

                # if i==w: out[:i, :, :] = pred_point[:i, :, :].cpu().detach()
                # else: out[i, :, :] = pred_point[i, :, : ].cpu().detach()
                out[:i, :, :] = pred_point.cpu().detach()
            
            return out[:-1, :, :], trues[1:, :]
    
    
    def _get_attention_mask(self, batch_size, seq_len):
        # create tensor with shape (batch_size, seq_len, seq_len) initialized with ones
        attention_mask = torch.ones(batch_size * self.args.heads, seq_len, seq_len)
        # attention_mask = torch.ones(batch_size, seq_len, seq_len)

        # set lower triangular part of attention mask to 0
        attention_mask = torch.triu(attention_mask, diagonal=1)
        """[0., 0., 0.,  ..., 0., 0., 0.],
         [1., 0., 0.,  ..., 0., 0., 0.],
         [1., 1., 0.,  ..., 0., 0., 0.],
         ...,
         [1., 1., 1.,  ..., 0., 0., 0.],
         [1., 1., 1.,  ..., 1., 0., 0.],
         [1., 1., 1.,  ..., 1., 1., 0.]"""
        
        new_attn_mask = attention_mask * -1e7 + 1

        #Transform the 1. to False and 0. to True. :: torch.logical_not(attention_mask)
        # attention_mask.bool() 
        # new_attn_mask = torch.zeros_like(attention_mask.bool() , dtype=torch.float32)
        # new_attn_mask.masked_fill_(attention_mask.bool() , -1e7)

        # return torch.logical_not(attention_mask)
        # return attention_mask.bool()
        return new_attn_mask