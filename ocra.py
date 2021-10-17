# --------------------
# Model
# --------------------
# code Refs:
# modified from https://github.com/XifengGuo/CapsNet-Pytorch
# modified from https://github.com/kamenbliznashki/generative_models/blob/master/draw.py

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import math

from utils import *
from loaddata import *

# --------------------
# Model modules and loss functions
# --------------------

def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, batch_size, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        self.b = None  #torch.zeros(batch_size, self.out_num_caps, self.in_num_caps)
        self.c = None # 0.1* torch.ones(batch_size, self.out_num_caps, self.in_num_caps)

    def forward(self, x):
        device = x.device
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1, in_num_caps, in_dim_caps,  1]
        # weight.size   =[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        
        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        self.b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).to(device)
        self.c = F.softmax(self.b, dim=1)
        
        # keep all the coupling coefficients for all routing steps 
        coups = []
        coups.append(self.c.detach())
        
        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(self.c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(self.c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                self.b = self.b + torch.sum(outputs * x_hat_detached, dim=-1)
                
                # use a MAX-MIN normalization to separate the coefficients 
                self.c = 0.01 + ( .99 * ( (self.b - torch.min(self.b, dim=1, keepdim=True)[0] ) /     \
                            (torch.max(self.b, dim=1, keepdim=True)[0] - torch.min(self.b, dim=1, keepdim=True)[0]) ))
                
                # The original Capsule network used softmax but that fails to really separate the coupling coefficients
                # self.c = F.softmax(self.b, dim=1)
                coups.append(self.c.detach())
                

        outputs = torch.squeeze(outputs, dim=-2)
        return outputs, coups #, self.c, x_hat 


class OCRA(nn.Module):
    """
    OCRA model is trained to detect, classify, and reconstruct the objects in the image. 
    This model 
        1) encodes an image through recurrent read operations that forms a spatial attention on objects
        2) effectively binds features and classify objects through capsule representation and its dynamic routing
        3) decodes/reconstructs an image from class capsules through recurrent write operations 
    """
        
    def __init__(self, args):
        super().__init__()
        
        self.task = args.task
        # dataset info
        self.C, self.H, self.W = args.image_dims
        self.image_dims = args.image_dims
        self.num_targets = args.num_targets # the number of objects in the image
        self.cat_dup = args.cat_dup # whether objects can be from the same category 
        self.num_classes = args.num_classes # number of categories 
        
        # read and write attention. if attention is not used, then the model read/write the whole image 
        self.use_read_attn = args.use_read_attn        
        self.use_write_attn = args.use_write_attn  
        
        self.use_recon_mask = args.use_recon_mask
            
        if self.use_read_attn: 
            self.read_size = args.read_size
        else:
            self.read_size = self.H
            
        if self.use_write_attn: 
            self.write_size = args.write_size
        else:
            self.write_size = self.H
            
        # whether to use convolutional read operation or directly feed raw image pixels to the encoder RNN
        self.conv1_nfilters = args.conv1_nfilters  # number of filters in the first backbone layer
        self.conv2_nfilters = args.conv2_nfilters  # number of filters in the second backbone layer
        self.use_backbone = args.use_backbone
            
        if(self.use_backbone == 'conv_small'):

            self.bb_conv1 = nn.Conv2d(self.C, self.conv1_nfilters, kernel_size=5, stride=1, padding=2)
            self.bb_conv2 = nn.Conv2d(self.conv1_nfilters, self.conv2_nfilters, kernel_size=3, stride=1, padding=1)
            
        elif(self.use_backbone == 'conv_med'):
        
            # self._hidden1 = nn.Sequential(
                # nn.Conv2d(in_channels=self.C, out_channels=48, kernel_size=5, padding=2),
                # nn.BatchNorm2d(num_features=48),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                # nn.Dropout(0.2)
            # )
            self._hidden1 = nn.Sequential(
                nn.Conv2d(in_channels=self.C, out_channels=self.conv1_nfilters, kernel_size=5, padding=2),
                nn.BatchNorm2d(num_features=self.conv1_nfilters),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(0.2)
            )
            self._hidden2 = nn.Sequential(
                nn.Conv2d(in_channels=self.conv1_nfilters, out_channels=self.conv2_nfilters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=self.conv2_nfilters),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(0.2)
            )
        # the number of complete cycle of encoder-decoder 
        self.time_steps = args.time_steps
        
        # encoder RNN 
        self.include_xhat = args.include_xhat #xhat: the diff btw the most recent cumulative canvas and the input image
        self.lstm_size = args.lstm_size
        self.decoder_encoder_feedback = args.decoder_encoder_feedback
        
        if self.use_read_attn:
            if self.read_size == 18:
                bb2encoder_size = 512
            elif self.read_size == 12:
                bb2encoder_size = 288
        else:
            bb2encoder_size = 32*9*9
            
        #if (self.conv2_nfilters == 64) and self.use_backbone:
        self.encoder_input_size = int(self.conv2_nfilters/32) * bb2encoder_size
                
        if self.decoder_encoder_feedback:
            self.encoder_input_size = self.lstm_size + self.encoder_input_size
            
        
        if(self.use_backbone):
            #int((self.conv2_nfilters * self.read_size * self.read_size)/8) + self.lstm_size
            self.encoder = nn.LSTMCell(self.encoder_input_size, self.lstm_size) 
        else:
            self.encoder = nn.LSTMCell((self.include_xhat+1)*self.read_size*self.read_size*self.C+self.lstm_size, self.lstm_size)
        
        self.use_capsnet = args.use_capsnet
        
        self.num_zcaps = args.num_zcaps # size of linear layer from encoder to primary caps
        self.dim_zcaps = args.dim_zcaps # primary caps dim
        self.z_size = self.num_zcaps * self.dim_zcaps
        self.z_linear = nn.Linear(self.lstm_size, self.z_size)
        if self.use_capsnet:
        # dynamic routing capsules
            self.routings = args.routings  # number of dynamic routings between two capsule layers 
            self.num_objectcaps = args.num_classes + args.backg_objcaps  # number of final class object caps
            self.dim_objectcaps = args.dim_objectcaps # final class object caps dim
            self.objectcaps_layer = DenseCapsule(batch_size = args.train_batch_size, in_num_caps=self.num_zcaps, 
                                                 in_dim_caps=self.dim_zcaps, out_num_caps=self.num_objectcaps,
                                                 out_dim_caps=self.dim_objectcaps, routings=self.routings)
        else:
            self.num_objectcaps = args.num_classes + args.backg_objcaps  # number of final class object caps
            self.dim_objectcaps = args.dim_objectcaps # final class object caps dim
            self.fc_linear = nn.Linear(self.z_size, self.num_objectcaps*self.dim_objectcaps)
            self.output_linear = nn.Linear(self.num_objectcaps*self.dim_objectcaps, self.num_objectcaps)
            self.softmax = nn.Softmax(dim=1)
            
        # decoder RNN/reconstruction  
        self.decoder = nn.LSTMCell(self.num_objectcaps * self.dim_objectcaps, self.lstm_size)   
        self.mask_objectcaps = args.mask_objectcaps # whether to use masked objectcaps for decoder
        self.recon_model = args.recon_model # whether decoder generates a reconstruction of the input         
        self.write_linear = nn.Linear(self.lstm_size, self.write_size * self.write_size *self.C) # write layer getting input from decoder RNN and generating what will be written to the canvas 
        self.relu = nn.ReLU()
        
        # attention layers that output 4 params that specify NxN array of gaussian filters     
        self.read_attention_linear = nn.Linear(self.lstm_size, 4) 
        self.write_attention_linear = nn.Linear(self.lstm_size, 4)
        
        if self.task == 'multisvhn':
            self.digits_readout_0 = nn.Linear(2*self.num_classes, 11)
            self.digits_readout_1 = nn.Linear(2*self.num_classes, 11)
            self.digits_readout_2 = nn.Linear(2*self.num_classes, 11)
            self.digits_readout_3 = nn.Linear(2*self.num_classes, 11)
            self.digits_readout_4 = nn.Linear(2*self.num_classes, 11)
                
    def read(self, x, x_hat, h_dec, external_att_param=None):
        
        if self.use_read_attn:
            
            if external_att_param: # if the attention parameters are externally set
                g_x, g_y, delta, mu_x, mu_y, F_x, F_y = external_att_param
                g_x, g_y, delta, mu_x, mu_y, F_x, F_y  = g_x.detach(), g_y.detach(), delta.detach(),\
                                                        mu_x.detach(), mu_y.detach(), F_x.detach(), F_y.detach() 
            else: # get attention params and compute gaussian filterbanks 
                g_x, g_y, logvar, logdelta = self.read_attention_linear(h_dec).split(split_size=1, dim=1)  # removed loggamma
                g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta,
                                                                                    self.H, self.W, self.read_size)
            # expand filterbanks to have input channel dimension 
            # [n_batch, n_channel, im_x or im_h] --> [n_batch, n_channel, read_attn_size, im_x or im_h] 
            F_x = torch.unsqueeze(F_x, 1)
            F_x = F_x.repeat(1, self.C, 1, 1)  
            F_y = torch.unsqueeze(F_y, 1)
            F_y = F_y.repeat(1, self.C, 1, 1)

            # apply filterbanks and get N*N (read_size) read patch output 
            # [B,C,N,N] = [B,C,N,H] @ [B,C,H,W] @ [B,C,W,N]
            new_x     = F_y @ x.view(-1, self.C, self.H, self.W)    @ F_x.transpose(-2, -1)  
            new_x_hat = F_y @ x_hat.view(-1, self.C, self.H, self.W) @ F_x.transpose(-2, -1) 
            read_att_param = [g_x, g_y, delta, mu_x, mu_y, F_x, F_y] 

            return new_x, new_x_hat, read_att_param
        
        else: # if not usring read attention, use raw images
            return x, x_hat, []

    def write(self, h_dec):
        # get write_size * write_size writing patch (if no write attn is used, write_size is set to be image height)
        w_g = self.write_linear(h_dec)
        
        if self.use_write_attn: 
            # get attention params and compute gaussian filterbanks
            g_x, g_y, logvar, logdelta = self.write_attention_linear(h_dec).split(split_size=1, dim=1) #removed loggamma 
            g_x, g_y, delta, mu_x, mu_y, F_x, F_y = compute_filterbank_matrices(g_x, g_y, logvar, logdelta, self.H, self.W, self.write_size)
           
            # expand filterbanks to have input channel dimension 
            # [n_batch, n_channel, im_x or im_h] --> [n_batch, n_channel, write_attn_size, im_x or im_h] 
            F_y = torch.unsqueeze(F_y, 1)
            F_y = F_y.repeat(1, self.C, 1, 1) 
            F_x = torch.unsqueeze(F_x, 1)
            F_x = F_x.repeat(1, self.C, 1, 1) 
            
            # apply filterbanks and get read output in original coords H*W 
            # [B,C,H,W] = [B,C,H,N] @ [B,C,N,N] @ [B,C,N,W]            
            w = F_y.transpose(-2, -1) @ w_g.view(-1, self.C, self.write_size, self.write_size) @ F_x
            write_att_param = [g_x, g_y, delta, mu_x, mu_y, F_x, F_y, w_g.detach()]
            
            return  w.view(w.shape[0], -1), write_att_param 

        else: # if not using write attention, just use raw write output from the decoder            
            return w_g, []

    def forward(self, x, y=None):
        
        batch_size = x.shape[0]
        device = x.device
        
        # intitalize the hidden/cell state for the encoder and decoder
        h_enc = torch.zeros(batch_size, self.lstm_size).to(device)
        c_enc = torch.zeros_like(h_enc)
        h_dec = torch.zeros(batch_size, self.lstm_size).to(device)
        c_dec = torch.zeros_like(h_dec)
        
        # initialize read output in orginal coords and objcaps length (these will culmulate over timesteps)
        read_x_step = torch.zeros(batch_size, self.time_steps, self.C*self.H*self.W).to(device)        
        objcaps_len_sum = torch.zeros(batch_size, self.num_objectcaps).to(device)
        objcaps_len_step = torch.zeros(batch_size, self.time_steps, self.num_objectcaps).to(device) 
        objcaps_step = torch.zeros(batch_size, self.time_steps, self.num_objectcaps*self.dim_objectcaps).to(device) 
        
        readout_logits = torch.zeros(batch_size, self.num_targets*self.num_classes).to(device) 
        
        # initialize the canvas 
        c = torch.zeros(x.shape).to(device)
        c_step = torch.zeros(batch_size, self.time_steps, self.C*self.H*self.W).to(device) 
        c_catmasked = torch.zeros(batch_size, self.num_objectcaps, self.C*self.H*self.W).to(device)  

        # run model forward
        for t in range(self.time_steps):
            
            # xhat: diff btw recent canvas and the orginal image, whatever left to be reconstructed 
            x_hat = x.to(c.device) - c  
                     
            # get read outputs (see function read for details)   
            new_x, new_x_hat, read_att_param = self.read(x, x_hat, h_dec)
            
            # whether to pass xhat reads to encoder
            # if using convolutional read, it is better not to include the x_hat
            if self.include_xhat:
                r = torch.cat([new_x.view(x.shape[0], -1), new_x_hat.view(x.shape[0], -1)], dim=1)
            else:
                r = torch.cat([new_x.view(x.shape[0], -1)], dim=1) 

            #Use these two lines to limit the number of samples the model can take
#             allowed_num_samples = 2
#             if t>= allowed_num_samples:
#                 r  = torch.zeros(batch_size,2*self.read_size*self.read_size*self.C).to(device)
      
            # store read attn param and read outputs
            if self.use_read_attn:
                [g_x, g_y, delta, mu_x, mu_y, F_x, F_y] = read_att_param            
                
                # cumulate what's read coverted to orginal coords 
                read_x = F_y.detach().transpose(-2,-1) @ new_x.detach() @ F_x.detach() # [B,C,H,W] = [B,C,H,R] @ [B,C,R,R] @ [B,C,R,W]
                read_x_step[:,t:t+1,:] = torch.unsqueeze(read_x.view(x.shape[0], -1), 1).detach()      
            else:
                read_x_step[:,t:t+1,:] = torch.unsqueeze(new_x.view(x.shape[0], -1), 1).detach()  
                
            # if using the convolutional read, apply the two layer conv on the read output     
            if(self.use_backbone == "conv_small"):
                #r = self.bb_cnn(r.view(x.shape[0], -1, self.read_size, self.read_size))
                r_conv1 = F.relu(F.max_pool2d(self.bb_conv1(r.view(x.shape[0], -1, self.read_size, self.read_size)), 2))
                r_conv2 = F.relu(F.max_pool2d(self.bb_conv2(r_conv1), 2))
                r = r_conv2.view(x.shape[0], -1)
            
            elif(self.use_backbone == "conv_med"):
                r = self._hidden1(r.view(x.shape[0], -1, self.read_size, self.read_size))
                r = self._hidden2(r)
                #r = self._hidden3(r)
                r = r.view(r.size(0), -1)
            
            # feed the read & decoder output to the encoder RNN
            if self.decoder_encoder_feedback:
                h_enc, c_enc = self.encoder(torch.cat([r, h_dec], dim=1), (h_enc, c_enc))
            else:
                h_enc, c_enc = self.encoder(r, (h_enc, c_enc))
 
            z_sample = self.z_linear(h_enc) # (n_batch, self.z_size)
            if self.use_capsnet:      
                # linear transformation to form primary capsules  
                
                zcaps = z_sample.contiguous().view(z_sample.size(0), -1, self.dim_zcaps) # (n_batch, z_size/z_dim, dim_zcaps)

                # class object capsules with dynamic routing (see DenseCapsule for details) 
                objectcaps, coups =  self.objectcaps_layer(zcaps) # objectcaps [n_batch, n_objects(class+bkg), dim_objectcaps] 

                # get the length of object capsule for each step and cumulate for predicting categories
                objcaps_len = objectcaps.norm(dim=-1)
                objcaps_len_step[:,t:t+1,:] = torch.unsqueeze(objcaps_len, 1) 

                # get y_onehot to mask objectcaps (for routing one final object casule to the decoder)
                if y is not None: # if y is given, use y to mask object caps
                    y_onehot = y
                else: # if y is not given, use the most active capsule
                    max_act_obj = objcaps_len.max(dim=1)[1]
                    y_onehot = torch.zeros(objcaps_len.size(), device=max_act_obj.device).scatter_(1, max_act_obj.view(-1,1), 1.)
                objectcaps_masked = (objectcaps * y_onehot[:, :, None])

                # whether to use masked object caps (or pass all object capsules 
                if self.mask_objectcaps:
                    objectcaps_re = objectcaps_masked.contiguous().view(objectcaps.size(0), -1)
                else:
                    objectcaps_re = objectcaps.contiguous().view(objectcaps.size(0), -1)
                    
                objcaps_step[:,t:t+1,:] = torch.unsqueeze(objectcaps_re, 1)
                
            else:
                linear_readout = self.fc_linear(z_sample)
                
                # get softmax output, treated as analogue to the length of object capsule for each step and cumulate for predicting categories
                output = self.output_linear(linear_readout) # for classification
                objcaps_len = self.softmax(output)
                objcaps_len_step[:,t:t+1,:] = torch.unsqueeze(objcaps_len, 1) 

                # no masking based object representation possible, just use linear readout above for reconstruction
                objectcaps_re = linear_readout 
                
            # feed objectcaps to the decoder RNN
            h_dec, c_dec = self.decoder(objectcaps_re, (h_dec, c_dec))  
            
            # whether do reconstruction from decoder outputs
            if self.recon_model: 
                # get write canvas ouput
                c_write, write_att_param = self.write(h_dec)
                c_write = self.relu(c_write) # only positive segments are written to the canvas, prevents deleting of earlier writes 
                c_step[:,t:t+1,:] = torch.unsqueeze(c_write, 1) # keep whats written to the canvas at each step
            

            if self.task == 'multisvhn':
                digits_logits_0 = self.digits_readout_0(objcaps_len_step[:,2:4,0:self.num_classes].view(objcaps_len_step.size(0), -1))
                digits_logits_1 = self.digits_readout_1(objcaps_len_step[:,4:6,0:self.num_classes].view(objcaps_len_step.size(0), -1))
                digits_logits_2 = self.digits_readout_2(objcaps_len_step[:,6:8,0:self.num_classes].view(objcaps_len_step.size(0), -1))
                digits_logits_3 = self.digits_readout_3(objcaps_len_step[:,8:10,0:self.num_classes].view(objcaps_len_step.size(0), -1))
                digits_logits_4 = self.digits_readout_4(objcaps_len_step[:,10:12,0:self.num_classes].view(objcaps_len_step.size(0), -1))
                readout_logits = torch.cat((digits_logits_0, digits_logits_1, digits_logits_2, digits_logits_3, digits_logits_4), dim = 1)  

               
        '''
        Returns
        objcaps_len_step -- length of object capsules at each timestep
        read_x_step -- Sum of all reads from image (use for masking reconstruction error more focused on the read parts)
        c_step --- list of canvases at each timestep
        
        
        old stuff
        c -- tensors of shape (B, 1); final cumulative reconstruction canvas
        y_pred -- cumulative length of object capsules; final prediction vector 
        objcaps_len_step -- length of object capsules at each timestep 
        att_param_step -- all model outputs stored at each step
            0) read attention param 
            1) read output 
            2) write attention param
            3) write output (canvas) 
            4) cumulative canvas 
            5) class masked canvas 
            6) coupling coefficients
        '''
        
        return objcaps_len_step, read_x_step, c_step, readout_logits   # att_param_step
    

def compute_filterbank_matrices(g_x, g_y, logvar, logdelta, H, W, attn_window_size):
    """ DRAW section 3.2 -- computes the parameters for an NxN grid of Gaussian filters over the input image.
    
    note. B = batch dim; N = attn window size; H = original heigh; W = original width
    
    Args 
        g_x, g_y -- tensors of shape (B, 1); unnormalized center coords for the attention window
        logvar -- tensor of shape (B, 1); log variance for the Gaussian filters (filterbank matrices) on the attention window
        logdelta -- tensor of shape (B, 1); unnormalized stride for the spacing of the filters in the attention window
        H, W -- scalars; original image dimensions
        attn_window_size -- scalar; size of the attention window (specified by the read_size / write_size input args

    Returns
        g_x, g_y -- tensors of shape (B, 1); normalized center coords of the attention window;
        delta -- tensor of shape (B, 1); stride for the spacing of the filters in the attention window
        mu_x, mu_y -- tensors of shape (B, attn_window_size); means location of the filters at row and column
        F_x, F_y -- tensors of shape (B, N, W) and (B, N, H) where N=attention_window_size; filterbank matrices
    """
    batch_size = g_x.shape[0]
    device = g_x.device

    # rescale attention window center coords and stride to ensure the initial patch covers the whole input image
    # eq 22 - 24
    g_x = 0.5 * (W + 1) * (g_x + 1)  # (B, 1)
    g_y = 0.5 * (H + 1) * (g_y + 1)  # (B, 1)
    delta = (max(H, W) - 1) / (attn_window_size - 1) * logdelta.exp()  # (B, 1)

    # compute the means of the filter
    # eq 19 - 20
    mu_x = g_x + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)
    mu_y = g_y + (torch.arange(1., 1. + attn_window_size).to(device) - 0.5*(attn_window_size + 1)) * delta  # (B, N)

    # compute the filterbank matrices
    # eq 25 -- combines logvar=(B, 1, 1) * ( range=(B, 1, W) - mu=(B, N, 1) ) = out (B, N, W); then normalizes over W dimension;
    F_x = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + W).repeat(batch_size, 1, 1).to(device) - mu_x.unsqueeze(-1))**2)
    F_x = F_x / torch.sum(F_x + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image
    # eq 26
    F_y = torch.exp(- 0.5 / logvar.exp().view(-1,1,1) * (torch.arange(1., 1. + H).repeat(batch_size, 1, 1).to(device) - mu_y.unsqueeze(-1))**2)
    F_y = F_y / torch.sum(F_y + 1e-8, dim=2, keepdim=True)  # normalize over the coordinates of the input image

    # return rescaled attention window center coords and stride + gaussian filters 
    return g_x, g_y, delta, mu_x, mu_y, F_x, F_y


def loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y_true, args, writer=None, step=None):
    """
    Total Loss = lamda_recon*(reconstruction_loss) + classification_loss
    
    Args
        c -- cumulative Canvas
        recon_mask -- a mask that is the normalized sum of all the read outputs
        x -- input
        y_true -- groundtruth y in n-hot format 
        y_pred -- model predictions; (normalized) cumulative length of class capsules 
    """
    
    y_pred = torch.sum(objcaps_len_step, dim=1)
    # Do not normalize the sum of object caps lengths if multiple items can be from the same category
    if (not args.cat_dup) and (args.task != 'mnist_ctrv') and (torch.min(torch.max(y_pred, dim=1, keepdim=True)[0]) != 0):
        y_pred = y_pred / torch.max(y_pred, dim=1, keepdim=True)[0] #self.num_objectcaps #objectcaps.norm(dim=-1)   
    
    y_pred = y_pred.narrow(1,0,args.num_classes) # in case a background cap was added, narrow the tensor before passing it to the loss function , dim, start, length
        
    #######   Reconstruction Loss      #######
    
    c = torch.sum(c_step, dim=1)
    
    # use recon mask to focus the ground truth reconstruction on the relevant stuff (what has been read by the model)
    if (args.use_recon_mask):
        read_x_sum = torch.sum(read_x_step, dim=1)
        recon_mask = read_x_sum/torch.max(read_x_sum, dim=1, keepdim=True)[0]
        x = .3*x + .7*recon_mask*x 
    
    # clip the predicted cumulative canvas so that the model can overlap object reconstructions without increasing error
    if (args.clip_c):
        c = torch.clip(c,0,1)
        
    # weight the reconstruction error before adding it to the classification loss
    L_recon = args.lam_recon * nn.MSELoss()(c, x)  
    

    #######   Classification Margin Loss      #######
    
    if len(y_true.shape) == 3: 
        y_true_1d = torch.clamp(torch.sum(y_true, dim=1), max=1)
    else:
        y_true_1d = y_true
         
    # classification error: margin error for class capsules -- allows for both GT and prediction to have any number for any class 
    m_neg = 0.1 # margin loss allowed for negative case (for absent digits)
    lam_abs = 0.5 # down-weighting loss for absent digits (prevent the initial learning from shrinking the lengths of the class capsules    
    L_present =  torch.clamp(y_true_1d, min=0., max=1.) * torch.clamp((y_true_1d-m_neg) - y_pred, min=0.) ** 2   
#     L_present =  y_true_1d * torch.clamp((y_true_1d-m_neg) - y_pred, min=0.) ** 2   # not clamped version
    L_absent = lam_abs * torch.clamp(1 - y_true_1d, min=0.) * torch.clamp(y_pred-m_neg, min=0.) ** 2
    L_margin = (L_present+L_absent).sum(dim=1).mean()
    
    
    #######   Readout Margin Loss      #######
    
    if args.task == 'multisvhn':
        #print(y_true[1])
        col_11 = torch.zeros((y_true.size(0),5, 1)).to(args.device)
        #print(torch.squeeze(y_true[1,0,10:args.num_class_caps]))
        col_11[:,1:5, 0] = torch.flip(torch.squeeze(y_true[:,0,10:args.num_classes]), (1,)) 
        #print(col_11[1])
        #print(y_true[:,:,0:10].shape)

        y_true = torch.cat( (y_true[:,:,0:10], col_11), dim=2).view(y_true.size(0), -1)
    
        L_present =  torch.clamp(y_true, min=0., max=1.) * torch.clamp((y_true-m_neg) - readout_logits, min=0.) ** 2   
        #L_present = L_present + torch.clamp(y_true_1d, min=0., max=1.) * torch.clamp(y_pred - (y_true_1d+0.5) , min=0.) ** 2  
    #     L_present =  y_true * torch.clamp((y_true-m_neg) - y_pred, min=0.) ** 2   # not clamped version
        L_absent = lam_abs * torch.clamp(1 - y_true, min=0.) * torch.clamp(readout_logits-m_neg, min=0.) ** 2
        L_margin_digits = (L_present+L_absent).sum(dim=1).mean()
        
        L_margin = L_margin + (10*L_margin_digits)
        
    return L_recon + L_margin, L_recon , L_margin 


# --------------------
# Train and Test
# --------------------

def get_topkacc(y_pred: torch.Tensor, y_true:  torch.Tensor, topk=1):
    """
    Get indices of topk model predictions and gather how many of them are correct (in percentage).
    e.g., if 1 correct out of top2 prediction --> 0.5
    
    Input: torch tensor
        - y_pred should be a vector of prediction score 
        - y_true should be in multi-hot encoding format (one or zero; can't deal with duplicates)

    Return: 
        - a vector of accuracy from each image --> [n_images,]
        - average acc
    """
    n_images = y_pred.size(0)
    topk_indices = y_pred.data.topk(topk, sorted=True)[1] 
    accs = torch.gather(y_true, dim=1, index=topk_indices).sum(dim=1)/topk
    average_acc = accs.cpu().sum().item()/n_images

    return average_acc, accs

def get_exactmatch(y_pred_hot: torch.Tensor, y_true: torch.Tensor):
    """
    See if y_pred and y_true matches exactly
    e.g., if match acc=1, not match acc=0
    
    Input: torch tensor 
        - both y_pred and y_true should be in the same format
        e.g., if y_true is multi-hot, then y_pred should be made in multi-hot as well
    Return: 
        - a vector of accuracy from each image --> [n_images,]
        - average acc
    """
    n_images = y_pred_hot.size(0)
    accs = (y_pred_hot == y_true).all(dim=1).float()
    average_acc = accs.cpu().sum().item()/n_images
    
    return average_acc, accs

def cal_accs(y_pred_nar, y_true, readout_logits, args):
    """ calculate accuracy
    1) when args.cat_dup == True --> use criterion value to indicate duplicate classification
    2) when args.cat_dup == False --> just topk based accuracy
    exact match: if one of digits incorrect  --> 0
    partial match: if one of digits incorrect --> 0.5
    
    return:
        - accuracy = accuracy sum over the whole batch
        - accs = a list of correct score for each image 
    """
    n_targets = args.num_targets
    
    
    if args.task=='multisvhn':  # when there are two targets with duplicates allowed
    
        col_11 = torch.zeros((y_true.size(0),5, 1)).to(args.device)
        col_11[:,1:5, 0] = torch.flip(torch.squeeze(y_true[:,0,10:args.num_classes]), (1,)) 
        y_true = torch.cat( (y_true[:,:,0:10], col_11), dim=2)


        readout_logits = readout_logits.view(-1, 5,11)
        out, pred_digits = torch.max(readout_logits,dim=2)
        out, true_digits = torch.max(y_true,dim=2)
        
        partial_accs = torch.sum(1*(pred_digits == true_digits), dim=1) / float(args.num_targets)        
        partial_accuracy = partial_accs.cpu().sum().item() 
        exact_accs = (partial_accs == 1)
        exact_accuracy =  exact_accs.cpu().sum().item() 
        
        y_pred_hot = torch.sum(readout_logits, dim=1)
        
        #exact_accuracy = torch.sum(1* (torch.sum(1*(pred_digits == true_digits), dim=1) == 5)).type('torch.FloatTensor').cpu()  # 
        
    
    elif args.cat_dup == True:  # when there are two targets with duplicates allowed
        # get bool indices for an image with duplicates
        dup_t = n_targets - 0.2# 1.8 for target=2; criterion value indicating duplicates (two targets are from the same category)
        bool_above = (y_pred_nar >= dup_t).any(dim=1)

        # when any of predictions are above dup_t, apply the following

        y_pred_dup = (y_pred_nar == torch.max(y_pred_nar, dim= 1)[0].reshape(-1,1)) # get y_pred for duplicates; the largest value --> mark as 1, e.g., y_pred  [2.1, 0.2, 0.1] --> [1, 0, 0] 
        # maxid = y_pred_nar.argmax(dim=1)
        # y_pred_dup = torch.zeros(y_pred_nar.shape).scatter(1, maxid.unsqueeze(dim=1), 1.0)
        dupaccs = torch.sum(y_true*y_pred_dup, dim=1)/2.0 # compare with y_true and get acc, 
        # e.g, when y_pred_dup = [1, 0, 0], acc=1 when y_true = [2,0,0] and and acc=0.5 when y_true = [1,1,0]

        # when none of predictions are above dup_t, apply the following
        y_true_clip = torch.clip(y_true,0,1) # y_true [2,0] --> [1,0]
        y_pred_nodup = (y_pred_nar >= y_pred_nar.topk(n_targets)[0][:,n_targets-1:n_targets]) # bool indices of predictions higher/equal than n_target highest value  y_pred = [1.0,0.9,0.1] --> [1, 1, 0]
        nodupaccs = torch.sum(y_true_clip*y_pred_nodup, dim=1)/float(n_targets)  # compare with y_true and get acc, 
        # e.g., when y_pred = [1.0,0.9,0.1], acc=1, when y_true = [1,1,0] and acc=0.5, when y_true = [2,0,0]

        # combine match for both no duplicates and yes duplicates

        y_pred_hot = n_targets*(bool_above.reshape(-1,1)*y_pred_dup)+ (~bool_above).reshape(-1,1)*y_pred_nodup
        partial_accs = bool_above*dupaccs + (~bool_above)*nodupaccs
        exact_accs = (partial_accs >= 1)
        partial_accuracy = partial_accs.cpu().sum().item() # if gt = 1, 2 and pred = 2, 3 --> 50 % acc
        exact_accuracy =  exact_accs.cpu().sum().item()  # if gt = 1, 2 and pred = 2, 3 --> 0 % acc
        
                    
    elif args.cat_dup == False: # when no duplicates
        y_pred_hot = (y_pred_nar >= y_pred_nar.topk(n_targets)[0][:,n_targets-1:n_targets]) # bool indices of predictions higher/equal than n_target highest value y_pred = [1.0,0.9,0.1] --> [1, 1, 0]
        partial_accs = torch.sum(y_pred_hot*y_true, dim=1)/float(n_targets) 
        partial_accuracy = partial_accs.cpu().sum().item() 
        exact_accs = (partial_accs >= 1)
        exact_accuracy =  exact_accs.cpu().sum().item() 
        
    return y_pred_hot, partial_accuracy, partial_accs, exact_accuracy, exact_accs  
            
    #     #####################
    #     # alternative version: using groundtruth to know whether duplicates trial or not
    #     """
    #     for multimnist, acc should be the same as the version above, acc was ~94% after 3 epoch
    #     for cluttered task, acc was 80% after 3 epoch (higher than acc from the version above; ~72-75%%, and 80% was reached around 10 epoch)
    #     """
    #     # get bool indices for an image with duplicates
    #     n_targets = args.num_targets
    #     bool_duplicate = (y_true==n_targets).any(dim=1) 

    #     # when no duplicates in the image, apply topk predictions
    #     _, top2accs = get_topkacc(y_pred_nar, y_true, topk=2)

    #     # when yes duplicates in the image, apply exact match 
    #     dup_t = 1.85 
    #     y_pred_dup = n_targets*(y_pred_nar >= dup_t) # if prediction > dup_t, we consider it as prediction for duplicates, e.g, prediction [1.9, 1.5, 0.5] -> [2, 0, 0]
    #     _, matchaccs = get_exactmatch(y_pred_dup, y_true)

    #     # combine topk (for no duplicates) + exactmatch (for yes duplicates) and get total sum
    #     combaccs = (~bool_duplicate)*top2accs + (bool_duplicate)*matchaccs
    #     accuracy = combaccs.cpu().sum().item()
    #     #######################

    
@torch.no_grad()
def evaluate(model, x, y_true, loss_fn, args, epoch=None):
    """
    Run model prediction on testing dataset and compute loss/acc 
    
    Args
        model -- trained model to be evaluated with no_grad()
        recon_mask -- A mask that is the normalized sum of all the read operatoin to focus the erorr reconstructio
        x -- input
        y_true -- input y in n hot format 
        y_pred -- (normalized) cumulative length of the class capsules 
    """

    # evaluate
    model.eval()
    
    # load testing dataset on device
    x = x.view(x.shape[0], -1).float().to(args.device)
    
    if args.task == 'mnist_ctrv':
        x = 1.0 - x
                
    y_true = y_true.to(args.device)
    
    # run model with testing data and get predictions
    if (args.class_cond_mask):
        # groundtruth info will be used to mask objectcaps
        objcaps_len_step, read_x_step, c_step, readout_logits = model(x,y_true)       
    else: # model prediction will be used to mask objectcaps
        objcaps_len_step, read_x_step, c_step, readout_logits  = model(x)
        
    y_pred = torch.sum(objcaps_len_step, dim=1)
    # Do not normalize the sum of object caps lengths if multiple items can be from the same category
    if (not args.cat_dup) and (args.task != 'mnist_ctrv') and (torch.min(torch.max(y_pred, dim=1, keepdim=True)[0]) != 0):
        y_pred = y_pred / torch.max(y_pred, dim=1, keepdim=True)[0] #self.num_objectcaps #objectcaps.norm(dim=-1)   
        
    # compute accuracy sum over whole batch

    _, partial_accuracy, _ , exact_accuracy, _ = cal_accs(y_pred.narrow(1,0,args.num_classes), y_true, readout_logits, args)
    
    # compute loss    
    loss, L_recon, L_margin = loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y_true, args)        


    return loss, L_recon , L_margin, exact_accuracy, partial_accuracy, read_x_step, c_step, y_pred, objcaps_len_step, readout_logits


def test(model, dataloader, args):
    """
    for each batch:
        - evaluate loss & acc ('evaluate')
    log average loss & acc  
    """   
    test_loss = 0
    test_L_recon = 0
    test_L_margin = 0
    test_acc_partial = 0
    test_acc_exact = 0
    
    # load batch data
    for x, y in dataloader:
        
        # if one target and y is not in one-hot format, convert it to one-hot encoding
        if args.num_targets == 1:
            if len(y.shape) < 2: 
                y = y.type(torch.int64)
                y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.)  
        
        # evaluate
        batch_loss, batch_L_recon, batch_L_margin, batch_acc_exact, batch_acc_partial, read_x_step, \
        c_each_step, y_pred, objcaps_len_step, readout_logits =  evaluate(model, x, y, loss_fn, args)

        # aggregate loss and acc
        test_loss += args.test_batch_size * batch_loss
        test_L_recon += args.test_batch_size * batch_L_recon
        test_L_margin += args.test_batch_size * batch_L_margin 
        test_acc_partial += batch_acc_partial
        test_acc_exact += batch_acc_exact

    # get average loss and acc
    test_loss /= len(dataloader.dataset)
    test_L_recon /= len(dataloader.dataset)
    test_L_margin /= len(dataloader.dataset)
    test_acc_partial /= (len(dataloader.dataset))
    test_acc_exact /= (len(dataloader.dataset))
    return test_loss, test_L_recon, test_L_margin, test_acc_partial, test_acc_exact



def train_epoch(model, train_dataloader, loss_fn, optimizer, epoch, writer, args):
    """
    for each batch:
        - forward pass  
        - compute loss
        - param update
    log average train_loss  
    """    
    model.train() 
    with tqdm(total=len(train_dataloader), desc='epoch {} of {}'.format(epoch, args.n_epochs)) as pbar:
#     time.sleep(0.1)        
        training_loss = 0.0
        
        # load batch from dataloader 
        for i, (x, y) in enumerate(train_dataloader):
            global_step = (epoch-1) * len(train_dataloader) + i + 1 #global batch number
            
            # if one target and y is not in one-hot format, convert it to one-hot encoding
            if args.num_targets == 1:
                if len(y.shape) < 2: 
                    y = y.type(torch.int64)
                    y = torch.zeros(y.size(0), args.num_classes).scatter_(1, y.view(-1, 1), 1.) 
            
            # load dataset on device
            x = x.view(x.shape[0], -1).to(args.device)
            
            if args.task == 'mnist_ctrv':
                x = 1.0 - x
            
            
            y = y.to(args.device)

            # forward pass
            objcaps_len_step, read_x_step, c_step, readout_logits  = model(x)
    
            # compute loss for this batch and append it to training loss
            loss, _ , _ = loss_fn(objcaps_len_step, read_x_step, c_step, readout_logits, x, y, args, writer, global_step)
            
            training_loss += loss.data #* x.size(0) 
            
            # zero out previous gradients and backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # record grad norm and clip to prevent exploding gradients
            if args.record_gradnorm:
                grad_norm = 0
                for name, p in model.named_parameters():
                    grad_norm += p.grad.norm().item() if p.grad is not None else 0
                writer.add_scalar('grad_norm', grad_norm, global_step)
            nn.utils.clip_grad_norm_(model.parameters(), 10)

            # update param
            optimizer.step()

            # end of each batch, update tqdm tracking
            pbar.set_postfix(batch_loss='{:.3f}'.format(loss.item()))
            pbar.update()
    
    # logging training info to tensorboard writer
    train_loss = training_loss / len(train_dataloader.dataset)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    
    return train_loss
    
def train_and_evaluate(model, train_dataloader, val_dataloader, loss_fn, optimizer, writer, args):
    """
    for each epoch:
        - train the model, update param, and log the training loss ('train_epoch')
        - save checkpoint
        - compute and log average val loss/acc and
        - save best model
        
    """
    start_epoch = 1

    if args.restore_file:
        print('Restoring parameters from {}'.format(args.restore_file))
        start_epoch = load_checkpoint(args.restore_file, [model], [optimizer], map_location=args.device.type)
        args.n_epochs += start_epoch
        print('Resuming training from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, args.n_epochs+1):
        
        # train epoch
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, epoch, writer, args)

        # save checkpoint 
        if args.save_checkpoint:
            save_checkpoint({'epoch': epoch,
                             'model_state_dicts': [model.state_dict()],
                             'optimizer_state_dicts': [optimizer.state_dict()]}, 
                            checkpoint=args.log_dir,
                            quiet=True)
        
        # compute validation loss and acc
        if (epoch) % args.validate_after_howmany_epochs == 0:
            val_loss, val_L_recon, val_L_margin, val_acc_partial, val_acc_exact = test(model, val_dataloader, args)
            
            # logging validation info to tensorboard writer
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/L_recon', val_L_recon, epoch)
            writer.add_scalar('Val/L_margin', val_L_margin, epoch)
            writer.add_scalar('Val/Accuracy_partial', val_acc_partial, epoch)
            writer.add_scalar('Val/Accuracy_exact', val_acc_exact, epoch)
                        
            if args.verbose:
                print("==> Epoch %02d: train_loss=%.5f, val_loss=%.5f, val_L_recon=%.5f, val_L_margin=%.5f, val_acc_partial=%.4f,  val_acc_exact=%.4f" \
                  % (epoch, train_loss, val_loss, val_L_recon, val_L_margin, val_acc_partial, val_acc_exact))
               
            # update best validation acc and save best model to output dir
            if (val_acc_exact > args.best_val_acc):  
                args.best_val_acc = val_acc_exact
                torch.save(model.state_dict(), args.log_dir +'/best_model_epoch%d_acc%.4f.pt'% (epoch, val_acc_exact))  #output_dir
                print("the model with best val_acc (%.4f) was saved to disk" % val_acc_exact)

        # for experiments, abort the local mimima trials
        if (epoch) % 100 == 0:
            if hasattr(args, 'abort_if_valacc_below'):
                if (args.best_val_acc < args.abort_if_valacc_below) or math.isnan(val_acc_exact):
                    status = f'===== EXPERIMENT ABORTED: val_acc_exact is {val_acc_exact} at epoch {epoch} (Criterion is {args.abort_if_valacc_below}) ===='
                    writer.add_text('Status', status, epoch)
                    print(status)
                    sys.exit()
                else:
                    status = '==== EXPERIMENT CONTINUE ===='
                    writer.add_text('Status', status, epoch)
                    print(status)

