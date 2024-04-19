import torch
import torch.nn as nn
import numpy as np
class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))
    
    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)

class MaxMinScaler(torch.nn.Module):    
    def __init__(self, min_val=0, max_val=5):    
        super(MaxMinScaler, self).__init__()    
        self.min_val = min_val    
        self.max_val = max_val      
      
    def forward(self, x):    
        return (x - self.min_val) / (self.max_val - self.min_val)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim,cls_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, cls_dim, bias=False))
        # self.last_layer.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = nn.functional.normalize(x, dim=-1, p=2)
        weights = self.mlp(x)
        # beta = self.last_layer(x)
        # x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        # logits = self.last_layer(x)
        return weights

class MLP(nn.Module):
    def __init__(self, in_dim,out_dim, bottleneck_dim=1024, use_bn=False,
                 nlayers=3, hidden_dim=2048):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.bais = nn.Sequential(*[
            nn.Linear(in_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, bottleneck_dim),
        ])
        self.apply(self._init_weights)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x,is_bais:bool=True):
        logits = self.mlp(x)
        if is_bais:
            bais = self.bais(x)
            return logits,bais
        return logits

class SENet(nn.Module):
    def __init__(self,backbone,head:DINOHead,query_embedding:DINOHead,key_embedding:DINOHead,cls_head:MLP,args=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.query_embedding = query_embedding
        self.key_embedding = key_embedding
        self.cls_head = cls_head
        self.args = args
        # self.samples = samples
        # self.base_labels = samples_labels
        # self.thres = AdaptiveSoftThreshold(1)
        self.base_N = args.mlp_out_dim
        self.base = torch.empty((self.base_N, args.feat_dim)).to(args.device)   
    
    def forward(self,x,is_train=True):
        x = self.backbone(x)
        x_proj,_ = self.head(x)
        temp = x.detach()
        query_embedding,_ = self.query_embedding(x)
        key_embedding,_ = self.key_embedding(self.base)
        Coff = torch.matmul(query_embedding, key_embedding.T)
        Coff = torch.nn.functional.normalize(Coff,dim=1)
        coff = torch.nn.functional.softmax(Coff/self.args.temperature,dim=1)
        if self.args.is_head:
            query_embedding = torch.nn.functional.normalize(query_embedding,dim=1)
            logits = self.cls_head(query_embedding)
        else:
            x = torch.nn.functional.normalize(x,dim=1)
            logits = self.cls_head(x)
        if not is_train:
            return logits
        else:
            return temp,coff,x_proj,logits

class DINO(nn.Module):
    def __init__(self,backbone,head:DINOHead,query_embedding:DINOHead,key_embedding:DINOHead,args=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.query_embedding = query_embedding
        self.key_embedding = key_embedding
        self.args = args
        # self.samples = samples
        # self.base_labels = samples_labels
        # self.thres = AdaptiveSoftThreshold(1)
        self.base_N = args.mlp_out_dim
        self.base = torch.empty((self.base_N, args.feat_dim)).to(args.device)
        
    def forward(self,x):
        x = self.backbone(x)
        x_proj,logits = self.head(x)
        return x,x_proj,logits
    
    def senet(self,x:torch.Tensor):
        query_embedding,_ = self.query_embedding(x)
        # query_embedding = self.scaler(query_embedding)
        key_embedding,_ = self.key_embedding(self.base)
        Coff = torch.matmul(query_embedding, key_embedding.T)
        Coff = torch.nn.functional.normalize(Coff,dim=1)/self.args.temperature
        # coff = coff-torch.max(coff,dim=1,keepdim=True)[0]
        coff = torch.nn.functional.softmax(Coff,dim=1)
        return coff

