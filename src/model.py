import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, num_classes, num_prototypes_per_class, embedding_dim=768, use_adaptor=True, temp=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.M = num_classes * num_prototypes_per_class
        self.D = embedding_dim
        self.temp = temp
        
        self.prototypes = nn.Parameter(torch.randn(self.M, self.D))
        
        self.use_adaptor = use_adaptor
        if self.use_adaptor:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.D, nhead=1, batch_first=True) #8
            self.adaptor = nn.TransformerEncoder(encoder_layer, num_layers=1)
            
        self.linear = nn.Linear(self.M, self.num_classes, bias=False)
        self._initialize_linear_layer()
        
        self.register_buffer('data_mean', torch.zeros(1, self.D))
        self.register_buffer('data_std', torch.ones(1, self.D))

    def _initialize_linear_layer(self):
        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            for c in range(self.num_classes):
                start_idx = c * self.num_prototypes_per_class
                end_idx = start_idx + self.num_prototypes_per_class
                self.linear.weight[c, start_idx:end_idx] = 1.0
        
    def set_normalization_stats(self, mean, std):
        self.data_mean.copy_(mean.view(1, -1))
        self.data_std.copy_(std.view(1, -1))
                
    def get_projected_prototypes(self):
        
        if self.use_adaptor:
            p_unsqueezed = self.prototypes.unsqueeze(0)
            z_p = self.adaptor(p_unsqueezed).squeeze(0)
        else:
            z_p = self.prototypes
        return z_p

    def forward(self, z_x):
        # z_x shape: (N, D)
        
        z_p = self.get_projected_prototypes() # (M, D)
        
        dist_sq = torch.cdist(z_x, z_p, p=2.0) ** 2 / self.D# (N, M)
        
        S = torch.exp(-dist_sq) # (N, M)
        
        logits = self.linear(S) # (N, C)
        
        return logits, S, z_p

    def compute_loss(self, logits, z_x, z_p, labels, lambda_weight=0.25):
        
        scaled_logits = logits / self.temp 
        loss_c = F.cross_entropy(scaled_logits, labels)
        #loss_c = F.cross_entropy(logits, labels)
        
        loss_p = 0.0
        
        for c in range(self.num_classes):
            
            mask_c = (labels == c)
            z_xc = z_x[mask_c] # (N_c, D)
            
            if z_xc.shape[0] == 0:
                continue
                
            start_idx = c * self.num_prototypes_per_class
            end_idx = start_idx + self.num_prototypes_per_class
            z_pc = z_p[start_idx:end_idx] # (num_prototypes_per_class, D)
            
            dist_sq_c = torch.cdist(z_xc, z_pc, p=2.0) ** 2 / self.D # (N_c, num_prototypes_per_class)
            
            min_dist_per_prototype, _ = torch.min(dist_sq_c, dim=0) # (num_prototypes_per_class,)
            
            loss_p += min_dist_per_prototype.sum()
            
        loss_p = loss_p / self.M
        
        total_loss = (lambda_weight * loss_c) + ((1.0 - lambda_weight) * loss_p)
        
        return total_loss, loss_c, loss_p
    
    def get_prototype(self, class_idx, n_th, projected=True, denormalize=True):
        
        if class_idx < 0 or class_idx >= self.num_classes:
            raise ValueError(f"class_idx must be between 0 and {self.num_classes - 1}")
        if n_th < 0 or n_th >= self.num_prototypes_per_class:
            raise ValueError(f"n_th must be between 0 and {self.num_prototypes_per_class - 1}")
            
        absolute_idx = (class_idx * self.num_prototypes_per_class) + n_th
        
        if projected:
            all_z_p = self.get_projected_prototypes()
            proto = all_z_p[absolute_idx]
        else:
            proto = self.prototypes[absolute_idx]
        
        if denormalize:
            proto = (proto * self.data_std) + self.data_mean
            
        return proto