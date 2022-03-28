from typing import Union, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, PReLU
import torch_geometric
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    CGConv,
)

#GNN model
class DOSpredict(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=3,
        batch_norm="True",
        batch_track_stats="True",
        dropout_rate=0.0,
        **kwargs
    ):
        super(DOSpredict, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            self.gc_dim = data.num_features
        else:
            self.gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])
        
        ##Set up pre-GNN dense layers 
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = Sequential(torch.nn.Linear(data.num_features, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
                else:
                    lin = Sequential(torch.nn.Linear(dim1, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GC_block(self.gc_dim, data.num_edge_features, aggr="mean")
            #conv = CGConv(self.gc_dim, data.num_edge_features, aggr="mean", batch_norm=False)            
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = BatchNorm1d(self.gc_dim, track_running_stats=self.batch_track_stats, affine=True)
                self.bn_list.append(bn)
        
        self.dos_mlp = Sequential(Linear(post_fc_dim, dim2), 
                              torch.nn.PReLU(),   
                              Linear(dim2, output_dim),                               
                              torch.nn.PReLU(),                          
                              )   
 
        self.scaling_mlp = Sequential(Linear(post_fc_dim, dim2), 
                              torch.nn.PReLU(),  
                              Linear(dim2, 1),                     
                              )    
                      
  
    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
            else:
                out = self.pre_lin_list[i](out)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)    
                            
        out = F.dropout(out, p=self.dropout_rate, training=self.training)              
        ##Post-GNN dense layers       
        dos_out = self.dos_mlp(out)        
        scaling = self.scaling_mlp(out)                
                                         
        if dos_out.shape[1] == 1:
            return dos_out.view(-1), scaling.view(-1)
        else:
            return dos_out, scaling.view(-1)

# Smooth Overlap of Atomic Positions with neural network
class SOAP_DOS(torch.nn.Module):
    def __init__(self, data, dim1, fc_count,  **kwargs):
        super(SOAP_DOS, self).__init__()

        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_SOAP.shape[1], dim1)

        self.lin_list_dos = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin_out_dos = torch.nn.Linear(dim1, output_dim)

        self.lin_list_scaling = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin_out_scaling = torch.nn.Linear(dim1, 1)        

    def forward(self, data):

        dos_out = F.relu(self.lin1(data.extra_features_SOAP))
        scaling = F.relu(self.lin1(data.extra_features_SOAP))
        
        for layer in self.lin_list_dos:
            dos_out = F.relu(layer(dos_out))
        dos_out = self.lin_out_dos(dos_out)        

        for layer in self.lin_list_scaling:
            scaling = F.relu(layer(scaling))
        scaling = self.lin_out_scaling(scaling)     

        if dos_out.shape[1] == 1:
            return dos_out.view(-1), scaling.view(-1)
        else:
            return dos_out, scaling.view(-1)

# Local Many Body Tensor with neural network
class LMBTR_DOS(torch.nn.Module):
    def __init__(self, data, dim1, fc_count,  **kwargs):
        super(LMBTR_DOS, self).__init__()

        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])
        
        self.lin1 = torch.nn.Linear(data[0].extra_features_LMBTR.shape[1], dim1)

        self.lin_list_dos = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin_out_dos = torch.nn.Linear(dim1, output_dim)

        self.lin_list_scaling = torch.nn.ModuleList(
            [torch.nn.Linear(dim1, dim1) for i in range(fc_count)]
        )

        self.lin_out_scaling = torch.nn.Linear(dim1, 1)        

    def forward(self, data):

        dos_out = F.relu(self.lin1(data.extra_features_LMBTR))
        scaling = F.relu(self.lin1(data.extra_features_LMBTR))
        
        for layer in self.lin_list_dos:
            dos_out = F.relu(layer(dos_out))
        dos_out = self.lin_out_dos(dos_out)        

        for layer in self.lin_list_scaling:
            scaling = F.relu(layer(scaling))
        scaling = self.lin_out_scaling(scaling)  
             
        if dos_out.shape[1] == 1:
            return dos_out.view(-1), scaling.view(-1)
        else:
            return dos_out, scaling.view(-1)

#Dummy model
class Dummy(torch.nn.Module):
    def __init__(
        self,
        data,
        **kwargs
    ):
        super(Dummy, self).__init__()

        self.lin = torch.nn.Linear(len(data[0].x[0]), len(data[0].y[0])) 
          
    def forward(self, data):
        
        out = self.lin(data.x)*0                    
        return out, torch.ones(out.shape[0]).to(out)
            
#####################################################
class GC_block(MessagePassing):

    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0, aggr: str = 'mean', **kwargs):
        super(GC_block, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)
              
        self.mlp = Sequential(Linear(sum(channels) + dim, channels[1]), 
                              torch.nn.PReLU(),
                              )    
        self.mlp2 = Sequential(Linear(dim, dim), 
                              torch.nn.PReLU(),
                              )  
                                                                                              
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out += x[1]
        return out


    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, self.mlp2(edge_attr)], dim=-1)
        z = self.mlp(z)
        return z
