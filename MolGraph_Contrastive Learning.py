from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import random
from rdkit.DataStructs import TanimotoSimilarity
import os
import gc
import csv
import time
import yaml
import math
import json
import shutil
from copy import deepcopy
from datetime import datetime

import torch.nn as nn
import torch.multiprocessing
import torch.nn.functional as F
# from torch.optim import AdamW 
import torch.optim as optim
from torch_scatter import scatter
from torch_cluster import radius_graph
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rdkit import RDLogger

# 禁用 RDKit 日志
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
# 忽略 RDKit 警告
warnings.filterwarnings("ignore")

device='cuda'

BOND_FDIM = 14
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    # type of atom (ex. C,N,O), by atomic number, size = 100
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    # number of bonds the atom is involved in, size = 6
    'degree': [0, 1, 2, 3, 4, 5],
    # integer electronic charge assigned to atom, size = 5
    'formal_charge': [-1, -2, 1, 2, 0],
    # chirality: unspecified, tetrahedral CW/CCW, or other, size = 4
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],  # number of bonded hydrogen atoms, size = 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)  # create the list with zeros
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    # feature vector for each atom
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features

    if functional_groups is not None:
        features += functional_groups
    features = np.array(features).astype(np.float32)
    return torch.from_numpy(features)#K)

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            # 0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def smiles_to_sdf_and_get_coordinates(smile,mol):#, output_sdf_file


    if AllChem.EmbedMolecule(mol) == -1:
        return None

    # 优化几何结构
    AllChem.UFFOptimizeMolecule(mol)

    # 提取原子坐标
    conf = mol.GetConformer()
    coordinates = [
        (atom.GetIdx(),atom.GetSymbol(), [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
        for i, atom in enumerate(mol.GetAtoms())
    ]


    return coordinates

class MolDataset(Dataset):
    def __init__(self, data_path,shuffle=True):

        data=pd.read_csv(data_path)
        
        if shuffle:
            shuffled_data = data.sample(frac=1, random_state=42) 
            # random.shuffle(self.self.smiles_List)
            self.smiles_List=list(shuffled_data['Ligand SMILES'].values)
        else:
            self.smiles_List=list(data['Ligand SMILES'].values)

    def __len__(self):
        return len(self.smiles_List)
    
    def __getitem__(self, index):
        # item=self._preprocess(index)
        return self.smiles_List[index]#item

    def collate_fn(self, batch):
        MolDatas=[]
        for smile in batch:
            temp_mol=Chem.MolFromSmiles(smile.strip())
            # coords
            coords =smiles_to_sdf_and_get_coordinates(smile,temp_mol)
            if coords is None:
                # print(smile)
                continue
            _,_,coords =zip(*coords)
            coords=np.array(coords).astype(np.float32)
            coords=torch.from_numpy(coords)
            # node features
            n_features = [(atom.GetIdx(),atom.GetSymbol(), atom_features(atom)) for atom in temp_mol.GetAtoms()]
            n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
            _,_, n_features = zip(*n_features)
            n_features = torch.stack(n_features)
            
            
            edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in temp_mol.GetBonds()])
            undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
            # for atom in temp_mol.GetAtoms():
            #     n_features=atom_features(atom)

            fbonds=[]
            for edge in undirected_edge_list:
                a1, a2 = edge.tolist()  
                bond = temp_mol.GetBondBetweenAtoms(a1, a2)  
                fbond=bond_features(bond)
                fbonds.append(list(map(int, fbond)))
            fbonds=np.array(fbonds).astype(np.float32)
            fbonds=torch.from_numpy(fbonds)
            
            undirected_edge_list=undirected_edge_list.T
            MolDatas.append(Data(x=n_features, edge_index=undirected_edge_list,edge_attr=fbonds,smiles=smile,coords=coords))
        Batch_MolDatas = Batch.from_data_list(MolDatas)
        return Batch_MolDatas
    

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def single_head_attention(q, k, v):
    # attention(q, k, v) = softmax(qK.T/sqrt(dk)V)
    d_k = q.size()[-1] # 64
    # Only transpose the last 2 dimensions, because the first dimension is the batch size
    # scale the value with square root of d_k which is a constant value
    val_before_softmax = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim = -1) # 200 x 200
    # Multiply attention matrix with value matrix
    values = torch.matmul(attention, v) # 200 x 64
    return values, attention

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    Reference: EGNN: V. G. Satorras et al., https://arxiv.org/abs/2102.09844 
    """
    def __init__(
        self, input_nf, output_nf, hidden_nf, add_edge_feats=0, act_fn=nn.SiLU(), 
        residual=True, attention=False, normalize=False, coords_agg='mean', static_coord = True
    ):
        '''
        :param intput_nf: Number of input node features
        :param output_nf: Number of output node features
        :param hidden_nf: Number of hidden node features
        :param add_edge_feats: Number of additional edge feature
        :param act_fn: Activation function
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
        :param coords_agg: aggregation function
        '''
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.static_coord = static_coord
        self.coords_agg = coords_agg
        # Number of features used to describe the relative positions between nodes
        # Because we're using radial distance, so dimension = 1
        edge_coords_nf = 1
        # input_edge stores the node values, one edge connects two nodes, so * 2
        input_edge = input_nf * 2
        # Prevents division by zeroes, numerical stability purpose
        self.epsilon = 1e-8
        # mlp operation for edges
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + add_edge_feats, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        # mlp operation for nodes
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        # Initializes layer weights using xavier uniform initialization
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # Update coodinates
        if not static_coord:
            # coordinates mlp sequntial layers
            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
        
        # attention mlp layer
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            
    def edge_model(self, source, target, radial, edge_feats):
        # concatenation of edge features
        if edge_feats is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            # Dimension analysis:
            # eg. source, target -> (num_edges, num_node_features)
            # radial -> (num_edges, 1)
            # edge_feats -> (num_edges, 3)
            # out -> (num_edges, num_node_features*2 + 1 + 3)
            out = torch.cat([source, target, radial, edge_feats], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_feats, node_attr = None):
        # Dimension analysis:
        # x -> (num_nodes, num_node_features)
        # edge_index -> (2, num_edges)
        # edge_feats -> (num_edges, num_edge_feats)

        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # unsorted_segment_sum sums up all edge features for each node
        # agg dimension -> (num_nodes, num_edge_feats)
        agg = unsorted_segment_sum(edge_feats, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    # Coordinates update function, this calculation is not used when static_coord is True
    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # calculate coordinate difference between node
        coord_diff = coord[row] - coord[col]
        # calculate the radial distance for each pair of node
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        # normalization
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord,batch, add_edge_feats=None, node_attr=None):
        # unpacks source and target nodes from edge_index
        row, col = edge_index
        # calculate radial distances for each pair of node
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # print(radial,radial.shape, coord_diff,coord_diff.shape)
        # Compute edge features
        edges = self.edge_model(h[row], h[col], radial, add_edge_feats)

        # Update coordinates
        if not self.static_coord:
            coord = self.coord_model(coord, edge_index, coord_diff, add_edge_feats)
            
        # Update node features
        h, agg = self.node_model(h, edge_index, edges, node_attr)
        h_scatter = scatter(h, batch, dim=0, reduce='add')
        return h, add_edge_feats,h_scatter
    

def calc_mol_sim(smile1,smile2):
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    if not mol1 or not mol2:
        raise ValueError("Invalid SMILES string(s) provided.")

    # Generate fingerprints for the molecules
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

    # Calculate Tanimoto similarity
    similarity = TanimotoSimilarity(fp1, fp2)
    return similarity


class MultiSupConLoss1(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, c_treshold=0.5):
        super(MultiSupConLoss1, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.c_treshold = c_treshold

    def forward(self, features, smiles=None, mask=None, multi=True):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if smiles is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif smiles is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
            multi_val = torch.ones_like(mask).to(device)
        elif smiles is not None:

            if len(smiles) != batch_size:
                raise ValueError('Num of smiles does not match num of features')
            smiles_sim = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=device)
            for x in range(batch_size):
                for y in range(batch_size):
                    smiles_sim[x, y] = calc_mol_sim(smiles[x], smiles[y])
            mask = torch.where(smiles_sim >= self.c_treshold, 1., 0.) 

            multi_val = smiles_sim 

        else:
            mask = mask.float().to(device)
            multi_val = torch.ones_like(mask).to(device)

        contrast_count = features.shape[1] 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), (self.temperature + 1e-8) )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask and multi_val as per the highlighted concern
        mask = mask.repeat(anchor_count, contrast_count)
        # multi_val = multi_val.repeat(anchor_count, contrast_count)

        # Mask-out self-contrast cases correctly as per your instructions
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask


        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # multi_log_prob = log_prob * multi_val#

        #print("Contains NaN:", torch.isnan(multi_labels).any().item())

        mean_multi_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = - (self.temperature / self.base_temperature) * mean_multi_log_prob_pos

        loss = loss.view(anchor_count, batch_size)

        return loss.mean()/1000
    
    
def main():    
    
    model_dir='./model_save'
    hidden_channels=133
    num_edge_feats=13
    model=E_GCL(
                input_nf = hidden_channels, 
                output_nf = hidden_channels, 
                hidden_nf = hidden_channels, 
                add_edge_feats = num_edge_feats,
                act_fn = nn.SiLU(), residual =  True, 
                attention = True, normalize = False,
                static_coord = True).to(device)
    print("model:",model)
    data_path='./Dataset.csv'
    train_data = MolDataset(data_path,shuffle=True)
    collate_fn = train_data.collate_fn
    train_loader = DataLoader(train_data, batch_size=128,
                                    num_workers=8,
                                    shuffle=True,
                                    collate_fn=collate_fn
                                    )#
    # Loss function
    temperature = 0.07
    contrast_mode = 'all'
    base_temperature = 0.07
    c_treshold = 0.7
    lr=1e-4
    criterion = MultiSupConLoss1(temperature=temperature, contrast_mode=contrast_mode,
                                    base_temperature=base_temperature, c_treshold=c_treshold).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    epochs = 50
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx,batchData in enumerate(train_loader):
            batchData=batchData.to(device)
            # print(batchData)
            optimizer.zero_grad()
            res,_,res_graph=model(batchData.x, batchData.edge_index, batchData.coords, batchData.batch,batchData.edge_attr)
            res_graph=res_graph.unsqueeze(1)
            loss =criterion(res_graph,batchData.smiles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')
        val_loss=epoch_loss / len(train_loader)
        # early_stopping 
        if len(val_losses)!=0 and min(val_losses) > val_loss:
            val_losses.clear() 
            val_losses.append(val_loss)
            torch.save(model.state_dict(),model_dir + '/molgraph_model_epoch_best.pth')
            
        else:
            val_losses.append(val_loss)
        if len(val_losses) > 10 and min(val_losses) < val_loss:
            print("Best val_loss: {}".format(round(min(val_losses),6)))
        torch.save(model.state_dict(),model_dir + f'/molgraph_model_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
