import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pycaret.regression import *
import math
import argparse
import torch.nn as nn
import torch.multiprocessing
import torch.nn.functional as F
from torch_scatter import scatter
import torchdrug
from openbabel import pybel
from torchdrug import models
from torchdrug.transforms import transform
from rdkit import RDLogger

# 禁用 RDKit 日志
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
# 忽略 RDKit 警告
warnings.filterwarnings("ignore")

# Define the device
device ="cpu" # "cuda" if torch.cuda.is_available() else "cpu" # 
device = torch.device(device)


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


def smiles_to_sdf_and_get_coordinates(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("无效的SMILES:", smile)
        return (None,None)

    # mol = Chem.AddHs(mol)
    success = False
    if AllChem.EmbedMolecule(mol) != -1:
        success = True
    # if AllChem.EmbedMolecule(mol) == -1:
    #     return (None,None)

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            success = False
    # AllChem.UFFOptimizeMolecule(mol)
    
    if success:
        conf = mol.GetConformer()
        coordinates = [
            (atom.GetIdx(),atom.GetSymbol(), [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
            for i, atom in enumerate(mol.GetAtoms())]
    else:
        mol_py = pybel.readstring('smi', smile)
        # mol_py.addh()
        mol_py.make3D()
        coordinates = [ (atom.idx - 1, atom.atomicnum, [atom.coords[0], atom.coords[1], atom.coords[2]]) for atom in mol_py.atoms]

    # conf = mol.GetConformer()
    # coordinates = [
    #     (atom.GetIdx(),atom.GetSymbol(), [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
    #     for i, atom in enumerate(mol.GetAtoms())
    # ]

    # writer = Chem.SDWriter(output_sdf_file)
    # writer.write(mol)
    # writer.close()
    
    if coordinates is None:
        print("生成coord 出错:", smile)
        return (None,None)
    else:
        return coordinates, mol
        

def load_protein(seq, pos):
    residue_type = torch.as_tensor(seq)
    num_residue = len(seq)
    residue_feature = torch.zeros((num_residue, 1), dtype=torch.float)
    residue_number = torch.arange(num_residue)
    num_atom = num_residue
    atom2residue = torch.arange(num_residue)
    node_position = torch.as_tensor(pos) # coords
    atom_type = torch.as_tensor([6 for _ in range(num_atom)])# residue_symbol2id 'C':6
    atom_name = torch.as_tensor([torchdrug.data.Protein.atom_name2id["CA"] for _ in range(num_atom)])

    edge_list = torch.as_tensor([[0, 0, 0]])
    bond_type = torch.as_tensor([0])

    protein = torchdrug.data.Protein(edge_list, atom_type, bond_type, num_node=num_atom, num_residue=num_residue,
                            node_position=node_position, atom_name=atom_name,
                            atom2residue=atom2residue, residue_feature=residue_feature, 
                            residue_type=residue_type, residue_number=residue_number)
    return protein

class Pro_Mol_Dataset(Dataset):
    def __init__(self, Mydata,file_path,shuffle=False,transform=None):

        # Mydata=pd.read_csv(data_path)
        self.transform = transform
        npy_dir = os.path.join(file_path, 'coordinates')
        
        # fasta_file = os.path.join(file_path, 'kinase_all_new_.fasta')
        self.smile_mol={}
        self.smiles_Protein_List=[]
        self.labels={}
        self.smile_coords={}

        Protein_List=list(Mydata['ProteinName'].values)
        Kinase_List=list(Mydata['Kinase'].values)
        fasta_file = os.path.join(file_path, f'{Protein_List[0]}.fasta')
        
        # Protein_List=list(shuffled_data['ProteinName'].values)
        # Label_List=list(Mydata['Kd (nM)'].values)
        for i,v in enumerate(list(Mydata['SMILES'].values)):
            v=v.replace(" ", "").replace("\n", "").replace("\t", "")
            # temp_mol_=Chem.MolFromSmiles(v.strip())
            # coords
            (coords,temp_mol_) =smiles_to_sdf_and_get_coordinates(v)#,temp_mol_
            if (coords is None) or (temp_mol_ is  None):
                continue
            v_p=str(v)+'_'+str(Protein_List[i])
            self.smile_coords[v_p]=coords
            self.smile_mol[v_p]=temp_mol_
            self.smiles_Protein_List.append((v,Protein_List[i],Kinase_List[i]))
            self.labels[v_p]=0.0
        
            
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(torchdrug.data.Protein.residue_symbol2id.get(amino, 0))
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.Proteindata = {}
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            protein = load_protein(amino_ids.astype(int), pos)#, **kwargs
            self.Proteindata[protein_name]= protein

        self.Moldats={}
        for (smile,proName,kinaseName) in self.smiles_Protein_List:
            sa_name=str(smile)+'_'+str(proName)
            coords=self.smile_coords[sa_name]
            temp_mol=self.smile_mol[sa_name]
            
            _,_,coords =zip(*coords)
            coords=np.array(coords).astype(np.float32)
            coords=torch.from_numpy(coords)
            # node features
            n_features = [(atom.GetIdx(),atom.GetSymbol(), atom_features(atom)) for atom in temp_mol.GetAtoms()]
            n_features.sort() 
            _,_, n_features = zip(*n_features)
            n_features = torch.stack(n_features)
            
            
            edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in temp_mol.GetBonds()])
            undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list


            fbonds=[]
            for edge in undirected_edge_list:
                a1, a2 = edge.tolist()  
                bond = temp_mol.GetBondBetweenAtoms(a1, a2)  
                fbond=bond_features(bond)
                fbonds.append(list(map(int, fbond)))
            fbonds=np.array(fbonds).astype(np.float32)
            fbonds=torch.from_numpy(fbonds)
            
            undirected_edge_list=undirected_edge_list.T
            label=torch.tensor([self.labels[sa_name]], dtype=torch.float32)
            self.Moldats[sa_name]=Data(x=n_features,edge_index=undirected_edge_list,
                          edge_attr=fbonds,smiles=smile,pron=proName,coords=coords,y=label,kinase=kinaseName)

        
    def __len__(self):
        return len( self.smiles_Protein_List)
    
    def __getitem__(self, index):
        item=self._preprocess(index)
        return item
    def _preprocess(self, index):
        res = {}
        (sm,proName,kiName)=self.smiles_Protein_List[index]
        sa_nama=str(sm)+'_'+str(proName)
        MolDatas=self.Moldats[sa_nama]
        
        protein=self.Proteindata[proName]
        item_tmp = {"graph": protein}
        if self.transform:
            item_tmp = self.transform(item_tmp)
        ProDatas=item_tmp
        res['MolDatas']=MolDatas
        res['ProDatas']=ProDatas
        return res

    def custom_collate(self,batch):
        mol_data_batch = [item['MolDatas'] for item in batch]
        protein_data_batch = [item['ProDatas'] for item in batch]
        mol_batch=Batch.from_data_list(mol_data_batch)
        ProteinDatas = torchdrug.data.dataloader.graph_collate(protein_data_batch)
        return {"MolDatas": mol_batch, "ProteinDatas": ProteinDatas}

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
    d_k = q.size()[-1] # 
    # Only transpose the last 2 dimensions, because the first dimension is the batch size
    # scale the value with square root of d_k which is a constant value
    val_before_softmax = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim = -1) # 
    # Multiply attention matrix with value matrix
    values = torch.matmul(attention, v) # 
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

class projection_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enzy_refine_layer_1 = nn.Linear(1280, 1280) 
        self.smiles_refine_layer_1 = nn.Linear(133, 133) 
        self.enzy_refine_layer_2 = nn.Linear(1280, 128) 
        self.smiles_refine_layer_2 = nn.Linear(133, 128) 
        self.batch_norm_enzy = nn.BatchNorm1d(1280)
        self.batch_norm_smiles = nn.BatchNorm1d(133)
        self.batch_norm_shared = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, enzy_embed, smiles_embed):
        refined_enzy_embed = self.enzy_refine_layer_1(enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_1(smiles_embed)

        refined_enzy_embed = self.batch_norm_enzy(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_smiles(refined_smiles_embed)

        refined_enzy_embed = self.relu(refined_enzy_embed)
        refined_smiles_embed = self.relu(refined_smiles_embed)

        refined_enzy_embed = self.enzy_refine_layer_2(refined_enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_2(refined_smiles_embed)

        refined_enzy_embed = self.batch_norm_shared(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_shared(refined_smiles_embed)
        refined_enzy_embed = torch.nn.functional.normalize(refined_enzy_embed, dim=1)
        refined_smiles_embed = torch.nn.functional.normalize(refined_smiles_embed, dim=1)

        return refined_enzy_embed, refined_smiles_embed
    
class Protein_mol_model(torch.nn.Module):
    def __init__(self, hidden_channels,out_channels=1,num_edge_feats=13,device=device):
        super(Protein_mol_model, self).__init__()


        self.graphencoder=E_GCL(
                input_nf = hidden_channels, 
                output_nf = hidden_channels, 
                hidden_nf = hidden_channels, 
                add_edge_feats = num_edge_feats,
                act_fn = nn.SiLU(), residual =  True, 
                attention = True, normalize = False,
                static_coord = True)
        
        self.graphencoder.load_state_dict(torch.load('/home/xudongguo/Projects/Guo/inhibitor_identification/model_save/molgraph_model_epoch_best.pth',
                                                     map_location=torch.device(device)))
        for parameter in self.graphencoder.parameters():
            parameter.requires_grad = False
        
        self.cross_infor=projection_layer()
        
        model_dir = "/home/xudongguo/Projects/Guo/ESM_PST_model"   # Set the path to your model dir
        self.esm = models.EvolutionaryScaleModeling(model_dir, model="ESM-2-650M", readout="mean")
        # Load ESM-2-650M-S
        self.esm.load_state_dict(torch.load(os.path.join(model_dir, "esm_650m_s.pth"), map_location=torch.device(device)))
        for parameter in self.esm.parameters():
            parameter.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )
        self.dropout=nn.Dropout(0.2)

    def forward(self, batchData):

        res,_,res_graph=self.graphencoder(batchData['MolDatas'].x, batchData['MolDatas'].edge_index, 
                                          batchData['MolDatas'].coords, batchData['MolDatas'].batch,batchData['MolDatas'].edge_attr)

        protein_graph_emb=self.esm(batchData['ProteinDatas']['graph'],batchData['ProteinDatas']['graph'].node_feature.float())
        refined_enzy_embed, refined_smiles_embed=self.cross_infor(protein_graph_emb['graph_feature'],res_graph)
        x=torch.cat((refined_enzy_embed, refined_smiles_embed),dim=1)
        cat_data=self.dropout(x)
        return self.fc(cat_data),x,res_graph,refined_smiles_embed



def read_input (args):

    with open(args.test_smiles, 'r') as file_in:
        smiles_list = file_in.readlines()
    smiles_list = [smile.strip() for smile in smiles_list]
    input_df = pd.DataFrame(smiles_list, columns=['SMILES'])
    input_df['Kinase']= args.kinase
    if args.kinase=='RAF1':
        input_df['ProteinName']= 'P04049'
    elif args.kinase=='BRAF':
        input_df['ProteinName']= 'P15056'
    elif args.kinase=='MAP2K2':
        input_df['ProteinName']= 'P36507'
    elif args.kinase=='MAP2K1':
        input_df['ProteinName']= 'Q02750'
    elif args.kinase=='MAPK1':
        input_df['ProteinName']= 'P28482'
    elif args.kinase=='Mapk3':
        input_df['ProteinName']= 'Q63844'
    elif args.kinase=='MAPK3':
        input_df['ProteinName']= 'P27361'
    elif args.kinase=='Mapk1':
        input_df['ProteinName']= 'P63085'
    return input_df

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Kinhibit:inhibitor-kinase binding affinity prediction.")
    parser.add_argument('--kinase', type=str, required=True, help="kinase")
    parser.add_argument('--test_smiles', type=str, required=True, help="Path to the input file.")
    parser.add_argument('--outputpath', type=str, required=True, help="Path to the output file.",default='/home/xudongguo/Projects/Guo/inhibitor_identification/model_save/')
    # Parse the arguments
    args = parser.parse_args()
    transform_=transform.Compose(
        transforms=[transform.ProteinView(view='residue')])
    # get input data
    pre_data_df=read_input(args)   
    pre_data=Pro_Mol_Dataset(pre_data_df ,'/home/xudongguo/Projects/Guo/inhibitor_identification/data',
                            transform=transform_)
    collate_fn = pre_data.custom_collate
    pre_loader = torchdrug.data.DataLoader(pre_data, batch_size=1,
                                num_workers=4,
                                shuffle=False,
                                collate_fn=collate_fn
                                )

    model=Protein_mol_model(hidden_channels=133,num_edge_feats=13).to(device)
    if args.kinase in ['RAF1','BRAF']:
        model.load_state_dict(torch.load('/home/xudongguo/Projects/Guo/inhibitor_identification/model_save/protein_mol_model/protein_mol_RAF_V3_model_weight_B.pth',map_location=torch.device(device)))
    elif args.kinase in ['MAP2K2','MAP2K1']:
        model.load_state_dict(torch.load('/home/xudongguo/Projects/Guo/inhibitor_identification/model_save/protein_mol_model/protein_mol_MEK_V3_model_weight_B.pth',map_location=torch.device(device)))
    else:
        model.load_state_dict(torch.load('/home/xudongguo/Projects/Guo/inhibitor_identification/model_save/protein_mol_model/protein_mol_ERK_V3_model_weight_B.pth',map_location=torch.device(device)))
    
    
    model.eval()
    SMILES=[]
    Kinases=[]
    ProNs=[]
    embeddings = []
    softmax_func=nn.Softmax(dim=1)
    for batch in pre_loader:
        
        smile=batch['MolDatas'].smiles
        proN=batch['MolDatas'].pron
        kiN=batch['MolDatas'].kinase
        batch['MolDatas']=batch['MolDatas'].to(device)
        batch['ProteinDatas']['graph']=batch['ProteinDatas']['graph'].to(device)
        out,cat_embedding,_,_=model(batch)
        
        soft_output=softmax_func(out)
        y_pred = soft_output.detach().numpy()
        
        SMILES.extend(smile)
        ProNs.extend(proN)
        Kinases.extend(kiN)
        # 提取特征
        embeddings.append(cat_embedding.detach().cpu())


    embeddings_array = np.vstack([emb.squeeze().numpy() for emb in embeddings]) 
    affinity_pre_data = pd.DataFrame(embeddings_array, columns=[f"embedding_{i}" for i in range(embeddings_array.shape[1])])
    affinity_pre_data['SMILES']=SMILES
    affinity_pre_data['Kinase']=Kinases
    affinity_pre_data['Uniprot']=ProNs
    if args.kinase in ['']:
        # 加载模型
        loaded_model = load_model('/home/xudongguo/Projects/Guo/inhibitor_identification/Regressor_results/MEK_model_pipeline_3')
    elif args.kinase in []:
        loaded_model = load_model('/home/xudongguo/Projects/Guo/inhibitor_identification/Regressor_results/RAF_model_pipeline_3')
    else:
        loaded_model = load_model('/home/xudongguo/Projects/Guo/inhibitor_identification/Regressor_results/ERK_model_pipeline_3')

    # 使用加载的模型进行预测
    predictions = predict_model(loaded_model, data=affinity_pre_data)
    pre_res = predictions [['SMILES', 'Kinase','Uniprot','Label']]
    pre_res['Affinity']=pre_res['Label']
    pre_res=pre_res.drop(['Label'],axis=1)
    pre_res.to_csv(args.outputpath+'/results.txt',index=False) 
    
    
    
if __name__ == "__main__":
    main()
    print('Completed')  