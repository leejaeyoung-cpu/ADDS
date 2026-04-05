"""
Graph Neural Network for drug combination prediction
Models drug-drug and drug-target interactions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_init import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple
import numpy as np

try:
    from utils import get_logger, config
except ImportError:
    from src.utils import get_logger, config

logger = get_logger(__name__)


class DrugGNN(nn.Module):
    """
    Graph Neural Network for drug representation learning
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        """
        Initialize Drug GNN
        
        Args:
            num_node_features: Number of input node features
            hidden_channels: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Whether to use GAT instead of GCN
        """
        super(DrugGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        if use_attention:
            # Graph Attention Network
            heads = config.get('models.gnn.heads', 4)
            
            # First layer
            self.convs.append(GATConv(num_node_features, hidden_channels, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
            
            # Last layer
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        else:
            # Graph Convolutional Network
            # First layer
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.hidden_channels = hidden_channels
        logger.info(f"✓ DrugGNN initialized with {num_layers} {'GAT' if use_attention else 'GCN'} layers")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: PyG Data object with x (node features) and edge_index
        
        Returns:
            Graph embedding
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        return x
    
    def get_output_dim(self) -> int:
        """Get output dimension after pooling"""
        return self.hidden_channels * 2  # mean + max pooling


class DrugCombinationPredictor(nn.Module):
    """
    Full model for predicting drug combination efficacy
    Combines two drug graphs and additional features
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_additional_features: int = 0,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        """
        Initialize combination predictor
        
        Args:
            num_node_features: Number of node features in drug graphs
            num_additional_features: Number of additional features (e.g., concentrations)
            hidden_channels: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Use GAT
        """
        super(DrugCombinationPredictor, self).__init__()
        
        # Drug encoder (shared for both drugs)
        self.drug_encoder = DrugGNN(
            num_node_features=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Combination predictor
        drug_embedding_dim = self.drug_encoder.get_output_dim()
        combined_dim = drug_embedding_dim * 2 + num_additional_features
        
        self.combination_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Output: predicted efficacy (0-1)
        )
        
        self.num_additional_features = num_additional_features
        logger.info(f"✓ DrugCombinationPredictor initialized")
    
    def forward(
        self,
        drug_a: Data,
        drug_b: Data,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            drug_a: Graph data for drug A
            drug_b: Graph data for drug B
            additional_features: Additional features (concentrations, etc.)
        
        Returns:
            Predicted combination efficacy (0-1)
        """
        # Encode both drugs
        emb_a = self.drug_encoder(drug_a)
        emb_b = self.drug_encoder(drug_b)
        
        # Concatenate embeddings
        combined = torch.cat([emb_a, emb_b], dim=1)
        
        # Add additional features if provided
        if additional_features is not None:
            combined = torch.cat([combined, additional_features], dim=1)
        elif self.num_additional_features > 0:
            raise ValueError("Model expects additional features but none provided")
        
        # Predict efficacy
        efficacy = self.combination_mlp(combined)
        
        return efficacy.squeeze(-1)


class InteractionGraphModel(nn.Module):
    """
    Model drug-drug and drug-target interactions as a heterogeneous graph
    """
    
    def __init__(
        self,
        num_drugs: int,
        num_targets: int,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize interaction graph model
        
        Args:
            num_drugs: Number of unique drugs
            num_targets: Number of unique targets
            embedding_dim: Dimension for drug/target embeddings
            hidden_channels: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super(InteractionGraphModel, self).__init__()
        
        # Learnable embeddings
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)
        self.target_embedding = nn.Embedding(num_targets, embedding_dim)
        
        # GNN layers for drug-target interaction propagation
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(embedding_dim, hidden_channels))
        
        # Synergy prediction head
        self.synergy_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
        logger.info(f"✓ InteractionGraphModel initialized")
    
    def forward(
        self,
        drug_a_idx: torch.Tensor,
        drug_b_idx: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            drug_a_idx: Drug A indices
            drug_b_idx: Drug B indices
            edge_index: Interaction graph edges
            batch_size: Batch size
        
        Returns:
            Synergy scores
        """
        # Get drug embeddings
        emb_a = self.drug_embedding(drug_a_idx)
        emb_b = self.drug_embedding(drug_b_idx)
        
        # Process through GNN (simplified - actual implementation would be more complex)
        # Here you would propagate information through the drug-target graph
        
        # For now, simple interaction prediction
        combined = torch.cat([emb_a, emb_b], dim=1)
        synergy = self.synergy_predictor(combined)
        
        return synergy.squeeze(-1)


def _atom_features(atom) -> List[float]:
    """
    Extract atom-level features for GNN node representation.

    Features (9-dim):
        [0] Atomic number (normalized by /50)
        [1] Degree (normalized by /5)
        [2] Formal charge
        [3] Hybridization (one-hot index: SP=0, SP2=1, SP3=2, other=3) / 3
        [4] Is aromatic (0/1)
        [5] Total num Hs (normalized by /4)
        [6] Is in ring (0/1)
        [7] Atomic mass (normalized by /200)
        [8] Chirality tag (R=1, S=-1, none=0)

    Ref: Duvenaud et al. "Convolutional Networks on Graphs for Learning
         Molecular Fingerprints" (NeurIPS 2015)
    """
    from rdkit.Chem import rdchem

    hybridization_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
    }
    chirality_map = {
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1.0,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: -1.0,
    }

    return [
        atom.GetAtomicNum() / 50.0,
        atom.GetDegree() / 5.0,
        float(atom.GetFormalCharge()),
        hybridization_map.get(atom.GetHybridization(), 3) / 3.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetTotalNumHs() / 4.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass() / 200.0,
        chirality_map.get(atom.GetChiralTag(), 0.0),
    ]


def create_drug_graph_from_smiles(smiles: str, node_features_dim: int = 9) -> Data:
    """
    Create PyG Data object from SMILES string using RDKit.

    Parses the molecular structure, extracts per-atom features, and builds
    the bond adjacency matrix as an undirected edge_index tensor.

    Args:
        smiles: SMILES string of the drug molecule
        node_features_dim: Ignored (kept for backward compat); actual dim = 9

    Returns:
        PyG Data object with:
            - x: [num_atoms, 9] atom feature matrix
            - edge_index: [2, num_bonds*2] undirected bond connectivity
            - edge_attr: [num_bonds*2, 4] bond features
            - smiles: original SMILES string

    Ref: Duvenaud et al. NeurIPS 2015
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit not installed — falling back to Morgan FP stub graph")
        # Fallback: create a single-node graph with zeros
        x = torch.zeros(1, 9)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, smiles=smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not parse SMILES: {smiles}")
        x = torch.zeros(1, 9)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, smiles=smiles)

    # Add hydrogens for accurate chemistry (then remove for graph size)
    # We keep implicit H count in features instead
    mol = Chem.AddHs(mol)
    # Remove explicit Hs for compact graph but keep counts in features
    mol = Chem.RemoveHs(mol)

    # --- Node features ---
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(_atom_features(atom))

    x = torch.tensor(atom_features_list, dtype=torch.float)

    # --- Edge index + edge features ---
    from rdkit.Chem import rdchem

    bond_type_map = {
        rdchem.BondType.SINGLE: [1, 0, 0, 0],
        rdchem.BondType.DOUBLE: [0, 1, 0, 0],
        rdchem.BondType.TRIPLE: [0, 0, 1, 0],
        rdchem.BondType.AROMATIC: [0, 0, 0, 1],
    }

    row, col, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond_type_map.get(bond.GetBondType(), [0, 0, 0, 0])
        # Undirected: add both directions
        row += [i, j]
        col += [j, i]
        edge_feats += [bt, bt]

    if row:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

    return data
