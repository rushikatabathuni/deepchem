
import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
import csv
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import rdmolfiles, rdmolops, FragmentCatalog
from deepchem.models.torch_models.layers import Encoder



import os
import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import FragmentCatalog, rdmolfiles, rdmolops
from rdkit import RDConfig

# Mapping of atom symbols to numerical values
str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'N': 3, 'O': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P': 9,
           'I': 10, 'Na': 11, 'B': 12, 'Se': 13, 'Si': 14, '<unk>': 15, '<mask>': 16, '<global>': 17}

# Reverse mapping from numerical values to atom symbols
num2str = {i: j for j, i in str2num.items()}

class InferenceDataset(Dataset):
    """
    A PyTorch dataset for handling molecular SMILES data and converting it into numerical representations.
    """

    def __init__(self, sml_list, max_len=500, addH=True):
        """
        Initializes the dataset
        Parameters:
        sml_list (list): A list of SMILES strings representing molecules.
        max_len (int): Maximum allowed length of SMILES strings to include in the dataset.
        addH (bool): Whether to include explicit hydrogen atoms.
        """
        self.vocab = str2num  # Dictionary for atom-to-number conversion
        self.devocab = num2str  # Dictionary for number-to-atom conversion
        self.sml_list = [sml for sml in sml_list if len(sml) < max_len]  # Filter molecules by length
        self.addH = addH  # Store hydrogen inclusion preference

    def numerical_smiles(self, smiles):
        """
        Converts a SMILES string into a numerical representation.
        Parameters:
        smiles (str): A SMILES string representing a molecule.
        Returns:
        tuple: (numerical representation, adjacency matrix, original SMILES, list of atoms)
        """
        smiles_origin = smiles  # Keep original SMILES string
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)  # Convert to atom list and adjacency matrix
        atoms_list = ["<global>"] + atoms_list  # Add global token for transformer-based models

        # Convert atom symbols to numerical indices using the vocabulary
        nums_list = [self.vocab.get(atom, self.vocab['<unk>']) for atom in atoms_list]

        # Create an adjacency matrix with padding
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix  # Copy adjacency matrix into temp
        adjoin_matrix = (1 - temp) * (-1e9)  # Convert to a format suitable for attention masking

        x = np.array(nums_list, dtype=np.int64)  # Convert to NumPy array
        return x, adjoin_matrix, smiles_origin, atoms_list

    def __len__(self):
        """Returns the number of SMILES molecules in the dataset."""
        return len(self.sml_list)

    def __getitem__(self, idx):
        """
        Retrieves the numerical representation of a molecule at a given index.
        Parameters:
        idx (int): Index of the molecule in the dataset.
        Returns:
        tuple: (numerical tensor, adjacency matrix tensor, original SMILES, atom list tensor)
        """
        smiles = self.sml_list[idx]
        x, adjoin_matrix, smiles_origin, atoms_list = self.numerical_smiles(smiles)
        x = torch.tensor(x, dtype=torch.long)  # Convert numerical sequence to PyTorch tensor
        adjoin_matrix = torch.tensor(adjoin_matrix, dtype=torch.float32)  # Convert adjacency matrix to tensor
        smiles = [smiles]  # Keep SMILES as a list
        atoms_list = torch.tensor([self.vocab.get(atom, self.vocab['<unk>']) for atom in atoms_list], dtype=torch.long)  # Convert atom list to tensor

        return x, adjoin_matrix, smiles_origin, atoms_list


def fg_list():
    """
    Retrieves a list of 47 functional groups.
    Returns:
    list: A list of SMILES strings representing functional groups.
    """
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')  # Path to functional groups file
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)  # Load fragment catalog parameters
    fg_list = []
    for i in range(fparams.GetNumFuncGroups()):
        fg_list.append(fparams.GetFuncGroup(i))  # Extract functional groups
    fg_list.pop(27)  # Remove an unwanted group
    # Convert functional groups to SMILES format and add extra groups
    x = [Chem.MolToSmiles(_) for _ in fg_list] + ['*C=C', '*F', '*Cl', '*Br', '*I', '[Na+]', '*P', '*P=O', '*[Se]', '*[Si]']
    y = set(x)  # Remove duplicates
    return list(y)


def smiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    """
    Converts a SMILES string into a list of atoms and an adjacency matrix.
    Parameters:
    smiles (str): The SMILES string of the molecule.
    explicit_hydrogens (bool): Whether to explicitly add hydrogen atoms.
    canonical_atom_order (bool): Whether to reorder atoms canonically.
    Returns:
    tuple: (list of atom symbols, adjacency matrix)
    """
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule object
    if mol is None:
        print('error')  # Print error if SMILES conversion fails
        return None, None
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)  # Add explicit hydrogen atoms
    else:
        mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms
    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)  # Get canonical atom order
        mol = rdmolops.RenumberAtoms(mol, new_order)  # Reorder atoms in the molecule
    num_atoms = mol.GetNumAtoms()  # Get number of atoms in the molecule
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)  # Get atom by index
        atoms_list.append(atom.GetSymbol())  # Store atomic symbol
    # Initialize adjacency matrix (identity matrix of size num_atoms)
    adjoin_matrix = np.eye(num_atoms)
    num_bonds = mol.GetNumBonds()  # Get number of bonds in the molecule

    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)  # Get bond by index
        u = bond.GetBeginAtomIdx()  # Get index of first atom in the bond
        v = bond.GetEndAtomIdx()  # Get index of second atom in the bond
        adjoin_matrix[u, v] = 1.0  # Mark bond presence in adjacency matrix
        adjoin_matrix[v, u] = 1.0  # Ensure symmetry

    return atoms_list, adjoin_matrix  # Return atom list and adjacency matrix


def cover_dict(path):
  """
  Reads a pickle file and converts its contents into a dictionary of PyTorch tensors.
  Args:
      path (str): The file path of the pickle file.
  Returns:
      dict: A dictionary where keys are indices and values are PyTorch tensors.
  """
  file_path = path
  with open(file_path, 'rb') as file:
    data = pickle.load(file)  # Load data from pickle file
  
  # Convert values in the dictionary to PyTorch tensors
  tensor_dict = {key: torch.tensor(value) for key, value in data.items()}
  # Reindex dictionary keys as sequential integers
  new_data = {i: value for i, (key, value) in enumerate(tensor_dict.items())}
  
  return new_data

def score(y_test, y_pred):
  """
  Computes various classification performance metrics.
  Args:
      y_test (array-like): Ground truth binary labels (0 or 1).
      y_pred (array-like): Predicted probabilities or scores.
  Returns:
      tuple: Contains the following metrics:
          - True Positives (tp)
          - True Negatives (tn)
          - False Negatives (fn)
          - False Positives (fp)
          - Sensitivity/Recall (se)
          - Specificity (sp)
          - Matthews Correlation Coefficient (mcc)
          - Accuracy (acc)
          - Area Under the ROC Curve (auc_roc_score)
          - F1 Score (F1)
          - Balanced Accuracy (BA)
          - Area Under Precision-Recall Curve (prauc)
          - Positive Predictive Value (PPV)
          - Negative Predictive Value (NPV)
  """
  auc_roc_score = roc_auc_score(y_test, y_pred)  # Compute AUC-ROC score
  # Compute Precision-Recall curve and PR-AUC
  prec, recall, _ = precision_recall_curve(y_test, y_pred)
  prauc = auc(recall, prec)
  # Convert predicted probabilities to binary values (0 or 1)
  y_pred_print = [round(y, 0) for y in y_pred]

  # Compute confusion matrix values
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred_print).ravel()
  # Compute Sensitivity (Recall)
  se = tp / (tp + fn)
  # Compute Specificity
  sp = tn / (tn + fp)
  # Compute Accuracy
  acc = (tp + tn) / (tp + fn + tn + fp)
  # Compute Matthews Correlation Coefficient (MCC)
  mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
  # Compute Precision (Positive Predictive Value)
  P = tp / (tp + fp)
  # Compute F1 Score
  F1 = (P * se * 2) / (P + se)
  # Compute Balanced Accuracy
  BA = (se + sp) / 2
  # Compute Positive Predictive Value (PPV)
  PPV = tp / (tp + fp)
  # Compute Negative Predictive Value (NPV)
  NPV = tn / (fn + tn)

  return tp, tn, fn, fp, se, sp, mcc, acc, auc_roc_score, F1, BA, prauc, PPV, NPV


def DAR_feature(file_path, column_name):
    """
    Reads an Excel file and extracts, standardizes, and normalizes a specific column.
    Args:
        file_path (str): Path to the Excel file.
        column_name (str): Name of the column to extract and process.
    Returns:
        dict: A dictionary mapping row indices to normalized values as PyTorch tensors.
    """
    df = pd.read_excel(file_path)  # Load Excel file into a DataFrame
    # Extract specified column and reshape it to a 2D array
    column_data = df[column_name].values.reshape(-1, 1)
    
    # Given mean and variance for standardization
    mean_value = 3.86845977
    variance_value = 1.569108443
    std_deviation = variance_value**0.5  # Compute standard deviation
    
    print(column_data, type(column_data))  # Print extracted data for debugging
    
    # Standardize column data
    column_data_standardized = (column_data - mean_value) / std_deviation
    # Normalize data using a specified range transformation
    normalized_data = (column_data_standardized - 0.8) / (12 - 0.8)
    # Convert normalized values into a dictionary with PyTorch tensors
    data_dict = {index: torch.tensor(value, dtype=torch.float32) for index, value in zip(df.index, normalized_data.flatten())}
    
    return data_dict

class PredictModelWrapper(dc.models.TorchModel):
  def __init__(self, model, loss_fn=nn.BCELoss(), learning_rate=0.001, **kwargs):
    # Initialize the TorchModel with given loss function and learning rate
    super().__init__(model, loss=loss_fn, learning_rate=learning_rate, **kwargs)
  def _prepare_batch(self, batch):
    # Extract input features and labels from the batch
    x1 = batch[0]
    x2 = batch[1]
    t1, t2, t3, t4 = batch[2], batch[3], batch[4], batch[5]
    labels = batch[10]  # Extract the labels

  def fit(self, dataset, np_epoch=10, batch_size=32, deterministic=False, metrics=None, callbacks=None, **kwargs):
    # Set the model to training mode
    self.model.train()
    metrics = metrics or []
    callbacks = callbacks or []
    train_losses = []
    # Extract input features from dataset
    inputs = dataset.X
    sml_list1 = df["Payload Isosmiles"].tolist()
    sml_list2 = df["Linker Isosmiles"].tolist()
    t1 = inputs[2]
    t2 = inputs[3]
    t3 = inputs[4]
    t4 = inputs[5]
    labels = dataset.y

    for epoch in range(np_epoch):  # Iterate through epochs
      for i in range(40):  # Process a fixed number of samples per epoch
        epoch_losses = []
        # Prepare input data for current iteration
        x1 = [sml_list1[i]]
        x2 = [sml_list2[i]]
        t1 = Heavy_dict[i]
        t2 = Light_dict[i]
        t3 = Antigen_dict[i]
        t4 = DAR_dict[i].numpy()
        # Convert tensor to correct shape
        t1 = t1.unsqueeze(0)
        t2 = t2.unsqueeze(0)
        t3 = t3.unsqueeze(0)
        t4 = torch.tensor(t4, dtype=torch.float32).view(1,1)
        # Create dataset and dataloaders for inference
        inference_dataset1 = InferenceDataset(sml_list1)
        inference_dataset2 = InferenceDataset(sml_list2)
        train_loader1 = torch.utils.data.DataLoader(inference_dataset1, batch_size=1, shuffle=False)
        train_loader2 = torch.utils.data.DataLoader(inference_dataset2, batch_size=1, shuffle=False)
        # Get next batch from data loaders
        infer1 = next(iter(train_loader1))
        infer2 = next(iter(train_loader2))
        x1 = infer1[0]
        x2 = infer2[0]
        adjoin_matrix1 = infer1[1]
        adjoin_matrix2 = infer2[1]
        # Forward pass through model
        x = self.model(x1=x1, mask1=None, adjoin_matrix1=adjoin_matrix1,
                       x2=x2, mask2=None, adjoin_matrix2=adjoin_matrix2,
                       t1=t1, t2=t2, t3=t3, t4=t4)
        print(x, labels[i])
        # Compute loss
        loss_fn = nn.BCELoss()
        loss = loss_fn(torch.tensor(x, dtype=torch.float32), torch.tensor(labels[i], dtype=torch.float32))
        print(loss)
        print(type(loss))
        
        # Backpropagation and optimizer step
        loss.backward()
        self.optimizer.step()
        epoch_losses.append(loss.item())

      # Compute and store epoch loss
      train_loss = np.mean(epoch_losses)
      train_losses.append(train_loss)
      print(train_losses)
      print(epoch_losses)
    return train_losses

  def predict(self, dataset, np_epoch=10, batch_size=32, deterministic=False, metrics=None, callbacks=None, **kwargs):
    # Set the model to evaluation mode
    self.model.eval()
    metrics = metrics or []
    callbacks = callbacks or []
    train_losses = []
    # Extract input features from dataset
    inputs = dataset.X
    sml_list1 = inputs[0]
    sml_list2 = inputs[1]
    t1 = inputs[2]
    t2 = inputs[3]
    t3 = inputs[4]
    t4 = inputs[5]
    labels = dataset.y
    
    # Prepare inputs for prediction
    x1 = [sml_list1[0]]
    x2 = [sml_list2[0]]
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    t3 = t3.unsqueeze(0)
    t4 = torch.tensor(t4, dtype=torch.float32).view(1,1)

    # Create dataset and dataloaders for inference
    inference_dataset1 = InferenceDataset(sml_list1)
    inference_dataset2 = InferenceDataset(sml_list2)
    train_loader1 = torch.utils.data.DataLoader(inference_dataset1, batch_size=1, shuffle=False)
    train_loader2 = torch.utils.data.DataLoader(inference_dataset2, batch_size=1, shuffle=False)

    # Get next batch from data loaders
    infer1 = next(iter(train_loader1))
    infer2 = next(iter(train_loader2))
    
    x1 = infer1[0]
    x2 = infer2[0]
    adjoin_matrix1 = infer1[1]
    adjoin_matrix2 = infer2[1]
    
    # Forward pass through model
    x = self.model(x1=x1, mask1=None, adjoin_matrix1=adjoin_matrix1,
                   x2=x2, mask2=None, adjoin_matrix2=adjoin_matrix2,
                   t1=t1, t2=t2, t3=t3, t4=t4)
    return x


class PredictModel(nn.Module):
  def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=18, dropout_rate=0.1):
    super().__init__()
    
    # Initialize the encoder with specified parameters
    self.encoder = Encoder(
        num_layers=num_layers, 
        d_model=d_model, 
        num_heads=num_heads, 
        dff=dff, 
        input_vocab_size=vocab_size, 
        maximum_position_encoding=200, 
        rate=dropout_rate
    )

    # Fully connected layers for processing the encoded representation
    self.fc1 = nn.Linear(4353, d_model)  # First linear transformation layer
    self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer to prevent overfitting
    self.fc2 = nn.Linear(d_model, 1)  # Final output layer

  def forward(self, x1, adjoin_matrix1, mask1, x2, adjoin_matrix2, mask2, t1, t2, t3, t4, training=False):
    # Pass x1 through the encoder with attention mechanism
    x1, attention_weights1 = self.encoder(x1, training=training, mask=mask1, adjoin_matrix=adjoin_matrix1)
    print(x1.shape)  # Debugging: Print shape of x1

    x1 = x1[:, 0, :]  # Extract the first token representation

    # Pass x2 through the encoder
    x2, attention_weights2 = self.encoder(x2, training=False, mask=mask2, adjoin_matrix=adjoin_matrix2)
    print(x2.shape)  # Debugging: Print shape of x2

    x2 = x2[:, 0, :]  # Extract the first token representation

    # Concatenate x1 and x2 along with additional feature tensors t1, t2, t3, t4
    x = torch.cat((x1, x2), dim=1)
    x = torch.cat((x, t1), dim=1)
    x = torch.cat((x, t2), dim=1)
    x = torch.cat((x, t3), dim=1)
    x = torch.cat((x, t4), dim=1)
    print(x.shape)  # Debugging: Print shape after concatenation
    # Pass through fully connected layers
    x = self.fc1(x)  # First linear layer
    x = self.dropout1(x)  # Apply dropout
    x = self.fc2(x)  # Second linear layer
    # Apply sigmoid activation to get probability score
    x = torch.sigmoid(x)
    print(x)  # Debugging: Print final output

    # Return binary classification based on threshold
    return 1 if x > 0.5 else 0  # Converts probability into class label (0 or 1)
