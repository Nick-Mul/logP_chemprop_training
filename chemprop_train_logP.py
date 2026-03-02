import atexit
import json
import os
import shutil
from pathlib import Path

from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tqdm import tqdm
from rdkit.rdBase import BlockLogs
from chemprop import data, featurizers, models, nn
from chemprop.utils import make_mol

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToRDKitPhysChem

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.preprocessing import StandardScaler

import pandas as pd

PIPELINE_PHYSCHEM = Pipeline(
    [
        ("auto2mol", AutoToMol()),
        (
            "physchem",
            MolToRDKitPhysChem(
                standardizer=None,
                descriptor_list=[
                    "MolLogP",
                    "TPSA",
                    "NumHAcceptors",
                    "NumHDonors",
                    "MolWt",
                ],
            ),
        ),
    ],
    n_jobs=-1,
)


def standardize_smiles(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_mol = rdMolStandardize.FragmentParent(clean_mol)
        return Chem.MolToSmiles(parent_mol)
    except Exception:
        return None


def calc_descriptors(
    unscaled: pd.DataFrame, scaler: StandardScaler | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, StandardScaler]:
    if not isinstance(unscaled, pd.DataFrame):
        unscaled = pd.DataFrame(unscaled)
    if scaler is None:
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(unscaled), columns=unscaled.columns
        )
        return scaled_df, scaler
    return pd.DataFrame(scaler.transform(unscaled), columns=unscaled.columns), scaler


def process_file(
    input_path: str, smiles_col: str = "smiles", logp_col: str = "logP"
) -> tuple[list[str], list[float]]:
    df = pd.read_csv(input_path)

    results = df[[smiles_col, logp_col]].copy()
    results["standardized_smiles"] = results[smiles_col].apply(standardize_smiles)
    results = results.dropna(subset=["standardized_smiles"])

    return results["standardized_smiles"].tolist(), results[logp_col].tolist()


def train_a_chemprop_model():
    return None


def main():
    base_path = "/Volumes/External/data/logP"
    infile = f"{base_path}/train.csv"
    smis, ys = process_file(infile)
    mols = [make_mol(smi, add_h=True, keep_h=True) for smi in smis]

    x_ds_unscaled = pd.DataFrame(PIPELINE_PHYSCHEM.transform(smis))

    train_indices_list, val_indices_list, test_indices_list = data.make_split_indices(
        mols, "random", (0.8, 0.1, 0.1)
    )
    train_indices = [list(train_indices_list[0])]
    val_indices = [list(val_indices_list[0])]
    test_indices = [list(test_indices_list[0])]

    x_d_scaler = StandardScaler()
    train_x_ds = [x_ds_unscaled.iloc[i].values for i in train_indices[0]]
    x_d_scaler.fit(train_x_ds)

    x_ds, _ = calc_descriptors(x_ds_unscaled, x_d_scaler)
    X_d_transform = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

    all_data = [
        data.MoleculeDatapoint(mol, name=smi, y=np.array([y]), x_d=np.array(x_d))
        for mol, smi, y, x_d in zip(mols, smis, ys, x_ds.values)
    ]

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)
    train_loader = data.build_dataloader(train_dset, num_workers=1)
    val_loader = data.build_dataloader(val_dset, num_workers=1, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=1, shuffle=False)

    mp = nn.BondMessagePassing()
    agg = nn.NormAggregation()
    ffn_input_dim = mp.output_dim + x_ds.shape[1]
    ffn = nn.RegressionFFN(input_dim=ffn_input_dim)
    batch_norm = True
    metric_list = [
        nn.metrics.MSE()
    ]  # Only the first metric is used for training and early stopping

    mpnn = models.MPNN(
        mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform
    )

    checkpointing = ModelCheckpoint(
        "checkpoints",  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=4,  # number of epochs to train for
        callbacks=[checkpointing],  # Use the configured checkpoint callback
    )

    trainer.fit(mpnn, train_loader, val_loader)

    results = trainer.test(
        dataloaders=test_loader, weights_only=False
    )  # weights_only=False is only required with pytorch lightning version 2.6.0 or newer

    print(results)
    exit()

if __name__ == "__main__":
    main()
    
