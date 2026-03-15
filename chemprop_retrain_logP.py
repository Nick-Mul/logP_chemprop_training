import argparse
import json
import os
from pathlib import Path

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from rdkit.rdBase import BlockLogs
from chemprop import data, featurizers, models, nn
from chemprop.utils import make_mol

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToRDKitPhysChem

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.preprocessing import StandardScaler
import sys
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

ORIGINAL_DATA_PATH = "/Volumes/External/data/logP/train.csv"


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

    # Select only the columns we need and ensure they exist
    if smiles_col not in df.columns or logp_col not in df.columns:
        raise ValueError(
            f"Required columns '{smiles_col}' and/or '{logp_col}' not found in data"
        )

    # Ensure we're working with a DataFrame
    results = df[[smiles_col, logp_col]].copy()
    results["standardized_smiles"] = [
        standardize_smiles(str(x)) if pd.notnull(x) else None
        for x in results[smiles_col]
    ]
    results = results.dropna(subset=["standardized_smiles"])

    return results["standardized_smiles"].tolist(), [
        float(y) for y in results[logp_col]
    ]


def load_checkpoint_for_retraining(checkpoint_path: str):
    mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
    return mpnn


def freeze_message_passing_layers(mpnn: models.MPNN):
    for name, param in mpnn.named_parameters():
        if name.startswith("message_passing"):
            param.requires_grad = False


additional_data = "/Volumes/External/data/logP/logP_without_overlap.csv"
smiles_col = "smiles"
target_col = "logP"
checkpoint = "/retrain_checkpoints/last.ckpt"
original_lr = 1e-5
output_dir = "out"
epochs = 50


def main():
    import argparse

    print(f"Loading original data from {ORIGINAL_DATA_PATH}...")
    orig_smis, orig_ys = process_file(ORIGINAL_DATA_PATH)
    print(f"  Loaded {len(orig_smis)} original molecules")

    add_smis, add_ys = process_file(
        additional_data, smiles_col=smiles_col, logp_col=target_col
    )
    print(f"  Loaded {len(add_smis)} additional molecules")

    all_smis = orig_smis + add_smis
    all_ys = orig_ys + add_ys
    print(f"Combined dataset: {len(all_smis)} molecules")

    print("Computing molecular descriptors...")
    mols = [make_mol(smi, add_h=True, keep_h=True) for smi in all_smis]
    x_ds_unscaled = pd.DataFrame(PIPELINE_PHYSCHEM.transform(all_smis))

    train_frac = 1.0 - 0.1 - 0.1
    train_indices_list, val_indices_list, test_indices_list = data.make_split_indices(
         mols, "random", (train_frac, 0.1, 0.1)
     )
    train_indices = [list(train_indices_list[0])]
    val_indices = [list(val_indices_list[0])]
    test_indices = [list(test_indices_list[0])]

    print(
         f"Split: {len(train_indices[0])} train, {len(val_indices[0])} val, {len(test_indices[0])} test"
     )

    x_d_scaler = StandardScaler()
    train_x_ds = [x_ds_unscaled.iloc[i].values for i in train_indices[0]]
    x_d_scaler.fit(train_x_ds)

    x_ds, _ = calc_descriptors(x_ds_unscaled, x_d_scaler)
    X_d_transform = nn.ScaleTransform.from_standard_scaler(x_d_scaler)

    all_data = [
         data.MoleculeDatapoint(mol, name=smi, y=np.array([y]), x_d=np.array(x_d))
         for mol, smi, y, x_d in zip(mols, all_smis, all_ys, x_ds.values)
     ]

    train_data, val_data, test_data = data.split_data_by_indices(
         all_data, train_indices, val_indices, test_indices
     )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)
    train_loader = data.build_dataloader(train_dset, num_workers=0)
    val_loader = data.build_dataloader(val_dset, num_workers=0, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=0, shuffle=False)

    print(f"Loading checkpoint from {checkpoint}...")
    mpnn = load_checkpoint_for_retraining(checkpoint)
    mpnn.X_d_transform = X_d_transform

    new_lr = original_lr * 0.1
    print(f"Original LR: {original_lr}, New LR: {new_lr}")

    print("Freezing message passing layers...")
    freeze_message_passing_layers(mpnn)

    trainable_params = [p for p in mpnn.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")

    new_optimizer = torch.optim.Adam(trainable_params, lr=new_lr)

    def new_configure_optimizers():
        return {"optimizer": new_optimizer}

    mpnn.configure_optimizers = new_configure_optimizers

    os.makedirs(output_dir, exist_ok=True)

    checkpointing = ModelCheckpoint(
        output_dir,
        "best-{epoch}-{val_loss:.2f}",
        "val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=50,
        callbacks=[checkpointing],
    )

    print(f"Starting training for {epochs} epochs...")
    trainer.fit(mpnn, train_loader, val_loader)

    results = trainer.test(dataloaders=test_loader, weights_only=False)

    print(results)


if __name__ == "__main__":
    main()
