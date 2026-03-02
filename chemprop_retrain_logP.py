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

    results = df[[smiles_col, logp_col]].copy()
    results["standardized_smiles"] = results[smiles_col].apply(standardize_smiles)
    results = results.dropna(subset=["standardized_smiles"])

    return results["standardized_smiles"].tolist(), results[logp_col].tolist()


def load_checkpoint_for_retraining(checkpoint_path: str):
    mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
    return mpnn


def freeze_message_passing_layers(mpnn: models.MPNN):
    for name, param in mpnn.named_parameters():
        if name.startswith("message_passing"):
            param.requires_grad = False


def get_original_learning_rate(checkpoint_path: str) -> float:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if "init_lr" in hparams:
        return hparams["init_lr"]
    if "learning_rate" in hparams:
        return hparams["learning_rate"]
    return 1e-4


def main():
    parser = argparse.ArgumentParser(
        description="Retrain a chemprop model with additional data"
    )
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to existing model checkpoint"
    )
    parser.add_argument(
        "--additional-data",
        required=True,
        help="Path to CSV with additional training data",
    )
    parser.add_argument(
        "--output-dir",
        default="retrain_checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of epochs to train"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation set fraction"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test set fraction"
    )
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        default=0.1,
        help="Learning rate multiplier (default: 0.1 = 10x lower)",
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="SMILES column name in additional data",
    )
    parser.add_argument(
        "--target-col",
        default="logP",
        help="Target column name in additional data",
    )

    args = parser.parse_args()

    print(f"Loading original data from {ORIGINAL_DATA_PATH}...")
    orig_smis, orig_ys = process_file(ORIGINAL_DATA_PATH)
    print(f"  Loaded {len(orig_smis)} original molecules")

    print(f"Loading additional data from {args.additional_data}...")
    add_smis, add_ys = process_file(
        args.additional_data, smiles_col=args.smiles_col, logp_col=args.target_col
    )
    print(f"  Loaded {len(add_smis)} additional molecules")

    all_smis = orig_smis + add_smis
    all_ys = orig_ys + add_ys
    print(f"Combined dataset: {len(all_smis)} molecules")

    print("Computing molecular descriptors...")
    mols = [make_mol(smi, add_h=True, keep_h=True) for smi in all_smis]
    x_ds_unscaled = pd.DataFrame(PIPELINE_PHYSCHEM.transform(all_smis))

    train_frac = 1.0 - args.val_split - args.test_split
    train_indices_list, val_indices_list, test_indices_list = data.make_split_indices(
        mols, "random", (train_frac, args.val_split, args.test_split)
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
    train_loader = data.build_dataloader(train_dset, num_workers=1)
    val_loader = data.build_dataloader(val_dset, num_workers=1, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=1, shuffle=False)

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    mpnn = load_checkpoint_for_retraining(args.checkpoint_path)
    mpnn.X_d_transform = X_d_transform

    original_lr = get_original_learning_rate(args.checkpoint_path)
    new_lr = original_lr * args.lr_multiplier
    print(f"Original LR: {original_lr}, New LR: {new_lr}")

    print("Freezing message passing layers...")
    freeze_message_passing_layers(mpnn)

    trainable_params = [p for p in mpnn.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")

    new_optimizer = torch.optim.Adam(trainable_params, lr=new_lr)

    def new_configure_optimizers():
        return {"optimizer": new_optimizer}

    mpnn.configure_optimizers = new_configure_optimizers

    os.makedirs(args.output_dir, exist_ok=True)

    checkpointing = ModelCheckpoint(
        args.output_dir,
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
        max_epochs=args.epochs,
        callbacks=[checkpointing],
    )

    print(f"Starting training for {args.epochs} epochs...")
    trainer.fit(mpnn, train_loader, val_loader)

    results = trainer.test(dataloaders=test_loader, weights_only=False)

    print(results)


if __name__ == "__main__":
    main()
