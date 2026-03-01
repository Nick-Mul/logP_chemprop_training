import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_mol = rdMolStandardize.FragmentParent(clean_mol)
        return Chem.MolToSmiles(parent_mol)
    except Exception:
        return None


def calculate_descriptors(smiles: str) -> dict:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "SlogP": Descriptors.MolLogP(mol),
            "MolWt": Descriptors.MolWt(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "TPSA": Descriptors.TPSA(mol),
        }
    except Exception:
        return None


def process_dataset(input_path: str, output_npz: str, output_csv: str) -> None:
    df = pd.read_csv(input_path)

    standardized_smiles = []
    logp_values = []
    descriptors_list = []

    for idx, row in df.iterrows():
        std_smiles = standardize_smiles(row["smiles"])
        if std_smiles is None:
            continue

        desc = calculate_descriptors(std_smiles)
        if desc is None:
            continue

        standardized_smiles.append(std_smiles)
        logp_values.append(row["logP"])
        descriptors_list.append(desc)

    smiles_arr = np.array(standardized_smiles, dtype=object)
    logp_arr = np.array(logp_values, dtype=float)
    slogp_arr = np.array([d["SlogP"] for d in descriptors_list], dtype=float)
    mwt_arr = np.array([d["MolWt"] for d in descriptors_list], dtype=float)
    hba_arr = np.array([d["NumHAcceptors"] for d in descriptors_list], dtype=float)
    hbd_arr = np.array([d["NumHDonors"] for d in descriptors_list], dtype=float)
    tpsa_arr = np.array([d["TPSA"] for d in descriptors_list], dtype=float)

    np.savez(
        output_npz,
        smiles=smiles_arr,
        logP=logp_arr,
        SlogP=slogp_arr,
        MWT=mwt_arr,
        HBA=hba_arr,
        HBD=hbd_arr,
        TPSA=tpsa_arr,
    )

    chemprop_df = pd.DataFrame({"smiles": smiles_arr, "logP": logp_arr})
    chemprop_df.to_csv(output_csv, index=False)

    print(f"Processed {len(smiles_arr)} molecules from {input_path}")
    print(f"  NPZ saved: {output_npz}")
    print(f"  CSV saved: {output_csv}")


def main():
    base_path = "/Volumes/External/data/logP"

    datasets = [
        ("train.csv", "train_descriptors.npz", "train_chemprop.csv"),
        ("val.csv", "val_descriptors.npz", "val_chemprop.csv"),
        ("test.csv", "test_descriptors.npz", "test_chemprop.csv"),
    ]

    for csv_name, npz_name, out_csv in datasets:
        input_path = f"{base_path}/{csv_name}"
        process_dataset(input_path, npz_name, out_csv)

    print("\nDone!")


if __name__ == "__main__":
    main()
