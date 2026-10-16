"""Join DLBS FreeSurfer summary workbooks with MRI-age metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


METADATA_COLUMNS = {
    "s#",
    "subject number",
    "subjectnumber",
    "participant_id",
    "airc_id",
    "constructname",
    "construct name",
    "constructnumber",
    "construct number",
    "epoch",
    "wave",
    "hasdata",
    "has data",
    "numscores",
    "number of scores",
    "global",
}
METADATA_COMPACT = {re.sub(r"[^a-z0-9#]", "", name.lower()) for name in METADATA_COLUMNS}


def subject_key(value: object) -> str:
    """Normalize BIDS IDs (sub-123) and numeric DLBS subject IDs for joining."""
    if pd.isna(value):
        raise ValueError("Found an empty participant identifier.")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", maxsplit=1)[0]
    match = re.search(r"(\d+)\s*$", text)
    if not match:
        raise ValueError(f"Cannot extract a numeric participant key from {value!r}.")
    return str(int(match.group(1)))


def epoch_number(value: object) -> int:
    """Convert epoch labels such as 1, 'Wave 1', or 'Epoch 1' to 1."""
    if pd.isna(value):
        raise ValueError("Found a row with an empty Epoch value.")
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer() and 1 <= int(numeric) <= 3:
        return int(numeric)
    match = re.search(r"([123])\s*$", str(value).strip())
    if not match:
        match = re.search(r"(?:wave|epoch)\s*([123])", str(value), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot map epoch label {value!r} to wave 1, 2, or 3.")
    return int(match.group(1))


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    """Read all Excel sheets or a single delimited file."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if suffix == ".tsv":
        return {path.stem: pd.read_csv(path, sep="\t")}
    if suffix == ".csv":
        return {path.stem: pd.read_csv(path)}
    raise ValueError(f"Unsupported structural file type: {path.suffix}")


def infer_epoch_from_sheet(sheet_name: str) -> int | None:
    match = re.search(r"(?:wave|epoch)\s*([123])", sheet_name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def numeric_features(
    table: pd.DataFrame,
    id_column: str,
    epoch_column: str,
) -> pd.DataFrame:
    """Return numeric morphometry fields, excluding identifiers/status metadata."""
    excluded = {id_column, epoch_column}
    features: dict[str, pd.Series] = {}
    for column in table.columns:
        normalized = re.sub(r"\s+", " ", str(column).strip()).lower()
        compact = re.sub(r"[^a-z0-9#]", "", normalized)
        if column in excluded or normalized in METADATA_COLUMNS or compact in METADATA_COMPACT:
            continue
        if any(token in compact for token in ("age", "participant", "subject", "session", "wave", "epoch", "hasdata", "construct", "numscores")):
            continue
        numeric = pd.to_numeric(table[column], errors="coerce")
        if numeric.notna().any():
            features[str(column).strip()] = numeric
    return pd.DataFrame(features, index=table.index)


def load_structural_features(
    paths: Iterable[Path],
    id_column: str,
    epoch_column: str,
) -> pd.DataFrame:
    """Read and merge feature tables on subject and epoch."""
    merged: pd.DataFrame | None = None
    for path in paths:
        for sheet_name, raw in read_workbook(path).items():
            table = raw.copy()
            table.columns = [str(column).strip() for column in table.columns]
            if id_column not in table.columns:
                raise ValueError(
                    f"{path.name}/{sheet_name}: ID column {id_column!r} not found. "
                    f"Available columns: {', '.join(table.columns[:12])}"
                )

            if epoch_column in table.columns:
                table["_epoch"] = table[epoch_column].map(epoch_number)
            else:
                sheet_epoch = infer_epoch_from_sheet(sheet_name)
                if sheet_epoch is None:
                    raise ValueError(
                        f"{path.name}/{sheet_name}: no {epoch_column!r} column and "
                        "the sheet name does not identify Wave/Epoch 1, 2, or 3."
                    )
                table["_epoch"] = sheet_epoch

            table["_subject_key"] = table[id_column].map(subject_key)
            values = numeric_features(table, id_column, epoch_column)
            if values.empty:
                raise ValueError(f"No numeric MRI feature columns found in {path.name}/{sheet_name}.")
            measure_family = re.sub(r"-W[123]$", "", sheet_name, flags=re.IGNORECASE)
            source_name = f"{path.stem}__{measure_family}"
            values = values.rename(columns={name: f"{source_name}__{name}" for name in values.columns})
            current = pd.concat(
                [table[["_subject_key", "_epoch"]].reset_index(drop=True), values.reset_index(drop=True)],
                axis=1,
            )
            current = current.dropna(subset=["_subject_key"])
            if current.duplicated(["_subject_key", "_epoch"]).any():
                duplicate = current.loc[
                    current.duplicated(["_subject_key", "_epoch"], keep=False),
                    ["_subject_key", "_epoch"],
                ].iloc[0]
                raise ValueError(
                    f"Duplicate participant/epoch row in {path.name}/{sheet_name}: "
                    f"{duplicate['_subject_key']} / {duplicate['_epoch']}"
                )
            if merged is None:
                merged = current
            else:
                shared_features = sorted(
                    (set(merged.columns) & set(current.columns))
                    - {"_subject_key", "_epoch"}
                )
                joined = merged.merge(
                    current,
                    on=["_subject_key", "_epoch"],
                    how="outer",
                    validate="one_to_one",
                    suffixes=("", "__incoming"),
                )
                for column in shared_features:
                    incoming = f"{column}__incoming"
                    joined[column] = joined[column].combine_first(joined[incoming])
                    joined = joined.drop(columns=incoming)
                merged = joined

    if merged is None:
        raise ValueError("Provide at least one structural feature file.")
    return merged


def build_table(
    participants_path: Path,
    structural_paths: Iterable[Path],
    id_column: str,
    epoch_column: str,
) -> pd.DataFrame:
    participants = pd.read_csv(participants_path, sep="\t", na_values=["NA", "xx"])
    required = {"participant_id", "AgeMRI_W1", "AgeMRI_W2", "AgeMRI_W3"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ValueError(f"participants.tsv is missing column(s): {', '.join(missing)}")
    participants = participants[["participant_id", "AgeMRI_W1", "AgeMRI_W2", "AgeMRI_W3"]].copy()
    participants["_subject_key"] = participants["participant_id"].map(subject_key)
    if participants["_subject_key"].duplicated().any():
        raise ValueError("participants.tsv contains duplicated normalized participant IDs.")

    age_table = participants.melt(
        id_vars=["participant_id", "_subject_key"],
        var_name="_age_column",
        value_name="age",
    )
    age_table["age"] = pd.to_numeric(age_table["age"], errors="coerce")
    age_table["_epoch"] = age_table["_age_column"].str.extract(r"W([123])$").astype(int)
    age_table = age_table.dropna(subset=["age"])

    features = load_structural_features(structural_paths, id_column, epoch_column)
    result = age_table.merge(
        features,
        on=["_subject_key", "_epoch"],
        how="inner",
        validate="one_to_one",
    )
    feature_columns = [
        column
        for column in result.columns
        if "__" in column and pd.api.types.is_numeric_dtype(result[column])
    ]
    if not feature_columns:
        raise ValueError("No usable morphometry rows were matched to MRI age values.")
    result = result.dropna(subset=feature_columns, how="all")
    result["wave"] = result["_epoch"].map({1: "wave1", 2: "wave2", 3: "wave3"})
    return result[["participant_id", "wave", "age", *feature_columns]].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True, help="DLBS participants.tsv")
    parser.add_argument(
        "--structural-file",
        type=Path,
        action="append",
        required=True,
        help="A FreeSurfer structural workbook (.xlsx) or exported table (.csv/.tsv); repeat as needed",
    )
    parser.add_argument("--id-column", default="S#", help="Subject ID column in structural files")
    parser.add_argument("--epoch-column", default="Wave", help="Wave/epoch column in structural files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/brain_age_features.csv"),
        help="Destination for the prepared feature table",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepared = build_table(args.participants, args.structural_file, args.id_column, args.epoch_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    print(f"Saved {len(prepared)} participant visits and {prepared.shape[1] - 3} MRI features to {args.output}")
