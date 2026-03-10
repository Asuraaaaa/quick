from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class STFTConfig:
    n_fft: int
    win_length: int
    hop_length: int
    center: bool
    normalized: bool
    pad_mode: str


class FaultDiagnosisDataset(Dataset):
    REQUIRED_COLUMNS = [
        "file_path",
        "label",
        "phi_in",
        "phi_out",
    ]

    def __init__(
        self,
        csv_path: str | Path,
        stft_cfg: STFTConfig,
        input_mode: str = "two_channel",
        root_dir: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()

        self.stft_cfg = stft_cfg
        self.input_mode = input_mode
        self.root_dir = Path(root_dir) if root_dir is not None else self.csv_path.parent

        self.window = torch.hann_window(stft_cfg.win_length)

    def _validate_columns(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, file_path: str) -> Path:
        fp = Path(file_path)
        if fp.is_absolute():
            return fp
        return (self.root_dir / fp).resolve()

    def _compute_stft(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: [2, 2000]
        specs: List[torch.Tensor] = []
        for c in range(signal.shape[0]):
            stft_complex = torch.stft(
                signal[c],
                n_fft=self.stft_cfg.n_fft,
                hop_length=self.stft_cfg.hop_length,
                win_length=self.stft_cfg.win_length,
                window=self.window,
                center=self.stft_cfg.center,
                normalized=self.stft_cfg.normalized,
                pad_mode=self.stft_cfg.pad_mode,
                return_complex=True,
            )
            # [freq, time] -> magnitude
            spec = stft_complex.abs()
            specs.append(spec)
        spec = torch.stack(specs, dim=0)  # [2, 192, 63]
        return spec

    def _format_input(self, spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.input_mode == "two_channel":
            return {"x": spec}
        if self.input_mode == "split_dual":
            return {"x1": spec[0], "x2": spec[1]}
        if self.input_mode == "concat":
            x = torch.cat([spec[0], spec[1]], dim=-1)  # [192, 126]
            return {"x": x}
        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = self._resolve_path(str(row["file_path"]))
        data = np.load(npy_path)  # expected [2000, 2]
        if data.shape != (2000, 2):
            raise ValueError(f"Unexpected npy shape at {npy_path}: {data.shape}, expected (2000, 2)")

        signal = torch.from_numpy(data.astype(np.float32)).transpose(0, 1)  # [2, 2000]
        spec = self._compute_stft(signal)

        sample = self._format_input(spec)
        sample["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        sample["phi_in"] = torch.tensor(float(row["phi_in"]), dtype=torch.float32)
        sample["phi_out"] = torch.tensor(float(row["phi_out"]), dtype=torch.float32)
        return sample


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    stft_cfg = STFTConfig(
        n_fft=int(data_cfg["n_fft"]),
        win_length=int(data_cfg["win_length"]),
        hop_length=int(data_cfg["hop_length"]),
        center=bool(data_cfg["stft_center"]),
        normalized=bool(data_cfg["stft_normalized"]),
        pad_mode=str(data_cfg["stft_pad_mode"]),
    )

    common = {
        "stft_cfg": stft_cfg,
        "input_mode": data_cfg["input_mode"],
    }

    train_set = FaultDiagnosisDataset(data_cfg["train_csv"], **common)
    val_set = FaultDiagnosisDataset(data_cfg["val_csv"], **common)
    test_set = None
    if data_cfg.get("test_csv"):
        test_set = FaultDiagnosisDataset(data_cfg["test_csv"], **common)

    def _loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=bool(data_cfg["pin_memory"]),
            drop_last=False,
        )

    train_loader = _loader(train_set, shuffle=True)
    val_loader = _loader(val_set, shuffle=False)
    test_loader = _loader(test_set, shuffle=False) if test_set is not None else None

    return train_loader, val_loader, test_loader
