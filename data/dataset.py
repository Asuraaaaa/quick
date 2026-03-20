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
        "id_cylinder",
    ]

    # 原始油缸编号 -> 全局油缸域编号
    GLOBAL_CYLINDER_MAPPING = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
    }

    LABEL_MAPPING = {
        "normal": 0,
        "chuanye": 1,
        "in_leak": 2,
        "out_leak": 3,
        "both_leak": 4,
    }

    def __init__(
        self,
        csv_path: str | Path,
        stft_cfg: STFTConfig,
        input_mode: str = "two_channel",
        root_dir: str | Path | None = None,
        target_domain_id: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()

        self.stft_cfg = stft_cfg
        self.input_mode = input_mode
        # 目标域全局编号（0~5），用于识别 target normal 样本
        # 若未显式指定，则尝试从当前 CSV 自动推断（适配多油缸轮换 target 训练）。
        self.target_domain_id = target_domain_id
        if self.target_domain_id is None:
            self.target_domain_id = self._infer_target_domain_id()

        self.root_dir = Path(root_dir) if root_dir is not None else self.csv_path.parent
        self.window = torch.hann_window(stft_cfg.win_length)

    def _validate_columns(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _infer_target_domain_id(self) -> int | None:
        """
        从当前 CSV 自动识别 target 域（仅 normal 类）：
        - 将原始油缸编号映射到全局域编号；
        - 若某个全局域仅出现 normal 标签，则视为 target 域候选；
        - 仅在候选唯一时返回该域，否则返回 None。
        """
        if "label" not in self.df.columns or "id_cylinder" not in self.df.columns:
            return None

        df_tmp = self.df[["label", "id_cylinder"]].copy()
        df_tmp["global_domain"] = df_tmp["id_cylinder"].map(self.GLOBAL_CYLINDER_MAPPING)
        df_tmp = df_tmp.dropna(subset=["global_domain"])

        target_candidates: List[int] = []
        for dom_id, group in df_tmp.groupby("global_domain"):
            labels = set(group["label"].tolist())
            if labels == {"normal"}:
                target_candidates.append(int(dom_id))

        if len(target_candidates) == 1:
            return target_candidates[0]
        return None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, file_path: str) -> Path:
        fp = Path(file_path)
        if fp.is_absolute():
            return fp
        return (self.root_dir / fp).resolve()

    def _compute_stft(self, signal: torch.Tensor) -> torch.Tensor:
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
            spec = stft_complex.abs()
            specs.append(spec)
        spec = torch.stack(specs, dim=0)
        return spec

    def _format_input(self, spec: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.input_mode == "two_channel":
            return {"x": spec}
        if self.input_mode == "split_dual":
            return {"x1": spec[0], "x2": spec[1]}
        if self.input_mode == "concat":
            x = torch.cat([spec[0], spec[1]], dim=-1)
            return {"x": x}
        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def _map_cylinder_to_global_domain(self, raw_cylinder_id: int) -> int:
        if raw_cylinder_id not in self.GLOBAL_CYLINDER_MAPPING:
            raise ValueError(f"Unexpected raw cylinder id: {raw_cylinder_id}")
        return self.GLOBAL_CYLINDER_MAPPING[raw_cylinder_id]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = self._resolve_path(str(row["file_path"]))
        data = np.load(npy_path)

        if data.shape != (2000, 2):
            raise ValueError(
                f"Unexpected npy shape at {npy_path}: {data.shape}, expected (2000, 2)"
            )

        signal = torch.from_numpy(data.astype(np.float32)).transpose(0, 1)
        spec = self._compute_stft(signal)

        sample = self._format_input(spec)

        label_id = self.LABEL_MAPPING[row["label"]]
        global_domain = self._map_cylinder_to_global_domain(int(row["id_cylinder"]))
        is_target_normal = False

        # 若 CSV 已提供显式标记，优先使用；否则按“目标域 + normal 类”规则识别
        if "is_target_normal" in self.df.columns:
            is_target_normal = bool(row["is_target_normal"])
        elif self.target_domain_id is not None:
            is_target_normal = (global_domain == int(self.target_domain_id)) and (label_id == self.LABEL_MAPPING["normal"])

        sample["label"] = torch.tensor(label_id, dtype=torch.long)

        # 保留全局域编号：0~5
        sample["id_cylinder"] = torch.tensor(global_domain, dtype=torch.long)

        # 可选保留原始油缸编号，便于调试或分析
        sample["id_cylinder_raw"] = torch.tensor(int(row["id_cylinder"]), dtype=torch.long)

        sample["phi_in"] = torch.tensor(float(row["phi_in"]), dtype=torch.float32)
        sample["phi_out"] = torch.tensor(float(row["phi_out"]), dtype=torch.float32)
        sample["is_target_normal"] = torch.tensor(is_target_normal, dtype=torch.bool)
        return sample


def build_dataloaders(
    cfg: Dict,
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]

    stft_cfg = STFTConfig(
        n_fft=int(data_cfg["n_fft"]),
        win_length=int(data_cfg["win_length"]),
        hop_length=int(data_cfg["hop_length"]),
        center=bool(data_cfg["stft_center"]),
        normalized=bool(data_cfg["stft_normalized"]),
        pad_mode=str(data_cfg["stft_pad_mode"]),
    )

    train_csv = train_csv if train_csv is not None else data_cfg["train_csv"]
    val_csv = val_csv if val_csv is not None else data_cfg["val_csv"]
    test_csv = test_csv if test_csv is not None else data_cfg["test_csv"]

    common = {
        "stft_cfg": stft_cfg,
        "input_mode": data_cfg["input_mode"],
        "root_dir": data_cfg.get("root_dir", None),
        "target_domain_id": data_cfg.get("target_domain_id", None),
    }

    train_set = FaultDiagnosisDataset(train_csv, **common)
    val_set = FaultDiagnosisDataset(val_csv, **common)
    test_set = FaultDiagnosisDataset(test_csv, **common)

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
    test_loader = _loader(test_set, shuffle=False)

    return train_loader, val_loader, test_loader
