# Coal Mine Hydraulic Support Fault Diagnosis (PyTorch)

## Assumptions
1. `label` has 5 classes (`0~4`), and weak-supervision masks are:
   - inner-leak related: classes `{1, 2}`
   - outer-leak related: classes `{3, 4}`
   - normal class: `0`
2. CSV stores split info and paths to `npy` files with shape `[2000, 2]`.
3. STFT target shape is obtained by `n_fft=382, win_length=382, hop_length=26` with `center=True`, resulting in `[2, 192, 63]`.

## Run
```bash
python train.py --config configs/config.yaml
```

## Input Modes
- `two_channel`: `x` is `[B, 2, 192, 63]`
- `split_dual`: `x1, x2` each is `[B, 192, 63]`
- `concat`: `x` is `[B, 192, 126]`

Configured via `data.input_mode`.
