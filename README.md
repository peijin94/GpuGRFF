# GpuGRFF

CUDA/C port of GRFF for use with CuPy RawModule. Initial focus: `GET_MW` and `GET_MW_SLICE`.

## Status
- Project scaffold + CuPy RawModule wrapper added.
- CUDA kernels are stubbed; full GRFF physics port is pending.

## Usage (planned)
See `python/grff_gpu.py` for the CuPy RawModule interface and expected array layouts.
# GpuGRFF

GPU (CUDA) port of GRFF with a CuPy RawModule interface.

## What works
- `GET_MW` and `GET_MW_SLICE` logic ported to CUDA (double precision).
- CuPy wrapper in `python/grff_gpu.py`.
- CPU vs GPU validation script: `python/compare_grff.py`.
- Multi‑pixel benchmark: `python/benchmark_slice.py`.
- GPU synthetic map: `python/synthetic_FF_gpu.py`.

## Usage (CuPy)
Minimal single‑pixel example:
```bash
python /home/pjzhang/dev/GpuGRFF/python/example_simple_gpu.py
```

CPU vs GPU comparison:
```bash
python /home/pjzhang/dev/GpuGRFF/python/compare_grff.py
```

Multi‑pixel benchmark:
```bash
python /home/pjzhang/dev/GpuGRFF/python/benchmark_slice.py
```

GPU synthetic free‑free map:
```bash
python /home/pjzhang/dev/GpuGRFF/python/synthetic_FF_gpu.py \
  --input /path/to/LOS_data.npz \
  --output emission_map_gpu \
  --freq0 4.5e8 --Nfreq 4 --freq-log-step 0.1
```

## Notes
- `GET_MW_SLICE` uses tiled launches (`tile_pixels`) to limit device `malloc` pressure.
- You can raise the device heap limit by passing `heap_bytes` in `get_mw_slice`.
- For best performance and robustness, a future step is removing device `malloc` and using preallocated workspaces.
