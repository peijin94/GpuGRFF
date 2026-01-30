# GpuGRFF

CUDA/C port of GRFF for use with CuPy RawModule. Initial focus: `GET_MW` and `GET_MW_SLICE`.

## Status
- Project scaffold + CuPy RawModule wrapper added.
- CUDA kernels are stubbed; full GRFF physics port is pending.

## Usage (planned)
See `python/grff_gpu.py` for the CuPy RawModule interface and expected array layouts.
