import time
import cupy as cp

from grff_gpu import get_mw_slice


def build_inputs(Npix=256, Nz=64, Nf=8):
    Lparms_M = cp.zeros(6, dtype=cp.int32, order="F")
    Lparms_M[0] = Npix
    Lparms_M[1] = Nz
    Lparms_M[2] = Nf
    Lparms_M[3] = 1

    Rparms_M = cp.zeros((3, Npix), dtype=cp.float64, order="F")
    Rparms_M[0, :] = 1e18
    Rparms_M[1, :] = 3e9
    Rparms_M[2, :] = 0.005

    ParmLocal = cp.zeros(15, dtype=cp.float64, order="F")
    ParmLocal[0] = 2e7
    ParmLocal[1] = 3e6
    ParmLocal[2] = 9e8
    ParmLocal[4] = 120.0
    ParmLocal[5] = 0.0
    ParmLocal[6] = 0
    ParmLocal[7] = 30

    Parms_M = cp.zeros((15, Nz, Npix), dtype=cp.float64, order="F")
    for i in range(Nz):
        Parms_M[:, i, :] = ParmLocal[:, None]
        Parms_M[3, i, :] = 1000.0 - 700.0 * i / (Nz - 1)

    T_arr = cp.asarray(0, dtype=cp.float64)
    DEM_arr_M = cp.asarray(0, dtype=cp.float64)
    DDM_arr_M = cp.asarray(0, dtype=cp.float64)

    RL_M = cp.zeros((7, Nf, Npix), dtype=cp.float64, order="F")

    return Lparms_M, Rparms_M, Parms_M, T_arr, DEM_arr_M, DDM_arr_M, RL_M


def main():
    Npix = 8192
    Nz = 256
    Nf = 16

    inputs = build_inputs(Npix=Npix, Nz=Nz, Nf=Nf)

    # Warmup
    get_mw_slice(*inputs)
    cp.cuda.Stream.null.synchronize()

    iters = 5
    start = time.perf_counter()
    for _ in range(iters):
        get_mw_slice(*inputs)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    per_iter = elapsed / iters
    print(f"Npix={Npix} Nz={Nz} Nf={Nf}")
    print(f"avg time per call: {per_iter:.4f} s")
    print(f"pixels per second: {Npix*Nf / per_iter:.2f}")


if __name__ == "__main__":
    main()
