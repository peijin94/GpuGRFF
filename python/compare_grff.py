import ctypes
import numpy as np
import cupy as cp

from grff_gpu import get_mw


def init_get_mw(libname):
    _intp = np.ctypeslib.ndpointer(dtype=np.int32, flags="F")
    _doublep = np.ctypeslib.ndpointer(dtype=np.float64, flags="F")

    libc_mw = ctypes.CDLL(libname)
    mwfunc = libc_mw.PyGET_MW
    mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int
    return mwfunc


def main():
    libname = "../../GRFF/binaries/GRFF_DEM_Transfer.so"
    cpu_get_mw = init_get_mw(libname)

    Nf = 32
    Nz = 32

    Lparms = np.zeros(5, dtype=np.int32, order="F")
    Lparms[0] = Nz
    Lparms[1] = Nf
    Lparms[2] = 1

    Rparms = np.zeros(3, dtype=np.float64, order="F")
    Rparms[0] = 1e18
    Rparms[1] = 3e9
    Rparms[2] = 0.005

    ParmLocal = np.zeros(15, dtype=np.float64, order="F")
    ParmLocal[0] = 2e7
    ParmLocal[1] = 3e6
    ParmLocal[2] = 9e8
    ParmLocal[4] = 120.0
    ParmLocal[5] = 0.0
    ParmLocal[6] = 0
    ParmLocal[7] = 30

    Parms = np.zeros((15, Nz), dtype=np.float64, order="F")
    for i in range(Nz):
        Parms[:, i] = ParmLocal
        Parms[3, i] = 1000.0 - 700.0 * i / (Nz - 1)

    RL_cpu = np.zeros((7, Nf), dtype=np.float64, order="F")
    dummy = np.array(0, dtype=np.float64)

    res_cpu = cpu_get_mw(Lparms, Rparms, Parms, dummy, dummy, dummy, RL_cpu)
    print(f"CPU status: {res_cpu}")

    # GPU
    Lparms_g = cp.asarray(Lparms, order="F")
    Rparms_g = cp.asarray(Rparms, order="F")
    Parms_g = cp.asarray(Parms, order="F")
    T_arr_g = cp.asarray(dummy, order="F")
    DEM_g = cp.asarray(dummy, order="F")
    DDM_g = cp.asarray(dummy, order="F")
    RL_g = cp.zeros((7, Nf), dtype=cp.float64, order="F")

    res_gpu = get_mw(Lparms_g, Rparms_g, Parms_g, T_arr_g, DEM_g, DDM_g, RL_g)
    print(f"GPU status: {res_gpu}")

    RL_gpu = RL_g.get()
    diff = np.nanmax(np.abs(RL_cpu - RL_gpu))
    rel = np.nanmax(np.abs(RL_cpu - RL_gpu) / (np.abs(RL_cpu) + 1e-30))
    print(f"max abs diff: {diff}")
    print(f"max rel diff: {rel}")


if __name__ == "__main__":
    main()
