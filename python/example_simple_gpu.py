import cupy as cp
from grff_gpu import get_mw


def main():
    Nf = 128
    Nz = 64

    Lparms = cp.zeros(5, dtype=cp.int32, order="F")
    Lparms[0] = Nz
    Lparms[1] = Nf
    Lparms[2] = 1

    Rparms = cp.zeros(3, dtype=cp.float64, order="F")
    Rparms[0] = 1e18
    Rparms[1] = 3e9
    Rparms[2] = 0.005

    ParmLocal = cp.zeros(15, dtype=cp.float64, order="F")
    ParmLocal[0] = 2e7
    ParmLocal[1] = 3e6
    ParmLocal[2] = 9e8
    ParmLocal[4] = 120.0
    ParmLocal[5] = 0.0
    ParmLocal[6] = 0
    ParmLocal[7] = 30

    Parms = cp.zeros((15, Nz), dtype=cp.float64, order="F")
    for i in range(Nz):
        Parms[:, i] = ParmLocal
        Parms[3, i] = 1000.0 - 700.0 * i / (Nz - 1)

    RL = cp.zeros((7, Nf), dtype=cp.float64, order="F")
    dummy = cp.asarray(0, dtype=cp.float64)

    res = get_mw(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)
    print("status", res)
    print("f[0:5] GHz:", RL[0, :5].get())
    print("I_L+I_R[0:5]", (RL[5, :5] + RL[6, :5]).get())


if __name__ == "__main__":
    main()
