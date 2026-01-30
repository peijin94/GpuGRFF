#!/usr/bin/env python
"""
Synthetic free-free emission calculation using GpuGRFF (CuPy).

Uses GET_MW_SLICE to process all pixels in parallel on GPU.
Input format matches GRFFradioSun/synthetic_FF.py.
"""

import argparse
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from grff_gpu import get_mw_slice

# Constants
R_sun = 6.957e10  # cm
c = 2.998e10      # cm/s
kb = 1.38065e-16  # erg/K
sfu2cgs = 1e-19   # SFU to CGS


def _prepare_parms(ne_los, te_los, b_los, ds_los):
    """Build Parms_M for all pixels with fixed Nz, compacting valid LOS points to the front."""
    N_pix, _, Nz = ne_los.shape
    Npix = N_pix * N_pix

    # CPU-side stable compaction to preserve LOS order
    ne = ne_los.reshape(Npix, Nz)
    te = te_los.reshape(Npix, Nz)
    bb = b_los.reshape(Npix, Nz)
    ds = ds_los.reshape(Npix, Nz)

    valid = ~(np.isnan(ne) | np.isnan(te) | np.isnan(bb))
    order = np.argsort(~valid, axis=1, kind="stable")

    ne = np.take_along_axis(ne, order, axis=1)
    te = np.take_along_axis(te, order, axis=1)
    bb = np.take_along_axis(bb, order, axis=1)
    ds = np.take_along_axis(ds, order, axis=1)

    counts = valid.sum(axis=1)
    idx = np.arange(Nz)[None, :]
    mask_tail = idx >= counts[:, None]
    ne[mask_tail] = 0.0
    te[mask_tail] = 0.0
    bb[mask_tail] = 0.0
    ds[mask_tail] = 0.0

    ne = cp.asarray(ne)
    te = cp.asarray(te)
    bb = cp.asarray(bb)
    ds = cp.asarray(ds)

    Parms_M = cp.zeros((15, Nz, Npix), dtype=cp.float64, order="F")
    Parms_M[0, :, :] = ds.T
    Parms_M[1, :, :] = te.T
    Parms_M[2, :, :] = ne.T
    Parms_M[3, :, :] = bb.T
    Parms_M[4, :, :] = 90.0
    Parms_M[5, :, :] = 0.0
    Parms_M[6, :, :] = 1 + 4  # FF only
    Parms_M[7, :, :] = 30

    return Parms_M


def SyntheticFF_GPU(fname_input, freq0, Nfreq, freq_log_step, fname_output):
    data = np.load(fname_input)
    Ne_LOS = data['Ne_LOS']
    Te_LOS = data['Te_LOS']
    B_LOS = data['B_LOS']
    ds_LOS = data['ds_LOS']
    x_coords = data['x_coords']
    y_coords = data['y_coords']

    N_pix = Ne_LOS.shape[0]
    Nz = Ne_LOS.shape[2]
    Nf = Nfreq
    frequencies_Hz = freq0 * (10.0 ** (freq_log_step * np.arange(Nf)))

    # Lparms_M: [Npix, Nz, Nf, NT, ...]
    Npix = N_pix * N_pix
    Lparms_M = cp.zeros(6, dtype=cp.int32, order="F")
    Lparms_M[0] = Npix
    Lparms_M[1] = Nz
    Lparms_M[2] = Nf
    Lparms_M[3] = 1

    # Rparms_M: per-pixel params (area, freq0, logstep)
    pixel_size_Rsun = (x_coords[1] - x_coords[0]) / (R_sun * 1e-2)
    pixel_size_cm = pixel_size_Rsun * R_sun
    area = pixel_size_cm * pixel_size_cm

    Rparms_M = cp.zeros((3, Npix), dtype=cp.float64, order="F")
    Rparms_M[0, :] = area
    Rparms_M[1, :] = freq0
    Rparms_M[2, :] = freq_log_step

    Parms_M = _prepare_parms(Ne_LOS, Te_LOS, B_LOS, ds_LOS)

    T_arr = cp.asarray(0, dtype=cp.float64)
    DEM_arr_M = cp.asarray(0, dtype=cp.float64)
    DDM_arr_M = cp.asarray(0, dtype=cp.float64)

    RL_M = cp.zeros((7, Nf, Npix), dtype=cp.float64, order="F")

    print(f"Image size: {N_pix}x{N_pix}, LOS points: {Nz}")
    print(f"Frequencies: {frequencies_Hz/1e6} MHz (Nf={Nf})")
    print("Running GET_MW_SLICE on GPU...")

    status = get_mw_slice(
        Lparms_M,
        Rparms_M,
        Parms_M,
        T_arr,
        DEM_arr_M,
        DDM_arr_M,
        RL_M,
        tile_pixels=256,
        heap_bytes=2 * 1024 * 1024 * 1024,
    )
    if np.any(status != 0):
        bad = np.where(status != 0)[0]
        print(f"Warning: {bad.size} pixels returned non-zero status")
        unique, counts = np.unique(status, return_counts=True)
        print("Status histogram:", dict(zip(unique.tolist(), counts.tolist())))
        if bad.size > 0:
            sample = bad[:10]
            coords = [(int(idx // N_pix), int(idx % N_pix)) for idx in sample]
            print(f"Sample bad pixel indices: {coords}")

    intensity = (RL_M[5] + RL_M[6]).T  # (Npix, Nf)
    denom = (RL_M[5] + RL_M[6])
    pol_vi = cp.where(denom != 0, (RL_M[5] - RL_M[6]) / denom, 0.0).T

    nu = cp.asarray(frequencies_Hz, dtype=cp.float64)
    conv = (sfu2cgs * c * c / (2.0 * kb * nu * nu) / area) * (1.49599e13 ** 2)

    emission = intensity * conv[None, :]

    emission_cube = emission.reshape(N_pix, N_pix, Nf)
    emission_polVI_cube = pol_vi.reshape(N_pix, N_pix, Nf)

    emission_cube_cpu = cp.asnumpy(emission_cube)
    emission_polVI_cube_cpu = cp.asnumpy(emission_polVI_cube)

    print("\nBrightness temperature calculation complete!")

    frequency_first = frequencies_Hz[0]
    emission_map_first = emission_cube_cpu[:, :, 0]
    emission_polVI_map_first = emission_polVI_cube_cpu[:, :, 0]

    center_size = 16
    center_start = N_pix // 2 - center_size // 2
    center_end = N_pix // 2 + center_size // 2
    center_region = emission_map_first[center_start:center_end, center_start:center_end]
    valid_center = center_region[center_region > 0]
    avg_center_str = f"{np.mean(valid_center):.2e}" if len(valid_center) > 0 else "N/A"
    if len(valid_center) > 0:
        print(f"\nAverage brightness temperature (center {center_size}x{center_size}, first freq): {np.mean(valid_center):.2e} K")

    print("\nSaving brightness temperature cube...")
    np.savez_compressed(fname_output + '.npz',
                        emission_cube=emission_cube_cpu,
                        emission_polVI_cube=emission_polVI_cube_cpu,
                        frequencies_Hz=frequencies_Hz,
                        x_coords=x_coords,
                        y_coords=y_coords)
    print(f"Brightness temperature cube saved to {fname_output}.npz (shape {N_pix} x {N_pix} x {Nf})")

    print("\nPlotting brightness temperature map (first frequency)...")
    fig, ax = plt.subplots(figsize=(6, 4.8))
    x_range = [x_coords[0] / (R_sun * 1e-2), x_coords[-1] / (R_sun * 1e-2)]
    y_range = [y_coords[0] / (R_sun * 1e-2), y_coords[-1] / (R_sun * 1e-2)]
    emission_plot = emission_map_first.copy()
    emission_plot[emission_plot == 0] = np.nan
    im = ax.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                   aspect='equal', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('x (R_sun)')
    ax.set_ylabel('y (R_sun)')
    ax.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz')
    plt.colorbar(im, ax=ax, label='T_b (K)')
    ax.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(fname_output + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Emission map saved to {fname_output}.png")

    fig_tb_vi, (ax_tb, ax_vi) = plt.subplots(1, 2, figsize=(12, 4.2))
    im_tb = ax_tb.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                         aspect='equal', cmap='hot', interpolation='bilinear')
    ax_tb.set_xlabel('x (R_sun)')
    ax_tb.set_ylabel('y (R_sun)')
    ax_tb.set_title(f'$T_b$ at {frequency_first/1e9:.3f} GHz')
    plt.colorbar(im_tb, ax=ax_tb, label='T_b (K)')
    pol_vi_plot = emission_polVI_map_first.copy()
    pol_vi_plot[emission_map_first == 0] = np.nan
    vmax_vi = np.nanmax(np.abs(pol_vi_plot))
    if np.isnan(vmax_vi) or vmax_vi == 0:
        vmax_vi = 1.0
    im_vi = ax_vi.imshow(pol_vi_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                         aspect='equal', cmap='RdBu_r', interpolation='bilinear', vmin=-vmax_vi, vmax=vmax_vi)
    ax_vi.set_xlabel('x (R_sun)')
    ax_vi.set_ylabel('y (R_sun)')
    ax_vi.set_title(f'V/I at {frequency_first/1e9:.3f} GHz')
    plt.colorbar(im_vi, ax=ax_vi, label='V/I')
    plt.tight_layout()
    plt.savefig(fname_output + '_Tb_VI.png', dpi=150, bbox_inches='tight')
    plt.close(fig_tb_vi)
    print(f"T_b and V/I side-by-side plot saved to {fname_output}_Tb_VI.png")

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    im2 = ax2.imshow(emission_plot, origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='equal', cmap='hot', interpolation='bilinear',
                     norm=mcolors.LogNorm(vmin=np.nanmin(emission_plot[emission_plot > 0]),
                                          vmax=np.nanmax(emission_plot)))
    ax2.set_xlabel('x (R_sun)')
    ax2.set_ylabel('y (R_sun)')
    ax2.set_title(f'synthetic $T_b$ map at {frequency_first/1e9:.3f} GHz (Log Scale)')
    plt.colorbar(im2, ax=ax2, label='T_b (K)')
    ax2.text(0.97, 0.97, f'Center $T_b$: {avg_center_str}', transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(fname_output + '_log.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Log-scale brightness temperature map saved to {fname_output}_log.png")
    print("\nSynthetic brightness temperature calculation complete!")

    return {
        'emission_cube': emission_cube_cpu,
        'emission_polVI_cube': emission_polVI_cube_cpu,
        'frequencies_Hz': frequencies_Hz,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic free-free emission via GpuGRFF (GPU).')
    parser.add_argument('--input', '-i', type=str, default='LOS_data.npz',
                        help='Path to LOS npz file (default: LOS_data.npz)')
    parser.add_argument('--output', '-o', type=str, default='emission_map_gpu',
                        help='Base path for output files, no extension (default: emission_map_gpu)')
    parser.add_argument('--freq0', '-f', type=float, default=450e6,
                        help='Start frequency in Hz (default: 450e6)')
    parser.add_argument('--Nfreq', '-n', type=int, default=4,
                        help='Number of frequencies (default: 4)')
    parser.add_argument('--freq-log-step', '-s', type=float, default=0.1,
                        help='log10 step between frequencies (default: 0.1)')
    args = parser.parse_args()

    SyntheticFF_GPU(args.input, args.freq0, args.Nfreq, args.freq_log_step, args.output)
