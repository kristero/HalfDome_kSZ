#In this file, we create the kSZ Map for the full FastPM Simulations

import numpy as np
import os, sys
import argparse
from bigfile import BigFile
import healpy as hp
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import scipy.integrate as integrate
from mpi4py import MPI

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# input params
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--res', type=int, default=1)   # 4 for 1/4 etc
parser.add_argument('--save_raw_data', type=int, default=0)
parser.add_argument('--chunk_size', type=int, default=1_000_000,
                    help='Number of particles read per chunk on each MPI rank')
parser.add_argument('--slice_batch_size', type=int, default=4,
                    help='Number of lightcone slices accumulated at once when writing per-slice maps')
parser.add_argument('--input_root', type=str,
                    default='/lustre/work/Globus-lt/halfdome/',
                    help='Base directory containing final_res/ and lower_res/')
parser.add_argument('--output_dir', type=str,
                    default='/home/kristero10/HalfDome_kSZ/kSZ_field/',
                    help='Directory where kSZ outputs will be written')
args = parser.parse_args()
seed = args.seed
res = args.res
save_raw_data = args.save_raw_data
chunk_size = args.chunk_size
slice_batch_size = args.slice_batch_size
input_root = os.path.abspath(args.input_root)
output_dir = os.path.abspath(args.output_dir)

if chunk_size <= 0:
    raise ValueError('--chunk_size must be positive')
if slice_batch_size <= 0:
    raise ValueError('--slice_batch_size must be positive')

dir_data = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
suffix = '_res%d_s%d' % (res, seed)

#Define cosmology

Om0 = 0.3089
Ob0 = 0.0486
cosmo = FlatLambdaCDM(H0=67.74, Om0 = Om0, Ob0 = Ob0, Tcmb0 = 2.7255, Neff = 3.046) #Neutrino mass is 0 by default
rho_crit = 27.75 *1e10 #(Msol/h)/(Mpc/h)^3
c_km = 3e5 #km/s
sigmat = 6.98426e-74 * (cosmo.h)**2 #(Mpc/h)^2
X = 0.86  #0.76
mu_e = 1.14   #2/(X+1)
m_p = 8.34895e-58 * cosmo.h #Units of Msol/h
nbar = (Ob0 * rho_crit)/(mu_e * m_p)
m_e = 9.1093837e-31/1.989e+30 * cosmo.h #Units of Msol/h
H0 = cosmo.h*100
rho_g0 = rho_crit * Ob0
f_e = 0.9

#Extract simulation data

if res == 1:
    sim = f'nbody/seed_{seed}/data/usmesh'
    cat_path = os.path.join(input_root, 'full_res', sim)
elif res == 16:
    sim = f'nbody/seed_{seed}/data/usmesh'
    cat_path = os.path.join(input_root, 'lower_res/16_res', sim, 'usmesh')
elif res == 8:
    sim = f'nbody/seed_{seed}/data/usmesh'
    cat_path = os.path.join(input_root, 'lower_res/8_res', sim, 'usmesh')
elif res == 4:
    sim = f'nbody/seed_{seed}/data/usmesh'
    cat_path = os.path.join(input_root, 'lower_res/4_res', sim, 'usmesh')
else:
    raise ValueError(f'Unsupported res={res}')

bf = BigFile(cat_path)
hp = bf['HEALPIX']


def block_length(block):
    shape = getattr(block, 'shape', None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    size_attr = getattr(block, 'size', None)
    if size_attr is not None:
        if isinstance(size_attr, tuple):
            return int(size_attr[0])
        return int(size_attr)
    return len(block)


def rank_bounds(total_size):
    start = total_size * rank // size
    stop = total_size * (rank + 1) // size
    return start, stop


def iter_local_chunks(total_size, step):
    start, stop = rank_bounds(total_size)
    for lo in range(start, stop, step):
        hi = min(lo + step, stop)
        yield lo, hi


def tau(z):
    H = H0 * np.sqrt(Om0*(1+z)**3 + 1 - Om0)
    c_div_H = c_km / H * cosmo.h   # this will be in Mpc/h, because H has units km/s/Mpc which will cancel with the km/s in c
    return sigmat * c_div_H * (1 + z)**2 * nbar


# kSZ has n*v = M/m/V * p/M = p/m/V. I.e. no need to compute the mass map. assuming I'm doing things correctly. I will not use this in notebook. But will for parallel!
# weight = -st * exp(tau) * n*v/c*dl = -st*exp(tau)*p/m/V/c*dl. so just bincount with this weight where you use correct tau etc for slice! see .py file.

def compute_chunk_weight(islice_chunk, rmom_chunk, z_slice, dl_slice, tau_slice, Mavg_slice, i_start):
    weight = np.zeros(rmom_chunk.shape, dtype=np.float64)
    valid = (islice_chunk >= i_start) & (Mavg_slice[islice_chunk] != 0.0)
    if not np.any(valid):
        return weight

    islice_valid = islice_chunk[valid]
    prefactor = sigmat * rho_g0 / mu_e / m_p / c_km * X
    weight[valid] = (
        prefactor
        * dl_slice[islice_valid]
        * np.exp(-tau_slice[islice_valid])
        * (1.0 + z_slice[islice_valid])**3
        * (rmom_chunk[valid] / Mavg_slice[islice_valid])
    )
    return weight


attrs = hp.attrs
#M0 = cat.attrs['MassTable'][1] * 1e10  # mass of dark matter simulation particle in Msun/h
aemitIndex_edges = attrs['aemitIndex.edges']
#aemitIndex_offset = cat.attrs['aemitIndex.offset']
nside =  attrs['healpix.nside'][0]
npix = attrs['healpix.npix'][0]
nslice = attrs['healpix.nslices'][0]  # len(aedegs)-1

pid_block = bf['HEALPIX/ID']
rmom_block = bf['HEALPIX/Rmom']
mass_block = bf['HEALPIX/Mass']
npart = block_length(pid_block)
local_start, local_stop = rank_bounds(npart)

# calculate tau for each slice (would be nice to vectorize quad, but not sure how). also do vol and dl
a_start = 0.2
i_start = np.where(aemitIndex_edges==a_start)[0][0]  # explicitly dont include all the "0" slices before the start of the lightcone. there is one non-0 entry here, don't know why, but explicitly cut it out.

z_slice = np.zeros(nslice)
dl_slice = np.zeros(nslice)
chi_slice = np.zeros(nslice)
tau_slice = np.zeros(nslice)
Msum_slice = np.zeros(nslice)    # average Mass in slice
for i in range(i_start, nslice):

    a1 = aemitIndex_edges[i]
    a2 = aemitIndex_edges[i + 1]
    z1 = 1. / a1 - 1.
    z2 = 1. / a2 - 1.
    d1 = cosmo.comoving_distance(z1).value * cosmo.h
    d2 = cosmo.comoving_distance(z2).value * cosmo.h
    l1 = d1 / (1+z1)
    l2 = d2 / (1+z2)

    z_slice[i] = 2/(a1+a2)-1
    dl_slice[i] = l1 - l2
    chi_slice[i] = cosmo.comoving_distance(2/(a1+a2)-1).value * cosmo.h
    tau_slice[i] = integrate.romberg(tau, 0, 2/(a1+a2)-1)

for lo, hi in iter_local_chunks(npart, chunk_size):
    pid_chunk = np.asarray(pid_block[lo:hi], dtype=np.int64)
    islice_chunk = pid_chunk // npix
    valid = islice_chunk >= i_start
    if not np.any(valid):
        continue

    mass_chunk = np.asarray(mass_block[lo:hi], dtype=np.float64) * 1e10
    Msum_slice += np.bincount(
        islice_chunk[valid],
        weights=mass_chunk[valid],
        minlength=nslice,
    )

comm.Allreduce(MPI.IN_PLACE, Msum_slice, op=MPI.SUM)
Mavg_slice = Msum_slice / npix

# save some useful data for debugging and postprocessing
if save_raw_data:
    os.makedirs(dir_data+'raw/', exist_ok=1)
    local_count = local_stop - local_start
    ipix_raw = np.lib.format.open_memmap(
        dir_data+'raw/ipix%s.%d.npy' % (suffix, rank),
        mode='w+',
        dtype=np.int64,
        shape=(local_count,),
    )
    islice_raw = np.lib.format.open_memmap(
        dir_data+'raw/islice%s.%d.npy' % (suffix, rank),
        mode='w+',
        dtype=np.int64,
        shape=(local_count,),
    )
    weight_raw = np.lib.format.open_memmap(
        dir_data+'raw/weight%s.%d.npy' % (suffix, rank),
        mode='w+',
        dtype=np.float64,
        shape=(local_count,),
    )
    rmom_raw = np.lib.format.open_memmap(
        dir_data+'raw/Rmom%s.%d.npy' % (suffix, rank),
        mode='w+',
        dtype=np.float64,
        shape=(local_count,),
    )
    mass_raw = np.lib.format.open_memmap(
        dir_data+'raw/Mass%s.%d.npy' % (suffix, rank),
        mode='w+',
        dtype=np.float64,
        shape=(local_count,),
    )
else:
    ipix_raw = islice_raw = weight_raw = rmom_raw = mass_raw = None


local_total = np.zeros(npix, dtype=np.float32)
raw_offset = 0
for lo, hi in iter_local_chunks(npart, chunk_size):
    pid_chunk = np.asarray(pid_block[lo:hi], dtype=np.int64)
    islice_chunk = pid_chunk // npix
    ipix_chunk = pid_chunk % npix
    rmom_chunk = np.asarray(rmom_block[lo:hi], dtype=np.float64) * 1e10
    weight_chunk = compute_chunk_weight(islice_chunk, rmom_chunk, z_slice, dl_slice, tau_slice, Mavg_slice, i_start)

    valid = islice_chunk >= i_start
    if np.any(valid):
        local_total += np.bincount(
            ipix_chunk[valid],
            weights=weight_chunk[valid],
            minlength=npix,
        ).astype(np.float32)

    if save_raw_data:
        local_len = hi - lo
        mass_chunk = np.asarray(mass_block[lo:hi], dtype=np.float64) * 1e10
        sl = slice(raw_offset, raw_offset + local_len)
        ipix_raw[sl] = ipix_chunk
        islice_raw[sl] = islice_chunk
        weight_raw[sl] = weight_chunk
        rmom_raw[sl] = rmom_chunk
        mass_raw[sl] = mass_chunk
        raw_offset += local_len

if save_raw_data and rank == root:
    np.save(dir_data+'raw/Mavg_slice%s' % (suffix), Mavg_slice)

# save each slice
os.makedirs(dir_data, exist_ok=1)
active_slices = np.arange(i_start, nslice, dtype=np.int64)
for batch_start in range(0, active_slices.size, slice_batch_size):
    batch_slices = active_slices[batch_start:batch_start + slice_batch_size]
    batch_maps = np.zeros((batch_slices.size, npix), dtype=np.float32)
    batch_index = {int(slice_id): j for j, slice_id in enumerate(batch_slices)}

    for lo, hi in iter_local_chunks(npart, chunk_size):
        pid_chunk = np.asarray(pid_block[lo:hi], dtype=np.int64)
        islice_full = pid_chunk // npix
        in_batch = (islice_full >= batch_slices[0]) & (islice_full <= batch_slices[-1])
        if not np.any(in_batch):
            continue

        ipix_chunk = pid_chunk[in_batch] % npix
        islice_chunk = islice_full[in_batch]
        rmom_chunk = np.asarray(rmom_block[lo:hi], dtype=np.float64) * 1e10
        weight_chunk = compute_chunk_weight(
            islice_chunk,
            rmom_chunk[in_batch],
            z_slice,
            dl_slice,
            tau_slice,
            Mavg_slice,
            i_start,
        )

        for slice_id in np.unique(islice_chunk):
            local_mask = islice_chunk == slice_id
            batch_maps[batch_index[int(slice_id)]] += np.bincount(
                ipix_chunk[local_mask],
                weights=weight_chunk[local_mask],
                minlength=npix,
            ).astype(np.float32)

    for j, slice_id in enumerate(batch_slices):
        owner = int((int(slice_id) - i_start) % size)
        if rank == owner:
            recvbuf = np.empty(npix, dtype=np.float32)
            comm.Reduce(batch_maps[j], recvbuf, op=MPI.SUM, root=owner)
            outpath = dir_data+'dTkSZ_div_TCMB%s.%d' % (suffix, int(slice_id))
            np.save(outpath, recvbuf)
        else:
            comm.Reduce(batch_maps[j], None, op=MPI.SUM, root=owner)

# save total
dTkSZ_div_TCMB = local_total
if rank == root:
    comm.Reduce(MPI.IN_PLACE, dTkSZ_div_TCMB, op=MPI.SUM, root=root)
else:
    comm.Reduce(dTkSZ_div_TCMB, None, op=MPI.SUM, root=root)  # dummy
if rank == root:
    os.makedirs(dir_data, exist_ok=1)
    np.save(dir_data+'dTkSZ_div_TCMB%s' % suffix, dTkSZ_div_TCMB)
    np.save(dir_data+'z_slice', z_slice)   # useful to save for dm2gas
    np.save(dir_data+'chi_slice', chi_slice)
