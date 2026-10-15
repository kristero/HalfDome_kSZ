"""FLAMINGO full snapshots and SOAP catalogues on local disk (h5py) or on the COSMA service (hdfstream).

``Store(data_root)`` hides the backend: a directory for local HDF5 files, or ``hdfstream:<path>``
for the anonymous COSMA service (``hdfstream.open('cosma', '/')``). ``Snapshot`` reads one
top-level cell of gas at a time in the units of SPEC section 4 from either layout:

  (a) virtual single file: NumFilesPerSnapshot == 1, Cells/Files all 0, global Cells/OffsetsInFile;
  (b) distributed: chunk files flamingo_NNNN.<i>.hdf5 next to the top-level file or in
      swift_snapshot_NNNN/, each cell in chunk Cells/Files at Cells/OffsetsInFile inside that chunk.

Chunk files are used when they exist (per-file Cells/Files and OffsetsInFile come from the top-level
file if it has NumFilesPerSnapshot > 1, otherwise from chunk 0, where they are per file); otherwise
the top-level datasets are read at global offset = start of the cell's file + OffsetsInFile. A field
whose top-level dataset carries the 'auxiliary file' attribute (recalculated data the chunk files do
not hold) is read through the top-level file (read_from 'auto'); read_from 'chunks' is then refused.
Checked on L2p8_m9 z = 0 (tests/check_chunk_reads.py): both paths give identical arrays for the
pipeline's fields (ComptonYParameters float32 in the chunks, float64 in the virtual file, same values).
``Soap`` reads catalogue columns for all rows (local only) or for given rows / cells through the
cell index (the only access allowed remotely: whole SOAP columns are never streamed).

Gas bound-halo membership (``MEMBERSHIP_CANDIDATES``: PartType0/HaloCatalogueIndex, else GroupNr_bound) exists
only in the top-level file (on L2p8_m9 it is linked from the membership files, not stored in the chunks), so
``Snapshot.read_membership`` always reads it there at the cell's global offset. It is the halo-finder index
(= SOAP InputHalos/HaloCatalogueIndex) of the subhalo a particle is bound to, -1 if unbound.

Local reads: chunk files (and a local top-level file) are opened with an HDF5 chunk cache of
``chunk_cache_bytes`` per dataset (rdcc_w0 = 1: fully read chunks are evicted first) and their datasets are kept
open, so consecutive cells of a file that share a compressed HDF5 chunk (SWIFT writes 2^20-row chunks, more than
a mean cell) decompress it once; the particle stage gives every task and worker contiguous runs of cells in
(file, offset) order for this.

Conversions come from the dataset attributes, never from assumed little-h powers.
"""
from __future__ import annotations

from collections import OrderedDict
import os
import time
import warnings

import numpy as np

import common
from common import MPC_CM, MSUN_G

GAS = "PartType0"
PHYS_ATTR = "Conversion factor to physical CGS (including cosmological corrections)"
COMOVING_ATTR = "Conversion factor to CGS (not including cosmological corrections)"
# name -> (CGS value of the target unit, comoving). Comoving quantities use the factor without a-scaling.
GAS_FIELDS = {
    "Coordinates": (MPC_CM, True),                         # comoving Mpc
    "SmoothingLengths": (MPC_CM, True),                    # comoving Mpc
    "Masses": (MSUN_G, False),                             # Msun
    "Densities": (MSUN_G / MPC_CM**3, False),              # physical Msun / Mpc^3
    "ComptonYParameters": (MPC_CM**2, False),              # physical Mpc^2
    "ElectronNumberDensities": (MPC_CM**-3, False),        # physical Mpc^-3
    "Temperatures": (1.0, False),                          # K
    "LastAGNFeedbackScaleFactors": (1.0, False),           # dimensionless
}
HEATING_FIELDS = ("LastAGNFeedbackScaleFactors", "Temperatures")
# Attribute of a top-level virtual dataset whose data come from an auxiliary file (recalculated after the run, e.g.
# FOFGroupIDs, XrayLuminosities on L2p8_m9): the chunk files then hold the ORIGINAL values, so such a field must be
# read through the top-level file (tests/check_chunk_reads.py found no pipeline field with it on L2p8_m9 z = 0).
AUXILIARY_ATTR = "auxiliary file"
# gas bound-halo membership in the top-level file (halo-finder index of the subhalo a particle is bound to, -1 if
# unbound; = SOAP InputHalos/HaloCatalogueIndex); 'auto' takes the first present (particles.membership_field)
MEMBERSHIP_CANDIDATES = ("HaloCatalogueIndex", "GroupNr_bound")

# SOAP catalogue columns: name -> (dataset, CGS value of the target unit; None = raw integers)
CATALOGUE_COLUMNS = {
    "halo_index": ("InputHalos/HaloCatalogueIndex", None),
    "is_central": ("InputHalos/IsCentral", None),
    "centre_mpc": ("InputHalos/HaloCentre", MPC_CM),
    "m200c_msun": ("SO/200_crit/TotalMass", MSUN_G),
    "m500c_msun": ("SO/500_crit/TotalMass", MSUN_G),
    "m100c_msun": ("SO/100_crit/TotalMass", MSUN_G),
    "dm_concentration_200c": ("SO/200_crit/DarkMatterConcentration", 1.0),
    "gas_temperature_200c_k": ("SO/200_crit/GasTemperature", 1.0),
}
for _ap in common.APERTURES:
    CATALOGUE_COLUMNS[f"r_{_ap}_mpc"] = (f"SO/{_ap}/SORadius", MPC_CM)
    CATALOGUE_COLUMNS[f"y_{_ap}_mpc2"] = (f"SO/{_ap}/ComptonY", MPC_CM**2)
    CATALOGUE_COLUMNS[f"y_noagn_{_ap}_mpc2"] = (f"SO/{_ap}/ComptonYWithoutRecentAGNHeating", MPC_CM**2)
    CATALOGUE_COLUMNS[f"gasmass_{_ap}_msun"] = (f"SO/{_ap}/GasMass", MSUN_G)
    CATALOGUE_COLUMNS[f"ngas_{_ap}"] = (f"SO/{_ap}/NumberOfGasParticles", None)
OPTIONAL_COLUMNS = {"gas_temperature_200c_k", "dm_concentration_200c"}
SELECTION_COLUMNS = {k: CATALOGUE_COLUMNS[k] for k in ("is_central", "m200c_msun", "r_200_crit_mpc")}


# --------------------------------------------------------------------------- remote helpers
def retry(function, *args, attempts=5, what="request", **kwargs):
    """Retry transient service failures with backoff (never retries a reached download cap)."""
    for attempt in range(attempts):
        try:
            return function(*args, **kwargs)
        except KeyError:
            raise
        except Exception as exc:  # noqa: BLE001
            if "download cap" in str(exc) or attempt == attempts - 1:
                raise
            delay = min(120, 10 * 2**attempt)
            print(f"  retry {attempt + 1}/{attempts - 1} for {what} in {delay} s: {exc!r}"[:300], flush=True)
            time.sleep(delay)


_DOWNLOADED = {"bytes": 0, "limit": None}


def install_byte_counter(limit_bytes=None):
    """Count HTTP payload bytes in this process and stop above ``limit_bytes`` (rehearsals); returns the counter."""
    import requests
    _DOWNLOADED["limit"] = limit_bytes
    if getattr(requests.sessions.Session, "_fio_counted", False):
        return _DOWNLOADED
    original_send = requests.sessions.Session.send

    def send(session, request, **kwargs):
        kwargs.setdefault("timeout", (30, 900))
        wanted_stream = kwargs.get("stream", False)
        kwargs["stream"] = True
        response = original_send(session, request, **kwargs)
        raw_read = response.raw.read

        def read(amt=None, *args, **read_kwargs):
            body = raw_read(amt, *args, **read_kwargs)
            if body:
                _DOWNLOADED["bytes"] += len(body)
                if _DOWNLOADED["limit"] and _DOWNLOADED["bytes"] > _DOWNLOADED["limit"]:
                    raise RuntimeError(f"download cap of {_DOWNLOADED['limit'] / 1e9:.2f} GB reached")
            return body

        response.raw.read = read
        if not wanted_stream:
            response.content  # noqa: B018  (consume the body through the counter)
        return response

    requests.sessions.Session.send = send
    requests.sessions.Session._fio_counted = True
    return _DOWNLOADED


# --------------------------------------------------------------------------- store
class Store:
    """A data root: local directory, or ``hdfstream:<path>`` on the COSMA service (anonymous).

    ``data_size_limit`` (remote only): datasets smaller than this are downloaded with a file's metadata
    (hdfstream's default 1 MB; the real-data checks pass a few kB so that opening a chunk file costs only
    its metadata). Remote file handles are cached, so exists() followed by open() downloads the metadata once.
    """

    def __init__(self, data_root, data_size_limit=None):
        root = str(data_root)
        self.remote = root.startswith("hdfstream:")
        self.base = root[len("hdfstream:"):].strip("/") if self.remote else root
        self._service = None
        self._data_size_limit = data_size_limit
        self._remote_files = {}

    def __repr__(self):
        return f"Store({'hdfstream:' if self.remote else ''}{self.base})"

    def resolve(self, rel):
        if self.remote:
            return f"{self.base}/{rel}".strip("/") if self.base else rel
        return os.path.join(self.base, rel)

    def service(self):
        if self._service is None:
            import hdfstream
            kwargs = {} if self._data_size_limit is None else {"data_size_limit": int(self._data_size_limit)}
            self._service = retry(hdfstream.open, "cosma", "/", what="hdfstream.open", **kwargs)
        return self._service

    def _remote(self, rel):
        path = self.resolve(rel)
        if path not in self._remote_files:
            self._remote_files[path] = retry(lambda: self.service()[path], what=rel)
        return self._remote_files[path]

    def exists(self, rel):
        if not self.remote:
            return os.path.exists(self.resolve(rel))
        try:
            self._remote(rel)
            return True
        except KeyError:
            return False

    def open(self, rel, cache_bytes=None, w0=None, nslots=1_000_003):
        """Open an HDF5 file (h5py.File locally, hdfstream RemoteFile remotely); `cache_bytes` is the raw-data chunk
        cache of every dataset opened from it (HDF5 keeps one cache per open dataset), `w0` its preemption policy."""
        if self.remote:
            return self._remote(rel)
        import h5py
        kwargs = {} if cache_bytes is None else {"rdcc_nbytes": int(cache_bytes), "rdcc_nslots": int(nslots)}
        if cache_bytes is not None and w0 is not None:
            kwargs["rdcc_w0"] = float(w0)
        return h5py.File(self.resolve(rel), "r", **kwargs)

    def read(self, dataset, start, stop):
        """dataset[start:stop] (all columns), with retries remotely."""
        if self.remote:
            return np.asarray(retry(lambda: dataset[start:stop], what=getattr(dataset, "name", "dataset")))
        return dataset[start:stop]

    def read_slices(self, dataset, starts, stops):
        """Concatenation of dataset[a:b] for the given runs (one request_slices call remotely)."""
        if not len(starts):
            return np.empty((0,) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
        if self.remote:
            extra = (slice(None),) * (len(dataset.shape) - 1)
            slices = [(slice(int(a), int(b)),) + extra if extra else slice(int(a), int(b)) for a, b in zip(starts, stops)]
            return np.asarray(retry(dataset.request_slices, slices, what=getattr(dataset, "name", "dataset")))
        return np.concatenate([dataset[int(a):int(b)] for a, b in zip(starts, stops)])


def sim_paths(sim_config, snap):
    """Relative paths of the top-level snapshot and the SOAP file (overridable per simulation)."""
    base = sim_config["path"].strip("/")
    snapshot = sim_config.get("snapshot_file", "snapshots/flamingo_{snap:04d}/flamingo_{snap:04d}.hdf5")
    soap = sim_config.get("soap_file", "SOAP-HBT/halo_properties_{snap:04d}.hdf5")
    return f"{base}/{snapshot.format(snap=int(snap))}", f"{base}/{soap.format(snap=int(snap))}"


def open_snapshot(cfg, sim, snap):
    top, _ = sim_paths(common.sim_cfg(cfg, sim), snap)
    soft = cfg.get("softening", {})
    part = cfg.get("particles", {})
    cache_mb = part.get("hdf5_chunk_cache_mb", 64)
    return Snapshot(Store(cfg["data_root"]), top, snap, read_from=part.get("read_from", "auto"),
                    softening_fallback=(soft.get("comoving_mpc"), soft.get("max_physical_mpc")),
                    max_open_chunks=int(part.get("max_open_chunks", 2)),
                    chunk_cache_bytes=None if cache_mb is None else int(float(cache_mb) * 2**20))


def open_soap(cfg, sim, snap):
    _, soap = sim_paths(common.sim_cfg(cfg, sim), snap)
    return Soap(Store(cfg["data_root"]), soap)


def _unit(attrs, name, default=None):
    return common.scalar(attrs[name]) if name in attrs else default


def _runs(rows, gap=0):
    """Sorted unique rows -> [start, stop) runs; gaps of <= `gap` missing rows are read through."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return rows[:0], rows[:0]
    breaks = np.flatnonzero(np.diff(rows) > gap + 1) + 1
    return rows[np.r_[0, breaks]], rows[np.r_[breaks - 1, len(rows) - 1]] + 1


def _take_runs(block, rows, starts, stops):
    """Rows out of the concatenated run blocks."""
    lengths = stops - starts
    first = np.r_[0, np.cumsum(lengths)[:-1]]
    run = np.searchsorted(starts, rows, side="right") - 1
    return block[rows - starts[run] + first[run]]


# --------------------------------------------------------------------------- snapshot
class Snapshot:
    """A FLAMINGO snapshot: header, cosmology, cell metadata, softening and per-cell gas reads."""

    def __init__(self, store, top_rel, snap, read_from="auto", softening_fallback=(None, None), max_open_chunks=2,
                 chunk_cache_bytes=64 * 2**20):
        self.store, self.top_rel, self.snapshot = store, top_rel, int(snap)
        self.dir_rel = os.path.dirname(top_rel)
        self._cache_bytes = chunk_cache_bytes
        self.file = store.open(top_rel, cache_bytes=chunk_cache_bytes, w0=1.0, nslots=10007)
        self._datasets = {}                  # (file key, name) -> open dataset (keeps its chunk cache between cells)
        header = self.header = dict(self.file["Header"].attrs)
        units = dict(self.file["Units"].attrs) if "Units" in self.file else {}
        self.unit_length_cgs = _unit(units, "Unit length in cgs (U_L)", MPC_CM)
        self.unit_mass_cgs = _unit(units, "Unit mass in cgs (U_M)", MSUN_G * 1e10)
        self.unit_time_cgs = _unit(units, "Unit time in cgs (U_t)", None)
        length = self.unit_length_cgs / MPC_CM                              # internal length -> Mpc (1.0 for FLAMINGO)
        self.a = common.scalar(header["Scale-factor"])
        self.redshift = common.scalar(header["Redshift"])
        self.box_size_cmpc = common.scalar(header["BoxSize"]) * length
        self.n_files = int(common.scalar(header.get("NumFilesPerSnapshot", [1])))
        total = np.asarray(header["NumPart_Total"], dtype=np.int64)
        if "NumPart_Total_HighWord" in header:
            total = total + (np.asarray(header["NumPart_Total_HighWord"], dtype=np.int64) << 32)
        self.n_gas_total = int(total[0])
        self.cosmology = common.cosmology_from_attrs(self.file["Cosmology"].attrs)
        # cell metadata (the same 3D grid as SOAP's)
        meta = dict(self.file["Cells/Meta-data"].attrs)
        self.cell_size_cmpc = np.asarray(meta["size"], dtype=float).reshape(3) * length
        self.dimension = np.asarray(meta["dimension"], dtype=np.int64).reshape(3)
        self._length = length
        self._centres = None                                                # Cells/Centres: loaded on first use
        self.counts = np.asarray(self.file[f"Cells/Counts/{GAS}"][...], dtype=np.int64)
        self.top_files = np.asarray(self.file[f"Cells/Files/{GAS}"][...], dtype=np.int64)
        self.top_offsets = np.asarray(self.file[f"Cells/OffsetsInFile/{GAS}"][...], dtype=np.int64)
        self.n_cells = len(self.counts)
        # softening (internal length units) from the run parameters, else the configured fallback
        params = dict(self.file["Parameters"].attrs) if "Parameters" in self.file else {}
        soft = []
        for key, fallback in zip(("Gravity:comoving_baryon_softening", "Gravity:max_physical_baryon_softening"),
                                 softening_fallback):
            if key in params:
                soft.append(float(common.attr_text(params[key])) * length)
            elif fallback is not None:
                soft.append(float(fallback))
            else:
                soft.append(np.nan)
        self.softening_comoving_mpc, self.softening_max_physical_mpc = soft
        self.softening_source = "Parameters" if all(k in params for k in ("Gravity:comoving_baryon_softening",
                                                                         "Gravity:max_physical_baryon_softening")) else "config"
        self._lookup = None
        self._factors = {}
        self._chunks = OrderedDict()
        self._max_open_chunks = int(max_open_chunks)
        self._chunk_checked = set()
        self._choose_source(read_from)

    # ------------------------------------------------------------------ layout
    def _chunk_template(self):
        stem = f"flamingo_{self.snapshot:04d}"
        for template in (f"{self.dir_rel}/{stem}.{{i}}.hdf5", f"{self.dir_rel}/swift_snapshot_{self.snapshot:04d}/{stem}.{{i}}.hdf5"):
            if self.store.exists(template.format(i=0)):
                return template
        return None

    def _top_has_gas(self):
        if getattr(self, "_top_gas", None) is None:
            try:
                self._top_gas = GAS in self.file and int(self.file[f"{GAS}/Coordinates"].shape[0]) == self.n_gas_total
            except KeyError:
                self._top_gas = False
        return self._top_gas

    def _choose_source(self, read_from):
        if read_from not in ("auto", "chunks", "top"):
            raise ValueError(f"particles.read_from must be auto, chunks or top, not {read_from!r}")
        auxiliary = self._auxiliary = self.auxiliary_fields()
        if auxiliary and read_from == "chunks":
            raise ValueError(f"{self.top_rel}: {auxiliary} are auxiliary-file links in the top-level file (recalculated "
                             f"data); the chunk files hold other values, so read_from='chunks' is refused (use 'auto' or 'top')")
        use_top = read_from == "top" or (read_from == "auto" and (self.store.remote or bool(auxiliary)) and self._top_has_gas())
        template = None if use_top else self._chunk_template()
        if template is None and read_from == "chunks":
            raise FileNotFoundError(f"no chunk files for {self.top_rel} (read_from='chunks')")
        if auxiliary and not use_top:
            raise FileNotFoundError(f"{self.top_rel}: {auxiliary} must be read through the top-level file (auxiliary-file "
                                    f"links), which lacks the full {GAS} datasets")
        # file start of every chunk in the global (virtual) order: particles of all cells, grouped by file
        # (= cumulative NumPart_ThisFile[0] of the preceding chunks; tests/check_chunk_reads.py checks this on real data)
        per_file = np.bincount(self.top_files, weights=self.counts, minlength=max(self.n_files, 1)).astype(np.int64)
        top_starts = np.r_[0, np.cumsum(per_file)[:-1]]
        self.top_file_counts = per_file
        self.top_global_offset = top_starts[self.top_files] + self.top_offsets   # into the top-level PartType0 datasets
        if template is None:
            if not self._top_has_gas():
                raise FileNotFoundError(f"{self.top_rel}: no chunk files and no full {GAS} datasets in the top-level file")
            self.source, self.chunk_template, self.n_chunks = "top", None, 1
            self.cell_file = np.zeros(self.n_cells, dtype=np.int64)
            self.cell_offset = self.top_global_offset
            if not self.store.remote:
                self._check_virtual_sources(self.file)
            return
        self.source, self.chunk_template = "chunks", template
        if self.n_files > 1:              # distributed top-level file: its Cells/Files and OffsetsInFile are per chunk
            self.n_chunks = self.n_files
            self.cell_file, self.cell_offset = self.top_files, self.top_offsets
            self.chunk_file_counts, self.chunk_starts = per_file, top_starts
            return
        chunk0 = self._chunk(0)           # virtual top-level file: per-file mapping from the chunk files
        self.n_chunks = int(common.scalar(chunk0["Header"].attrs["NumFilesPerSnapshot"]))
        files = np.asarray(chunk0[f"Cells/Files/{GAS}"][...], dtype=np.int64)
        offsets = np.asarray(chunk0[f"Cells/OffsetsInFile/{GAS}"][...], dtype=np.int64)
        counts = np.asarray(chunk0[f"Cells/Counts/{GAS}"][...], dtype=np.int64)
        if not np.array_equal(counts, self.counts):
            raise ValueError(f"{self.top_rel}: chunk 0 and top-level Cells/Counts differ")
        chunk_counts = np.bincount(files, weights=counts, minlength=self.n_chunks).astype(np.int64)
        starts = np.r_[0, np.cumsum(chunk_counts)[:-1]]
        if np.array_equal(offsets, self.top_offsets) and files.max() > 0:
            offsets = offsets - starts[files]  # chunk files that carry global offsets
        # the virtual file's global offsets must be the chunk start + offset inside the chunk
        if self.top_files.max() == 0 and not np.array_equal(starts[files] + offsets, self.top_offsets):
            raise ValueError(f"{self.top_rel}: chunk-file offsets do not add up to the top-level global offsets")
        self.cell_file, self.cell_offset = files, offsets
        # particles per chunk and chunk starts in the virtual order, from the cell table (the header's
        # NumPart_ThisFile[0] of each chunk opened is checked against chunk_file_counts in _handle)
        self.chunk_file_counts, self.chunk_starts = chunk_counts, starts

    def auxiliary_fields(self, fields=tuple(GAS_FIELDS)):
        """Pipeline fields whose top-level dataset carries the 'auxiliary file' attribute (see AUXILIARY_ATTR)."""
        try:
            group = self.file[GAS]
        except KeyError:
            return []
        names = set(group.keys())
        return [name for name in fields if name in names and AUXILIARY_ATTR in dict(group[name].attrs)]

    def _check_virtual_sources(self, handle):
        """Local virtual datasets silently return fill values when a source file is missing: refuse that."""
        dataset = handle[f"{GAS}/Coordinates"]
        if not getattr(dataset, "is_virtual", False):
            return
        base = os.path.dirname(self.store.resolve(self.top_rel))
        missing = sorted({s.file_name for s in dataset.virtual_sources()
                          if s.file_name != "." and not os.path.exists(os.path.join(base, s.file_name))
                          and not os.path.exists(s.file_name)})
        if missing:
            raise FileNotFoundError(f"{self.top_rel}: {len(missing)} virtual-dataset source files are missing, e.g. {missing[0]}")

    def _chunk(self, index):
        index = int(index)
        if index in self._chunks:
            self._chunks.move_to_end(index)
            return self._chunks[index]
        handle = self.store.open(self.chunk_template.format(i=index), cache_bytes=self._cache_bytes, w0=1.0, nslots=10007)
        self._chunks[index] = handle
        while len(self._chunks) > max(self._max_open_chunks, 1):
            old_index, old = self._chunks.popitem(last=False)
            for key in [k for k in self._datasets if k[0] == old_index]:
                del self._datasets[key]
            if hasattr(old, "close"):
                old.close()
        return handle

    def _field(self, key, handle, name):
        """PartType0/<name> of an open file, kept open so that its chunk cache survives from cell to cell."""
        ds = self._datasets.get((key, name))
        if ds is None:
            ds = self._datasets[(key, name)] = handle[f"{GAS}/{name}"]
        return ds

    def _handle(self, cell):
        if self.source == "top":
            return self.file
        f = int(self.cell_file[cell])
        handle = self._chunk(f)
        if f not in self._chunk_checked:      # every cell of this chunk must fit in the chunk's own arrays
            n_this = int(np.asarray(handle["Header"].attrs["NumPart_ThisFile"], dtype=np.int64)[0])
            mine = self.cell_file == f
            if np.any(self.cell_offset[mine] + self.counts[mine] > n_this):
                raise ValueError(f"cells of chunk {f} run beyond its NumPart_ThisFile[0] = {n_this}")
            if n_this != int(self.chunk_file_counts[f]):
                raise ValueError(f"chunk {f}: NumPart_ThisFile[0] = {n_this} but its cells hold {int(self.chunk_file_counts[f])}")
            self._chunk_checked.add(f)
        return handle

    def close(self):
        self._datasets.clear()
        for handle in list(self._chunks.values()) + [self.file]:
            if hasattr(handle, "close"):
                handle.close()
        self._chunks.clear()

    # ------------------------------------------------------------------ geometry
    @property
    def cell_centres_cmpc(self):
        if self._centres is None:
            dataset = self.file["Cells/Centres"]
            if self.store.remote:     # regular grid checked against a sample: saves ~6 MB of metadata for L2p8
                self._centres = regular_centres(self.store, dataset, self.cell_size_cmpc, self.dimension, self._length)
            else:
                self._centres = np.asarray(dataset[...], dtype=float).reshape(-1, 3) * self._length
        return self._centres

    @property
    def lookup(self):
        """ijk -> cell index for the regular grid (built from Cells/Centres)."""
        if self._lookup is None:
            self._lookup = cell_lookup(self.cell_centres_cmpc, self.cell_size_cmpc, self.dimension)
        return self._lookup

    def cell_of(self, positions_cmpc):
        return cell_of(positions_cmpc, self.cell_size_cmpc, self.dimension, self.lookup)

    def read_order(self, cells):
        """Cells sorted for sequential I/O: by (file, offset) of the top-level metadata."""
        cells = np.asarray(cells, dtype=np.int64)
        return cells[np.lexsort((self.top_offsets[cells], self.top_files[cells]))]

    # ------------------------------------------------------------------ fields
    def dataset(self, name, handle=None):
        handle = handle if handle is not None else (self.file if self.source == "top" else self._chunk(0))
        return handle[f"{GAS}/{name}"]

    def field_bytes(self, fields=tuple(GAS_FIELDS), membership=None):
        """Stored bytes per particle of the given fields (+ the membership field of the top-level file)."""
        total = 0
        for name in fields:
            ds = self.dataset(name)
            total += np.dtype(ds.dtype).itemsize * int(np.prod(ds.shape[1:]))
        if membership is not None:
            total += np.dtype(self.file[f"{GAS}/{membership}"].dtype).itemsize
        return total

    def factor(self, name):
        """Multiplier from the stored value to the unit of GAS_FIELDS (comoving Mpc for lengths)."""
        if name not in self._factors:
            target, comoving = GAS_FIELDS[name]
            attrs = dict(self.dataset(name).attrs)
            exponent = common.scalar(attrs.get("a-scale exponent", [0.0]))
            if comoving:
                cgs = common.scalar(attrs[COMOVING_ATTR])
            else:
                cgs = common.scalar(attrs[PHYS_ATTR]) if PHYS_ATTR in attrs else common.scalar(attrs[COMOVING_ATTR]) * self.a**exponent
                if COMOVING_ATTR in attrs:
                    expected = common.scalar(attrs[COMOVING_ATTR]) * self.a**exponent
                    if not np.isclose(cgs, expected, rtol=1e-5, atol=0.0):
                        raise ValueError(f"{name}: physical factor {cgs:.6e} != CGS factor x a^{exponent:g} = {expected:.6e}")
            self._factors[name] = cgs / target
        return self._factors[name]

    def read_cell_raw(self, cell, fields=tuple(GAS_FIELDS), source=None):
        """Stored (unconverted) arrays of one cell's gas, from `source` 'chunks' (the chunk file at its per-file
        offset) or 'top' (the top-level datasets at the global offset); default: the source chosen at open."""
        cell = int(cell)
        source = source or self.source
        count = int(self.counts[cell])
        if source == "top":
            if not self._top_has_gas():
                raise FileNotFoundError(f"{self.top_rel}: the top-level file lacks the full {GAS} datasets")
            key, handle, start = "top", self.file, int(self.top_global_offset[cell])
        elif source == "chunks":
            if self.source != "chunks":
                raise FileNotFoundError(f"{self.top_rel}: no chunk files were found (source {self.source!r})")
            key, handle, start = int(self.cell_file[cell]), self._handle(cell), int(self.cell_offset[cell])
        else:
            raise ValueError(f"source must be 'top' or 'chunks', not {source!r}")
        return {name: self.store.read(self._field(key, handle, name), start, start + count) for name in fields}

    # ------------------------------------------------------------------ bound-halo membership
    def membership(self, wanted="auto"):
        """(field, None) of the gas membership dataset in the top-level file, or (None, reason) when it cannot be used.

        `wanted`: 'auto' (MEMBERSHIP_CANDIDATES in order), a dataset name, or None / 'none' (disabled). Usable means:
        an integer dataset with one row per gas particle and, for a local virtual dataset, every source file present
        (a missing source reads as fill values, i.e. halo index 0, silently)."""
        if wanted in (None, "none", False):
            return None, "disabled (particles.membership_field = none)"
        names = MEMBERSHIP_CANDIDATES if wanted == "auto" else (str(wanted),)
        try:
            group = self.file[GAS]
            present = [n for n in names if n in set(group.keys())]
        except KeyError:
            return None, f"the top-level file {self.top_rel} has no {GAS} group"
        if not present:
            return None, f"none of {list(names)} in {self.top_rel} {GAS}"
        name = present[0]
        ds = group[name]
        if len(ds.shape) != 1 or int(ds.shape[0]) != self.n_gas_total:
            return None, f"{GAS}/{name} has shape {tuple(ds.shape)}, not ({self.n_gas_total},)"
        if np.dtype(ds.dtype).kind not in "iu":
            return None, f"{GAS}/{name} has dtype {ds.dtype}, not an integer type"
        if not self.store.remote and getattr(ds, "is_virtual", False):
            base = os.path.dirname(self.store.resolve(self.top_rel))
            missing = sorted({v.file_name for v in ds.virtual_sources() if v.file_name != "."
                              and not os.path.exists(os.path.join(base, v.file_name)) and not os.path.exists(v.file_name)})
            if missing:
                return None, f"{GAS}/{name}: {len(missing)} virtual-dataset source files are missing, e.g. {missing[0]}"
        return name, None

    def read_membership(self, cell, name):
        """Membership values (int64) of one cell's gas, in the order of read_cell (top-level file, global offset)."""
        cell = int(cell)
        start, count = int(self.top_global_offset[cell]), int(self.counts[cell])
        return np.asarray(self.store.read(self._field("top", self.file, name), start, start + count), dtype=np.int64)

    def read_cell(self, cell, fields=tuple(GAS_FIELDS), heating=None, membership=None):
        """Gas of one top-level cell: {field: float64 array in GAS_FIELDS units}.

        With ``heating`` (dict a_limit, tmin_k, tmax_k) the SOAP recently-AGN-heated flag is added as
        ``recently_heated``, evaluated in the stored dtype as stack_products.stream_haloes did; with
        ``membership`` (a field name from membership()) the bound-halo index as ``membership`` (int64).
        """
        names = list(dict.fromkeys(list(fields) + (list(HEATING_FIELDS) if heating is not None else [])))
        raw = self.read_cell_raw(cell, names)
        out = {name: raw[name].astype(np.float64) * self.factor(name) for name in names if name in fields}
        if heating is not None:
            last, temperature = (raw[n] * self.factor(n) for n in HEATING_FIELDS)   # stays float32 (weak scalar)
            out["recently_heated"] = common.recently_heated(last, temperature, heating)
        if membership is not None:
            out["membership"] = self.read_membership(cell, membership)
            if len(out["membership"]) != len(raw[names[0]]):
                raise ValueError(f"cell {cell}: {len(out['membership'])} membership rows for {len(raw[names[0]])} particles")
        return out

    def describe(self):
        return {"top_level": self.store.resolve(self.top_rel), "source": self.source, "n_chunks": int(self.n_chunks),
                "chunk_template": self.chunk_template, "redshift": self.redshift, "scale_factor": self.a,
                "box_size_cmpc": self.box_size_cmpc, "n_cells": int(self.n_cells), "dimension": self.dimension.tolist(),
                "cell_size_cmpc": self.cell_size_cmpc.tolist(), "n_gas_total": self.n_gas_total,
                "softening_comoving_mpc": self.softening_comoving_mpc,
                "softening_max_physical_mpc": self.softening_max_physical_mpc, "softening_source": self.softening_source,
                "auxiliary_fields": self._auxiliary}


# --------------------------------------------------------------------------- SOAP
class Soap:
    """A SOAP halo_properties file. Rows are sorted by top-level cell (Cells/Counts|OffsetsInFile/Subhalos)."""

    def __init__(self, store, rel):
        self.store, self.rel = store, rel
        self.remote = store.remote
        self.file = store.open(rel, cache_bytes=256 * 1024**2)
        self.n_rows = int(self.file["InputHalos/HaloCatalogueIndex"].shape[0])
        self.cell_counts = np.asarray(self.file["Cells/Counts/Subhalos"][...], dtype=np.int64)
        self.cell_offsets = np.asarray(self.file["Cells/OffsetsInFile/Subhalos"][...], dtype=np.int64)
        self._centres = None
        self.cosmology = common.cosmology_from_attrs(self.file["Cosmology"].attrs)

    @property
    def cell_centres_cmpc(self):
        """Cells/Centres in the file's internal length unit (comoving Mpc for FLAMINGO)."""
        if self._centres is None:
            dataset = self.file["Cells/Centres"]
            meta = dict(self.file["Cells/Meta-data"].attrs) if self.has("Cells/Meta-data") else {}
            if self.remote and "size" in meta and "dimension" in meta:
                self._centres = regular_centres(self.store, dataset, np.asarray(meta["size"], float).reshape(3),
                                                np.asarray(meta["dimension"], np.int64).reshape(3))
            else:
                self._centres = np.asarray(dataset[...], dtype=float).reshape(-1, 3)
        return self._centres

    def heating_filter(self):
        """SOAP's definition of recently AGN-heated gas (Parameters/RecentlyHeatedGasFilter), as Python floats."""
        attrs = dict(self.file["Parameters/RecentlyHeatedGasFilter"].attrs)
        out = {"a_limit": common.scalar(attrs["a_limit"]), "tmin_k": common.scalar(attrs["Tmin_in_K"]),
               "tmax_k": common.scalar(attrs["Tmax_in_K"])}
        if "delta_time_in_Myr" in attrs:
            out["delta_time_myr"] = common.scalar(attrs["delta_time_in_Myr"])
        return out

    def has(self, path):
        try:
            self.file[path]
            return True
        except KeyError:
            return False

    def cell_rows(self, cells):
        """(rows, cell of each row) of all subhaloes stored in `cells` (contiguous runs, cell order)."""
        cells = np.unique(np.asarray(cells, dtype=np.int64))
        counts = self.cell_counts[cells]
        rows = np.concatenate([np.arange(o, o + c) for o, c in zip(self.cell_offsets[cells], counts)]) if len(cells) else np.empty(0, np.int64)
        return rows.astype(np.int64), np.repeat(cells, counts)

    def cell_of_rows(self, rows):
        return np.searchsorted(self.cell_offsets, np.asarray(rows), side="right") - 1

    def read(self, path, rows=None, target=None, chunk_rows=8_000_000, gap=None):
        """A column at `rows` (sorted unique; None = all rows, local only), converted if `target` is given.

        Conversion as flamingo_reduced_io.read_rows: raw.astype(float64) * (physical CGS factor / target).
        """
        dataset = self.file[path]
        if rows is None:
            if self.remote:
                raise RuntimeError(f"remote mode: whole SOAP columns are never read ({path}); select rows through the cell index")
            values = np.concatenate([dataset[a:min(a + chunk_rows, self.n_rows)] for a in range(0, self.n_rows, chunk_rows)]) \
                if self.n_rows else dataset[0:0]
        else:
            rows = np.asarray(rows, dtype=np.int64)
            if len(rows) > 1 and np.any(np.diff(rows) <= 0):
                raise ValueError("rows must be sorted and unique")
            starts, stops = _runs(rows, gap=(0 if self.remote else 4096) if gap is None else gap)
            if not self.remote and len(starts) > 200_000:        # very scattered rows: block reads beat per-run reads
                values = self._read_blocked(dataset, rows, chunk_rows)
            else:
                values = _take_runs(self.store.read_slices(dataset, starts, stops), rows, starts, stops)
        if target is None:
            return np.asarray(values)
        return np.asarray(values).astype(np.float64) * (common.physical_factor(dataset) / target)

    @staticmethod
    def _read_blocked(dataset, rows, chunk_rows):
        """Many scattered rows locally: read [first, last] of the requested rows in each block of chunk_rows."""
        out = np.empty((len(rows),) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
        block = rows // chunk_rows
        bounds = np.r_[0, np.flatnonzero(np.diff(block)) + 1, len(rows)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            first, last = int(rows[lo]), int(rows[hi - 1])
            out[lo:hi] = dataset[first:last + 1][rows[lo:hi] - first]
        return out

    def read_columns(self, rows=None, columns=None, optional=OPTIONAL_COLUMNS):
        """{name: values} for CATALOGUE_COLUMNS (or `columns`); missing optional columns are skipped."""
        columns = CATALOGUE_COLUMNS if columns is None else columns
        out = {}
        for name, (path, target) in columns.items():
            if not self.has(path):
                if name in optional:
                    warnings.warn(f"{self.rel}: optional column {path} missing")
                    continue
                raise KeyError(f"{self.rel}: required column {path} missing")
            out[name] = self.read(path, rows, target)
        return out

    def columns(self, names=None, rows=None, cells=None):
        """Converted CATALOGUE_COLUMNS `names` (default all) for all rows (local only), sorted `rows`, or the
        rows of `cells` (through the cell index); the SOAP rows are returned as 'soap_row'."""
        if cells is not None:
            rows, _ = self.cell_rows(cells)
        chosen = CATALOGUE_COLUMNS if names is None else {n: CATALOGUE_COLUMNS[n] for n in names}
        out = self.read_columns(rows, chosen)
        out["soap_row"] = np.arange(self.n_rows, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        return out

    def scan(self, columns, keep, chunk_rows=8_000_000):
        """Rows where keep({name: values of the chunk}) is True, reading `columns` chunk by chunk (local only)."""
        if self.remote:
            raise RuntimeError("remote mode: a full scan of SOAP columns is not allowed")
        found = []
        for start in range(0, self.n_rows, chunk_rows):
            stop = min(start + chunk_rows, self.n_rows)
            values = {}
            for name, (path, target) in columns.items():
                dataset = self.file[path]
                raw = dataset[start:stop]
                values[name] = raw if target is None else raw.astype(np.float64) * (common.physical_factor(dataset) / target)
            found.append(start + np.flatnonzero(keep(values)))
        return np.concatenate(found).astype(np.int64) if found else np.empty(0, np.int64)

    def close(self):
        if hasattr(self.file, "close"):
            self.file.close()


# --------------------------------------------------------------------------- grid helpers
def regular_centres(store, dataset, size, dimension, length=1.0, n_sample=64):
    """Centres (i + 1/2) * size of a C-ordered regular grid, checked against a sample of the stored Cells/Centres
    (first and last cells and an even spread; one request remotely)."""
    size, dimension = np.asarray(size, float), np.asarray(dimension, np.int64)
    n = int(np.prod(dimension))
    centres = (np.stack(np.unravel_index(np.arange(n), tuple(int(d) for d in dimension)), axis=1) + 0.5) * size
    sample = np.unique(np.r_[np.arange(min(8, n)), np.arange(max(n - 8, 0), n), np.linspace(0, n - 1, n_sample).astype(np.int64)])
    if int(dataset.shape[0]) != n:
        raise ValueError(f"Cells/Centres has {dataset.shape[0]} rows, the grid {n}")
    stored = np.asarray(store.read_slices(dataset, sample, sample + 1), dtype=float).reshape(-1, 3) * length
    if not np.allclose(stored, centres[sample], rtol=1e-9, atol=1e-6 * float(size.max())):
        raise ValueError("Cells/Centres is not the C-ordered regular grid (i + 1/2) * size; read it in full instead")
    return centres


def cell_lookup(centres, size, dimension):
    """ijk -> cell index, from the cell centres of a complete regular grid."""
    dimension = np.asarray(dimension, dtype=np.int64)
    ijk = np.floor(np.asarray(centres) / np.asarray(size) + 1e-6).astype(np.int64) % dimension
    lookup = np.full(tuple(dimension), -1, dtype=np.int64)
    lookup[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(len(centres))
    if (lookup < 0).any():
        raise ValueError("the cell grid is not a complete lattice")
    return lookup


def cell_of(positions, size, dimension, lookup):
    ijk = np.floor(np.asarray(positions) / np.asarray(size)).astype(np.int64) % np.asarray(dimension, dtype=np.int64)
    return lookup[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def rho_crit_msun_mpc3(cosmology, unit_mass_cgs=None, unit_length_cgs=None, unit_time_cgs=None):
    """Physical critical density at the snapshot redshift [Msun/Mpc^3] from the Cosmology attributes.

    Uses 'Critical density [internal units]', else 'H [internal units]', else flat LCDM from h and
    Omega_m + Omega_nu_0 (Omega_lambda if given).
    """
    crit = cosmology.get("critical_density_internal", np.nan)
    if np.isfinite(crit) and unit_mass_cgs and unit_length_cgs:
        return crit * unit_mass_cgs / unit_length_cgs**3 * MPC_CM**3 / MSUN_G
    hubble = cosmology.get("H_internal", np.nan)
    if np.isfinite(hubble) and unit_time_cgs:
        h_si = hubble / unit_time_cgs
    else:
        om = cosmology["omega_m_total"]
        ol = cosmology["omega_lambda"] if np.isfinite(cosmology.get("omega_lambda", np.nan)) else 1.0 - om
        h_si = cosmology["h"] * 1e5 / (MPC_CM / 1e2) * np.sqrt(om * (1 + cosmology["redshift"]) ** 3 + ol)
    return 3 * h_si**2 / (8 * np.pi * common.G_SI) * (common.MPC_M**3) / (MSUN_G * 1e-3)
