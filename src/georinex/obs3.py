from collections.abc import Hashable 
from pathlib import Path 
import numpy as np 
import logging 
from datetime import datetime, timedelta 
import io 
import xarray 
import typing as T

try:
    from pymap3d import ecef2geodetic
except ImportError:
    ecef2geodetic = None
#
from .rio import opener, rinexinfo
from .common import determine_time_system, check_time_interval, check_unique_times

"""https://github.com/mvglasow/satstat/wiki/NMEA-IDs"""

SBAS = 100  # offset for ID
GLONASS = 37
QZSS = 192
BEIDOU = 0

__all__ = ["rinexobs3", "obsheader3", "obstime3"]


# def rinexobs3(
#     fn: T.TextIO | Path,
#     use: set[str] | None = None,
#     tlim: tuple[datetime, datetime] | None = None,
#     useindicators: bool = False,
#     meas: list[str] | None = None,
#     verbose: bool = False,
#     *,
#     fast: bool = False,
#     interval: float | int | timedelta | None = None,
# ):
#     """
#     process RINEX 3 OBS data

#     fn: RINEX OBS 3 filename
#     use: 'G'  or ['G', 'R'] or similar

#     tlim: read between these time bounds
#     useindicators: SSI, LLI are output
#     meas:  'L1C'  or  ['L1C', 'C1C'] or similar

#     fast:
#           TODO: FUTURE, not yet enabled for OBS3
#           speculative preallocation based on minimum SV assumption and file size.
#           Avoids double-reading file and more complicated linked lists.
#           Believed that Numpy array should be faster than lists anyway.
#           Reduce Nsvmin if error (let us know)

#     interval: allows decimating file read by time e.g. every 5 seconds.
#                 Useful to speed up reading of very large RINEX files
#     """

#     interval = check_time_interval(interval)

#     if isinstance(use, str):
#         use = {use}

#     if isinstance(meas, str):
#         meas = [meas]

#     if not meas or not meas[0].strip():
#         meas = None
#     # %% allocate
#     # times = obstime3(fn)
#     data = xarray.Dataset({}, coords={"time": [], "sv": []})
#     if tlim is not None and not isinstance(tlim[0], datetime):
#         raise TypeError("time bounds are specified as datetime.datetime")

#     last_epoch = None
#     # %% loop
#     with opener(fn) as f:
#         hdr = obsheader3(f, use, meas)

#         # %% process OBS file
#         time_offset = []
#         for ln in f:
#             if not ln.startswith(">"):  # end of file
#                 break

#             try:
#                 time = _timeobs(ln)
#             except ValueError:  # garbage between header and RINEX data
#                 logging.debug(f"garbage detected in {fn}, trying to parse at next time step")
#                 continue

#             try:
#                 time_offset.append(float(ln[41:56]))
#             except ValueError:
#                 pass
#             # %% get SV indices
#             sv = []
#             raw = ""
#             # Number of visible satellites this time %i3  pg. A13
#             for _ in range(int(ln[33:35])):
#                 ln = f.readline()
#                 sv.append(ln[:3])
#                 raw += ln[3:]

#             if tlim is not None:
#                 if time < tlim[0]:
#                     continue
#                 elif time > tlim[1]:
#                     break

#             if interval is not None:
#                 if last_epoch is None:  # initialization
#                     last_epoch = time
#                 else:
#                     if time - last_epoch < interval:
#                         continue
#                     else:
#                         last_epoch += interval

#             if verbose:
#                 print(time, end="\r")

#             # this time epoch is complete, assemble the data.
#             data = _epoch(data, raw, hdr, time, sv, useindicators, verbose)

#     # %% patch SV names in case of "G 7" => "G07"
#     data = data.assign_coords(sv=[s.replace(" ", "0") for s in data.sv.values.tolist()])
#     # %% other attributes
#     data.attrs["version"] = hdr["version"]

#     # Get interval from header or derive it from the data
#     if "interval" in hdr.keys():
#         data.attrs["interval"] = hdr["interval"]
#     elif "time" in data.coords.keys():
#         # median is robust against gaps
#         try:
#             data.attrs["interval"] = np.median(np.diff(data.time) / np.timedelta64(1, "s"))
#         except TypeError:
#             pass
#     else:
#         data.attrs["interval"] = np.nan

#     data.attrs["rinextype"] = "obs"
#     data.attrs["fast_processing"] = 0  # bool is not allowed in NetCDF4
#     data.attrs["time_system"] = determine_time_system(hdr)
#     if isinstance(fn, Path):
#         data.attrs["filename"] = fn.name

#     if "position" in hdr.keys():
#         data.attrs["position"] = hdr["position"]
#         if ecef2geodetic is not None:
#             data.attrs["position_geodetic"] = hdr["position_geodetic"]

#     if time_offset:
#         data.attrs["time_offset"] = time_offset
#     if "rxmodel" in hdr.keys():
#         data.attrs["rxmodel"] = hdr["rxmodel"]
#     if "RCV CLOCK OFFS APPL" in hdr.keys():
#         try:
#             data.attrs["receiver_clock_offset_applied"] = int(hdr["RCV CLOCK OFFS APPL"])
#         except ValueError:
#             pass
#     if "LEAP SECONDS" in hdr.keys():
#         data.attrs["leap_seconds"] = int(hdr["LEAP SECONDS"].split()[0])
#     return data

def rinexobs3(
    fn: T.TextIO | Path,
    use: set[str] | None = None,
    tlim: tuple[datetime, datetime] | None = None,
    useindicators: bool = False,
    meas: list[str] | None = None,
    verbose: bool = False,
    *,
    fast: bool = False,
    interval: float | int | timedelta | None = None,
):
    """
    Optimized RINEX 3 reader using collectors to avoid per-epoch xarray ops.
    Maintains the same API/behavior as original rinexobs3.
    """
    interval = check_time_interval(interval)

    if isinstance(use, str):
        use = {use}
    if isinstance(meas, str):
        meas = [meas]
    if not meas or not meas[0].strip():
        meas = None

    data = xarray.Dataset({}, coords={"time": [], "sv": []})

    if tlim is not None and not isinstance(tlim[0], datetime):
        raise TypeError("time bounds are specified as datetime.datetime")

    last_epoch = None
    time_offsets = []

    # Open file and read header
    with opener(fn) as f:
        hdr = obsheader3(f, use, meas)

        # Initialize collectors structure
        collectors = {
            "time": [],
            "systems": {},
        }
        for sk in hdr["fields"]:
            collectors["systems"][sk] = {
                "epoch_svs": [],  # list of per-epoch arrays of SV names for this system
                "obs": {obs: [] for obs in hdr["fields"][sk]},
                "lli": {obs: [] for obs in hdr["fields"][sk]},
                "ssi": {obs: [] for obs in hdr["fields"][sk]},
            }

        # loop through file epochs
        for ln in f:
            if not ln.startswith(">"):
                break

            try:
                time = _timeobs(ln)
            except ValueError:
                # garbage between header and data — skip until next epoch mark
                logging.debug("garbage detected in file while parsing epoch")
                continue

            # optional clock offset reading same as original
            try:
                time_offsets.append(float(ln[41:56]))
            except ValueError:
                pass

            # Number of visible satellites on this epoch
            try:
                nsat_this = int(ln[33:35])
            except Exception:
                nsat_this = 0

            sv = []
            raw = ""
            for _ in range(nsat_this):
                l2 = f.readline()
                sv.append(l2[:3])
                raw += l2[3:]

            # time filtering
            if tlim is not None:
                if time < tlim[0]:
                    continue
                if time > tlim[1]:
                    break

            # interval decimation
            if interval is not None:
                if last_epoch is None:
                    last_epoch = time
                else:
                    if time - last_epoch < interval:
                        continue
                    else:
                        last_epoch += interval

            if verbose:
                print(time, end="\r")

            # collect epoch data (fast)
            _collect_epoch(collectors, raw, hdr, time, sv, useindicators)

    # build final dataset from collectors
    data = _build_from_collectors(collectors, hdr, useindicators)

    # patch SV names like original code: "G 7" -> "G07"
    if "sv" in data.coords:
        try:
            data = data.assign_coords(sv=[s.replace(" ", "0") for s in data.sv.values.tolist()])
        except Exception:
            # if sv coordinate is not a simple 1D array, skip patch
            pass

    # copy attributes from header, similar to original
    data.attrs["version"] = hdr.get("version", None)
    if "interval" in hdr:
        data.attrs["interval"] = hdr["interval"]
    else:
        try:
            data.attrs["interval"] = np.median(np.diff(data.time) / np.timedelta64(1, "s"))
        except Exception:
            data.attrs["interval"] = np.nan

    data.attrs["rinextype"] = "obs"
    data.attrs["fast_processing"] = 1
    data.attrs["time_system"] = determine_time_system(hdr)
    if isinstance(fn, Path):
        data.attrs["filename"] = fn.name

    if "position" in hdr:
        data.attrs["position"] = hdr["position"]
        try:
            from pymap3d import ecef2geodetic
        except Exception:
            ecef2geodetic = None
        if ecef2geodetic is not None and "position_geodetic" in hdr:
            data.attrs["position_geodetic"] = hdr["position_geodetic"]

    if time_offsets:
        data.attrs["time_offset"] = time_offsets

    if "RCV CLOCK OFFS APPL" in hdr:
        try:
            data.attrs["receiver_clock_offset_applied"] = int(hdr["RCV CLOCK OFFS APPL"])
        except Exception:
            pass

    return data

def _timeobs(ln: str) -> datetime:
    """
    convert time from RINEX 3 OBS text to datetime
    """

    if not ln.startswith("> "):  # pg. A13
        raise ValueError('RINEX 3 line beginning "> " is not present')

    return datetime(
        int(ln[2:6]),
        int(ln[7:9]),
        int(ln[10:12]),
        hour=int(ln[13:15]),
        minute=int(ln[16:18]),
        second=int(ln[19:21]),
        microsecond=int(float(ln[19:29]) % 1 * 1000000),
    )


def obstime3(fn: T.TextIO | Path, verbose: bool = False):
    """
    return all times in RINEX file
    """

    times = []

    with opener(fn) as f:
        for ln in f:
            if ln.startswith("> "):
                try:
                    times.append(_timeobs(ln))
                except (ValueError, IndexError):
                    logging.debug(f"was not a time:\n{ln}")
                    continue

    times = np.asarray(times, dtype="datetime64[ms]")

    check_unique_times(times)

    return times


def _epoch(
    data: xarray.Dataset,
    raw: str,
    hdr: dict[T.Hashable, T.Any],
    time: datetime,
    sv: list[str],
    useindicators: bool,
    verbose: bool,
) -> xarray.Dataset:
    """
    block processing of each epoch (time step)
    """
    darr = np.atleast_2d(
        np.genfromtxt(io.BytesIO(raw.encode("ascii")), delimiter=(14, 1, 1) * hdr["Fmax"])
    )
    # %% assign data for each time step
    for sk in hdr["fields"]:  # for each satellite system type (G,R,S, etc.)
        # satellite indices "si" to extract from this time's measurements
        si = [i for i, s in enumerate(sv) if s[0] in sk]
        if len(si) == 0:  # no SV of this system "sk" at this time
            continue

        # measurement indices "di" to extract at this time step
        di = hdr["fields_ind"][sk]
        garr = darr[si, :]
        garr = garr[:, di]

        gsv = np.array(sv)[si]

        dsf: dict[str, tuple] = {}
        for i, k in enumerate(hdr["fields"][sk]):
            dsf[k] = (("time", "sv"), np.atleast_2d(garr[:, i * 3]))

            if useindicators:
                dsf = _indicators(dsf, k, garr[:, i * 3 + 1 : i * 3 + 3])

        if verbose:
            print(time, "\r", end="")

        epoch_data = xarray.Dataset(dsf, coords={"time": [time], "sv": gsv})
        if len(data) == 0:
            data = epoch_data
        elif len(hdr["fields"]) == 1:  # one satellite system selected, faster to process
            data = xarray.concat((data, epoch_data), dim="time")
        else:  # general case, slower for different satellite systems all together
            data = xarray.merge((data, epoch_data))

    return data


def _indicators(d: dict, k: str, arr: np.ndarray) -> dict[str, tuple]:
    """
    handle LLI (loss of lock) and SSI (signal strength)
    """
    if k.startswith(("L1", "L2")):
        d[k + "lli"] = (("time", "sv"), np.atleast_2d(arr[:, 0]))

    d[k + "ssi"] = (("time", "sv"), np.atleast_2d(arr[:, 1]))

    return d


def obsheader3(
    f: T.TextIO, use: set[str] | None = None, meas: list[str] | None = None
) -> dict[T.Hashable, T.Any]:
    """
    get RINEX 3 OBS types, for each system type
    optionally, select system type and/or measurement type to greatly
    speed reading and save memory (RAM, disk)
    """
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return obsheader3(h, use, meas)

    fields = {}
    Fmax = 0

    # %% first line
    hdr = rinexinfo(f)

    for ln in f:
        if "END OF HEADER" in ln:
            break

        hd = ln[60:80]
        c = ln[:60]
        if "SYS / # / OBS TYPES" in hd:
            k = c[0]
            fields[k] = c[6:60].split()
            N = int(c[3:6])
            # %% maximum number of fields in a file, to allow fast Numpy parse.
            Fmax = max(N, Fmax)

            n = N - 13
            while n > 0:  # Rinex 3.03, pg. A6, A7
                ln = f.readline()
                assert "SYS / # / OBS TYPES" in ln[60:]
                fields[k] += ln[6:60].split()
                n -= 13

            assert len(fields[k]) == N

            continue

        if hd.strip() not in hdr:  # Header label
            hdr[hd.strip()] = c  # don't strip for fixed-width parsers
            # string with info
        else:  # concatenate to the existing string
            hdr[hd.strip()] += " " + c

    # %% list with x,y,z cartesian (OPTIONAL)
    # Rinex 3.03, pg. A6, Table A2
    try:
        # some RINEX files have bad headers with mulitple APPROX POSITION XYZ.
        # we choose to use the first such header.
        hdr["position"] = [float(j) for j in hdr["APPROX POSITION XYZ"].split()][:3]
        if ecef2geodetic is not None and len(hdr["position"]) == 3:
            hdr["position_geodetic"] = ecef2geodetic(*hdr["position"])
    except (KeyError, ValueError):
        pass
    # %% time
    try:
        t0s = hdr["TIME OF FIRST OBS"]
        # NOTE: must do second=int(float()) due to non-conforming files
        hdr["t0"] = datetime(
            year=int(t0s[:6]),
            month=int(t0s[6:12]),
            day=int(t0s[12:18]),
            hour=int(t0s[18:24]),
            minute=int(t0s[24:30]),
            second=int(float(t0s[30:36])),
            microsecond=int(float(t0s[30:43]) % 1 * 1000000),
        )
    except (KeyError, ValueError):
        pass

    try:
        hdr["interval"] = float(hdr["INTERVAL"][:10])
    except (KeyError, ValueError):
        pass
    # %% select specific satellite systems only (optional)
    if use:
        if not set(fields.keys()).intersection(use):
            raise KeyError(f"system type {use} not found in RINEX file")

        fields = {k: fields[k] for k in use if k in fields}

    # perhaps this could be done more efficiently, but it's probably low impact on overall program.
    # simple set and frozenset operations do NOT preserve order, which would completely mess up reading!
    sysind: dict[str, T.Any] = {}
    if isinstance(meas, (tuple, list, np.ndarray)):
        for sk in fields:  # iterate over each system
            # ind = np.isin(fields[sk], meas)  # boolean vector
            ind = np.zeros(len(fields[sk]), dtype=bool)
            for m in meas:
                for i, field in enumerate(fields[sk]):
                    if field.startswith(m):
                        ind[i] = True

            fields[sk] = np.array(fields[sk])[ind].tolist()
            sysind[sk] = np.empty(Fmax * 3, dtype=bool)  # *3 due to LLI, SSI
            for j, i in enumerate(ind):
                sysind[sk][j * 3 : j * 3 + 3] = i
    else:
        sysind = {k: np.s_[:] for k in fields}

    hdr["fields"] = fields
    hdr["fields_ind"] = sysind
    hdr["Fmax"] = Fmax
    
    try:
        s = " "
        hdr["rxmodel"] = s.join(hdr["REC # / TYPE / VERS"].split()[1:-1])
    except (KeyError, ValueError):
        pass

    return hdr


def _parse_raw_fixedwidth(raw: str, nsat: int, Fmax: int) -> np.ndarray:
    """
    Parse the 'raw' block returned by rinexobs3 into an ndarray shaped (nsat, Fmax*3)
    where each observation field is:
       - value (14 chars) -> float or np.nan
       - LLI indicator (1 char)
       - SSI indicator (1 char)
    The RINEX 3 observation block normally repeats 16 characters per observation (14+1+1).
    """
    # Split into lines (one line per satellite row read earlier)
    rows = raw.splitlines()
    # Some files might wrap observations across multiple lines per satellite. The calling
    # code concatenates lines read from the file; rows should correspond to satellites.
    # defensive: trim empty lines
    rows = [r for r in rows if r]
    if len(rows) != nsat:
        # Fallback: allow rows >= nsat or <= nsat, but prefer the parsed number
        # If inconsistent, fall back to genfromtxt behaviour (rare)
        pass

    # preallocate with NaN
    total_cols = Fmax * 3
    garr = np.full((len(rows), total_cols), np.nan, dtype=float)

    # local references for speed
    _float = float
    _strip = str.strip

    for ri, row in enumerate(rows):
        # make sure row has at least total_cols*? chars, pad if necessary
        # observation chunk width is 16 characters (14 value + 1 LLI + 1 SSI)
        # but the row might be shorter if fewer fields; we only parse what's available
        max_parse = min(Fmax, (len(row) + 15) // 16)
        base = 0
        for oi in range(max_parse):
            start = oi * 16
            vstr = row[start : start + 14]
            if vstr.strip():  # only parse non-empty numeric strings
                # A small try/except protects against rare bad values but is relatively fast
                try:
                    garr[ri, oi * 3] = _float(vstr)
                except ValueError:
                    garr[ri, oi * 3] = np.nan
            # LLI (1 char) and SSI (1 char) are present at start+14 and start+15
            # parse LLI as int if present (0-9 typically) or NaN
            lli_pos = start + 14
            ssi_pos = start + 15
            if lli_pos < len(row):
                lli_char = row[lli_pos]
                if lli_char.strip():
                    # convert to numeric code if wanted (keep float NaN for missing)
                    try:
                        garr[ri, oi * 3 + 1] = _float(lli_char)
                    except ValueError:
                        garr[ri, oi * 3 + 1] = np.nan
            if ssi_pos < len(row):
                ssi_char = row[ssi_pos]
                if ssi_char.strip():
                    try:
                        garr[ri, oi * 3 + 2] = _float(ssi_char)
                    except ValueError:
                        garr[ri, oi * 3 + 2] = np.nan

    return garr

def _parse_epoch_fixedwidth(raw: str, Fmax: int) -> np.ndarray:
    """
    raw = multiple lines of epoch SV blocks, each line containing Fmax obs fields.
    Returns ndarray of shape (num_sv, Fmax*3)
    """
    lines = raw.splitlines()
    out = np.empty((len(lines), Fmax*3), dtype=float)

    for i, ln in enumerate(lines):
        # Each observable is 16 characters: 14 value + LLI + SSI
        # Example positions: [0:14], [14], [15], [16:30], ...
        for k in range(Fmax):
            base = 16*k
            obs = ln[base:base+14].strip()
            lli = ln[base+14:base+15]
            ssi = ln[base+15:base+16]

            out[i, 3*k]   = float(obs) if obs else np.nan
            out[i, 3*k+1] = int(lli) if lli.strip() else -1
            out[i, 3*k+2] = int(ssi) if ssi.strip() else -1

    return out

# def _epoch_fast(
#     data: xarray.Dataset,
#     raw: str,
#     hdr: dict,
#     time: datetime,
#     sv_list: list[str],
#     useindicators: bool,
#     verbose: bool,
# ) -> xarray.Dataset:

#     fields = hdr["fields"]
#     fields_ind = hdr["fields_ind"]
#     Fmax = hdr["Fmax"]      # max # of observables across systems
#     Fcount = len(fields)

#     # ---- FAST FIXED-WIDTH PARSE (replaces genfromtxt) -------------------------
#     try:
#         arr = _parse_epoch_fixedwidth(raw, Fmax)
#     except:
#         print (time)
#     sv_arr = np.array(sv_list)

#     for system in fields:
#         # Find satellites of this GNSS system (e.g., G, R, E)
#         mask = np.char.startswith(sv_arr, system)
#         idx = np.where(mask)[0]
#         if idx.size == 0:
#             continue

#         gsv = sv_arr[idx]
#         # Select only columns relevant to this GNSS system
#         di = fields_ind[system]
#         garr = arr[idx][:, di]

#         # Build variables using vectorized extraction:
#         sys_vars = {}
#         for i, obs in enumerate(fields[system]):
#             # Values (3*k), LLI (3*k+1), SSI (3*k+2)
#             vals = garr[:, 3*i]

#             sys_vars[obs] = (("time", "sv"), vals[None, :])

#             if useindicators:
#                 sys_vars[obs + "lli"] = (("time", "sv"), garr[:, 3*i+1][None, :])
#                 sys_vars[obs + "ssi"] = (("time", "sv"), garr[:, 3*i+2][None, :])

#         sys_ds = xarray.Dataset(sys_vars, coords={"time": [time], "sv": gsv})

#         # ---- Append to master dataset -----------------------------------------
#         if data.sizes.get("time", 0) == 0:
#             data = sys_ds
#         elif Fcount == 1:
#             data = xarray.concat([data, sys_ds], dim="time")
#         else:
#             data = xarray.merge([data, sys_ds], compat="no_conflicts")

#     if verbose:
#         print(time, end="\r")

#     return data

def _collect_epoch(
    collectors: dict,
    raw: str,
    hdr: dict,
    time: datetime,
    sv_list: list[str],
    useindicators: bool,
):
    """
    Parse the epoch block and append values into collectors.
    collectors structure:
    {
      "time": [datetime,...],
      "systems": {
          sk: {
              "epoch_svs": [array_of_sv_for_epoch0, array_of_sv_for_epoch1, ...],
              "obs": {obsname: [1D-array (len=nsat_epoch) , ...] , ...},
              "lli": {obsname: [...], ...},     # only if useindicators
              "ssi": {obsname: [...], ...},
          }, ...
      }
    }
    """
    Fmax = hdr["Fmax"]
    fields = hdr["fields"]
    fields_ind = hdr["fields_ind"]
    try:
        arr = _parse_epoch_fixedwidth(raw, Fmax)
    except:
        arr = np.nan * np.empty((len(raw.splitlines()), Fmax*3), dtype=float)
    sv_arr = np.array(sv_list)  # e.g. ['G07', 'C01', ...]

    # register time
    collectors["time"].append(time)

    # For each system, extract satellite subset and append per-observable arrays
    for sk in fields:
        # boolean mask for satellites of this system (first char is system)
        # use np.char.startswith to handle numpy array of strings
        mask = np.char.startswith(sv_arr, sk)
        idx = np.nonzero(mask)[0]
        
        # if idx.size == 0:
        #     # nothing to collect for this system at this epoch
        #     print ("Notning in this line")
        #     continue

        # selected SV names and slice of arr
        sv_epoch = sv_arr[idx]
        di = fields_ind[sk]  # index selector into 3*Fmax columns (could be slice or boolean)
        garr = arr[idx][:, di]  # shape (nsat_sys, 3 * nobs_sys)

        # append the sv list for this epoch
        collectors["systems"][sk]["epoch_svs"].append(sv_epoch)

        # append each observable's values and optionally indicators
        for i, obs in enumerate(fields[sk]):
            vals = garr[:, i * 3]  # 1D array length nsat_sys
            collectors["systems"][sk]["obs"][obs].append(vals)

            if useindicators:
                collectors["systems"][sk]["lli"][obs].append(garr[:, i * 3 + 1])
                collectors["systems"][sk]["ssi"][obs].append(garr[:, i * 3 + 2])
                
def _build_from_collectors(collectors: dict, hdr: dict, useindicators: bool) -> xarray.Dataset:
    """
    Build one xarray.Dataset per GNSS system using a master SV ordering
    (order of first appearance) and stack per-epoch arrays into shape (ntime, nsv_master).
    Then merge the system datasets and return final Dataset.
    """
    times = np.asarray(collectors["time"], dtype="datetime64[ns]")
    systems = hdr["fields"].keys()
    system_dsets = []
    
    for sk in systems:
        sysc = collectors["systems"][sk]
        if len(sysc["epoch_svs"]) == 0:
            # system not present in file (or filtered out)
            continue

        ntime = len(sysc["epoch_svs"])

        # Build master SV list preserving first appearance order
        master_svs = []
        seen = set()
        for sv_epoch in sysc["epoch_svs"]:
            for s in sv_epoch:
                if s not in seen:
                    seen.add(s)
                    master_svs.append(s)
        nsv = len(master_svs)
        # mapping sv->col index
        sv_to_col = {sv: j for j, sv in enumerate(master_svs)}

        # For each observable create an (ntime, nsv) array filled with NaN and fill per epoch values
        sys_vars = {}
        for obs in hdr["fields"][sk]:
            arr_obs = np.full((ntime, nsv), np.nan, dtype=float)
            # iterate epochs
            for t_idx, vals in enumerate(sysc["obs"][obs]):
                # vals length equals number of SVs in that epoch for this system
                sv_epoch = sysc["epoch_svs"][t_idx]
                for j, s in enumerate(sv_epoch):
                    col = sv_to_col[s]
                    arr_obs[t_idx, col] = vals[j]
            sys_vars[obs] = (("time", "sv"), arr_obs)

            if useindicators:
                # LLI
                arr_lli = np.full((ntime, nsv), np.nan, dtype=float)
                arr_ssi = np.full((ntime, nsv), np.nan, dtype=float)
                for t_idx, vals in enumerate(sysc["lli"][obs]):
                    sv_epoch = sysc["epoch_svs"][t_idx]
                    for j, s in enumerate(sv_epoch):
                        col = sv_to_col[s]
                        arr_lli[t_idx, col] = vals[j]
                for t_idx, vals in enumerate(sysc["ssi"][obs]):
                    sv_epoch = sysc["epoch_svs"][t_idx]
                    for j, s in enumerate(sv_epoch):
                        col = sv_to_col[s]
                        arr_ssi[t_idx, col] = vals[j]
                sys_vars[obs + "lli"] = (("time", "sv"), arr_lli)
                sys_vars[obs + "ssi"] = (("time", "sv"), arr_ssi)
        # for key, value in sys_vars.items():
            # print (np.squeeze(value[1]))
            # print(f"Key: '{key}', Shape: {value[1].shape}")
        # Make sure the time dimensions agree
        
        # Build dataset for this system and set coords (time, sv_master)
        ds = xarray.Dataset(sys_vars, coords={"time": times, "sv": np.array(master_svs)})
        system_dsets.append(ds)

    if len(system_dsets) == 0:
        # empty file
        return xarray.Dataset({}, coords={"time": times, "sv": []})

    # Merge all system datasets once (no_conflicts to mirror original behavior)
    final = xarray.merge(system_dsets, compat="no_conflicts")
    return final
    
# def build_dataset_from_collectors(collectors, hdr, useindicators):
#     datasets = []
#     times = collectors["time"]

#     for system, obsfields in hdr["fields"].items():
#         # If system not present
#         if len(collectors["sv"][system]) == 0:
#             continue

#         # Build SV coordinate: every epoch can have different SV set
#         # xarray allows this as long as dims match variable shapes
#         sv_coord = collectors["sv"][system]

#         # For each observable → stack into (ntime, nsv_ep)
#         sys_vars = {}
#         for obs in obsfields:
#             vals = collectors["obs"][system][obs]
#             vals = [v[np.newaxis, :] for v in vals]
#             sys_vars[obs] = (("time", "sv"), np.concatenate(vals, axis=0))

#             if useindicators:
#                 lli = [v[np.newaxis, :] for v in collectors["lli"][system][obs]]
#                 ssi = [v[np.newaxis, :] for v in collectors["ssi"][system][obs]]
#                 sys_vars[obs+"lli"] = (("time", "sv"), np.concatenate(lli, axis=0))
#                 sys_vars[obs+"ssi"] = (("time", "sv"), np.concatenate(ssi, axis=0))

#         # Build coords
#         ds = xarray.Dataset(
#             sys_vars,
#             coords={
#                 "time": times,
#                 "sv": ["PLACEHOLDER"],   # overwritten next step
#             }
#         )

#         # Replace sv coordinate per-epoch
#         # xarray allows per-time-variable coordinates
#         ds = ds.assign_coords(sv=("time", sv_coord))

#         datasets.append(ds)

#     # Merge all systems once
#     return xarray.merge(datasets, compat="no_conflicts")
