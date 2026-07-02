#!/usr/bin/env python3
"""Rebuild Abacus lensplane patch datasets from HEALPix lightcone products.

This is a parameterized version of the notebook/script path that produced
``lp_dataset_patched`` and ``lp_dataset_patched_c009``:

    HEALPix ASDF particles -> 128x128 overdensity patches
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


NSIDE = 16384
NPIX = 12 * NSIDE * NSIDE
DEFAULT_ROTATIONS = ((45.0, 64.9, 45.0), (71.7, 17.4))


def require_asdf():
    try:
        import asdf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The ASDF stages require the 'asdf' package in this Python environment.") from exc
    return asdf


def require_healpy():
    try:
        import healpy as hp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("The count-projection stage requires the 'healpy' package in this Python environment.") from exc
    return hp


def parse_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def none_or_float(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def phase_heal_dir(raw_root: Path, phase_prefix: str, ph: int) -> Path:
    return raw_root / f"{phase_prefix}ph{ph:03d}" / "heal"


def existing_heal_path(raw_root: Path, phase_prefix: str, ph: int, step: int) -> Path | None:
    heal_dir = phase_heal_dir(raw_root, phase_prefix, ph)
    for lcid in range(3):
        path = heal_dir / f"LightCone{lcid}_heal_Step{step:04d}.asdf"
        if path.exists():
            return path
    return None


def read_shell_metadata(raw_root: Path, phase_prefix: str, ph: int, step: int) -> dict[str, float] | None:
    asdf = require_asdf()
    path = existing_heal_path(raw_root, phase_prefix, ph, step)
    if path is None:
        return None
    with asdf.open(path) as handle:
        a = float(handle["header"]["ScaleFactor"])
        return {
            "a": a,
            "z": (1.0 / a) - 1.0,
            "dist": float(handle["header"]["CoordinateDistanceHMpc"]),
            "nbar": float(handle["header"]["NP"]) / (float(handle["header"]["BoxSizeHMpc"]) ** 3),
        }


def available_steps(raw_root: Path, phase_prefix: str, ph: int, step_source: str = "lightcone0") -> list[int]:
    pattern = "LightCone*_heal_Step*.asdf" if step_source == "any" else "LightCone0_heal_Step*.asdf"
    files = sorted((phase_heal_dir(raw_root, phase_prefix, ph)).glob(pattern))
    steps = set()
    for path in files:
        stem = path.stem
        steps.add(int(stem.rsplit("Step", 1)[1]))
    return sorted(steps)


def available_phases(raw_root: Path, phase_prefix: str) -> list[int]:
    phases = []
    for path in sorted(raw_root.glob(f"{phase_prefix}ph*/heal")):
        name = path.parent.name
        if not name.startswith(phase_prefix + "ph"):
            continue
        phases.append(int(name.removeprefix(phase_prefix + "ph")))
    return phases


def selected_steps(args: argparse.Namespace, ph: int) -> list[int]:
    steps = available_steps(args.raw_root, args.phase_prefix, ph, args.step_source)
    if args.step_min is not None:
        steps = [step for step in steps if step >= args.step_min]
    if args.step_max is not None:
        steps = [step for step in steps if step <= args.step_max]
    return sorted(steps, reverse=True)


def histogram_hp(rho: np.ndarray, pix_ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(pix_ids, minlength=len(rho))
    return rho + counts.astype(rho.dtype)


def projected_count_maps(
    ph: int,
    step: int,
    raw_root: Path,
    phase_prefix: str,
    xsize: int,
    reso_arcmin: float,
    projected_lightcones: list[int],
) -> tuple[dict[int, np.ndarray], float | None, float | None]:
    asdf = require_asdf()
    hp = require_healpy()
    rho = np.zeros(NPIX, dtype=np.float32)
    radius = None
    scale_factor = None
    heal_dir = phase_heal_dir(raw_root, phase_prefix, ph)
    for cone_id in range(3):
        path = heal_dir / f"LightCone{cone_id}_heal_Step{step:04d}.asdf"
        if not path.exists():
            continue
        with asdf.open(path) as handle:
            rho = histogram_hp(rho, handle["data"]["heal"][:])
            radius = handle["header"]["CoordinateDistanceHMpc"]
            scale_factor = handle["header"]["ScaleFactor"]

    if radius is None or scale_factor is None:
        return {}, None, None

    projected = {}
    for lcid in projected_lightcones:
        rotation = DEFAULT_ROTATIONS[lcid - 1]
        projected[lcid] = hp.gnomview(
            rho,
            rot=list(rotation),
            xsize=xsize,
            reso=reso_arcmin,
            nest=True,
            return_projected_map=True,
            no_plot=True,
        )
    return projected, float(radius), float(scale_factor)


def project_counts_for_step(args: tuple[int, int, str, str, str, int, float, str, bool]) -> None:
    ph, step, raw_root_s, phase_prefix, counts_root_s, xsize, reso_arcmin, lightcones_s, skip_existing = args
    raw_root = Path(raw_root_s)
    counts_root = Path(counts_root_s)
    lightcones = parse_ints(lightcones_s)
    out_paths = [counts_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz" for lcid in lightcones]
    if skip_existing and all(path.exists() for path in out_paths):
        return

    maps, radius, scale_factor = projected_count_maps(
        ph=ph,
        step=step,
        raw_root=raw_root,
        phase_prefix=phase_prefix,
        xsize=xsize,
        reso_arcmin=reso_arcmin,
        projected_lightcones=lightcones,
    )
    if radius is None or scale_factor is None:
        return

    counts_root.mkdir(parents=True, exist_ok=True)
    for lcid, projected in maps.items():
        np.savez_compressed(
            counts_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz",
            data=projected,
            r=radius,
            a=scale_factor,
        )


def build_projected_counts(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    tasks = []
    for ph in phases:
        for step in selected_steps(args, ph):
            tasks.append(
                (
                    ph,
                    step,
                    str(args.raw_root),
                    args.phase_prefix,
                    str(args.counts_root),
                    args.xsize,
                    args.reso_arcmin,
                    ",".join(str(lcid) for lcid in lightcones),
                    args.skip_existing,
                )
            )

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            list(tqdm(pool.imap_unordered(project_counts_for_step, tasks), total=len(tasks), desc="counts"))
    else:
        for task in tqdm(tasks, desc="counts"):
            project_counts_for_step(task)


def build_overdensity_shells(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    asdf = require_asdf()
    args.od_root.mkdir(parents=True, exist_ok=True)
    for ph in tqdm(phases, desc="phases"):
        steps = selected_steps(args, ph)
        last_dist = None
        for step in tqdm(steps, leave=False, desc=f"ph{ph:03d} shells"):
            meta_path = existing_heal_path(args.raw_root, args.phase_prefix, ph, step)
            if meta_path is None:
                continue
            with asdf.open(meta_path) as handle:
                a = float(handle["header"]["ScaleFactor"])
                z = (1.0 / a) - 1.0
                current_dist = float(handle["header"]["CoordinateDistanceHMpc"])
                nbar = float(handle["header"]["NP"]) / (float(handle["header"]["BoxSizeHMpc"]) ** 3)

            if z > args.z_max:
                break
            if last_dist is None:
                last_dist = current_dist
                continue

            dchi = current_dist - last_dist
            chi = 0.5 * (current_dist + last_dist)
            expected_per_pix = nbar * (4.0 * np.pi / NPIX) * (chi**2) * dchi
            for lcid in lightcones:
                out_path = args.od_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz"
                if args.skip_existing and out_path.exists():
                    continue
                counts_path = args.counts_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz"
                if not counts_path.exists():
                    print(f"missing counts: {counts_path}")
                    continue
                counts = np.load(counts_path)["data"]
                od = (counts / expected_per_pix) - 1.0
                np.savez_compressed(out_path, od=od, a=a, chi=chi, dchi=dchi, epp=expected_per_pix)
            last_dist = current_dist


def build_overdensity_shells_direct(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    asdf = require_asdf()
    args.od_root.mkdir(parents=True, exist_ok=True)
    for ph in tqdm(phases, desc="phases"):
        steps = selected_steps(args, ph)
        last_dist = None
        for step in tqdm(steps, leave=False, desc=f"ph{ph:03d} direct shells"):
            meta_path = existing_heal_path(args.raw_root, args.phase_prefix, ph, step)
            if meta_path is None:
                continue
            with asdf.open(meta_path) as handle:
                a = float(handle["header"]["ScaleFactor"])
                z = (1.0 / a) - 1.0
                current_dist = float(handle["header"]["CoordinateDistanceHMpc"])
                nbar = float(handle["header"]["NP"]) / (float(handle["header"]["BoxSizeHMpc"]) ** 3)

            if z > args.z_max:
                break
            if last_dist is None:
                last_dist = current_dist
                continue

            pending_lightcones = [
                lcid
                for lcid in lightcones
                if not (args.skip_existing and (args.od_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz").exists())
            ]
            if not pending_lightcones:
                last_dist = current_dist
                continue

            maps, _, _ = projected_count_maps(
                ph=ph,
                step=step,
                raw_root=args.raw_root,
                phase_prefix=args.phase_prefix,
                xsize=args.xsize,
                reso_arcmin=args.reso_arcmin,
                projected_lightcones=pending_lightcones,
            )
            if not maps:
                last_dist = current_dist
                continue

            dchi = current_dist - last_dist
            chi = 0.5 * (current_dist + last_dist)
            expected_per_pix = nbar * (4.0 * np.pi / NPIX) * (chi**2) * dchi
            for lcid, counts in maps.items():
                od = (counts / expected_per_pix) - 1.0
                np.savez_compressed(
                    args.od_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz",
                    od=od,
                    a=a,
                    chi=chi,
                    dchi=dchi,
                    epp=expected_per_pix,
                )
            last_dist = current_dist


def uniform_chi_edges(chi_min: float, chi_max: float, n_lp: int) -> np.ndarray:
    return np.linspace(chi_min, chi_max, n_lp + 1)


def split_assign_for_shell(center: float, width: float, edges: np.ndarray):
    left_edge = center - 0.5 * width
    right_edge = center + 0.5 * width
    k0 = np.searchsorted(edges, left_edge, side="right") - 1
    k1 = np.searchsorted(edges, right_edge, side="left")
    for k in range(max(k0, 0), min(k1, len(edges) - 1)):
        left = max(left_edge, edges[k])
        right = min(right_edge, edges[k + 1])
        overlap = right - left
        if overlap > 0:
            yield k, overlap


def robust_stats(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    if mask is None:
        mu = float(np.mean(arr))
        return mu, float(np.sqrt(max(np.var(arr), 0.0)))
    w = mask.astype(np.float64)
    total = w.sum()
    if total <= 0:
        return 0.0, 0.0
    mu = float((arr * w).sum() / total)
    var = float(((arr - mu) ** 2 * w).sum() / total)
    return mu, float(np.sqrt(max(var, 0.0)))


def clean_shell_map(
    od_i: np.ndarray,
    mask: np.ndarray | None,
    clip_sigma: float | None,
    hard_clip: float | None,
    demean: bool,
) -> np.ndarray:
    x = od_i.astype(np.float64, copy=False)
    x = np.where(np.isfinite(x), x, 0.0)
    mu, sigma = robust_stats(x, mask)
    if clip_sigma is not None and sigma > 0:
        x = np.clip(x, mu - clip_sigma * sigma, mu + clip_sigma * sigma)
    if hard_clip is not None:
        x = np.clip(x, -hard_clip, hard_clip)
    if demean:
        if mask is None:
            x = x - mu
        else:
            w = mask.astype(np.float64)
            x = x - ((x * w).sum() / max(w.sum(), 1e-30))
    return x


def winsorize_plane(arr: np.ndarray, q: float) -> np.ndarray:
    return np.clip(arr, np.quantile(arr, 1 - q), np.quantile(arr, q))


def build_lensplanes_for_phase(args: argparse.Namespace, ph: int, lcid: int, shell_mask: np.ndarray | None) -> None:
    files = sorted(args.od_root.glob(f"ph{ph:03d}_lc{lcid}_step*.npz"))
    if not files:
        raise FileNotFoundError(f"No shells found for ph={ph:03d}, lc={lcid} under {args.od_root}")

    chis, dchis, scale_factors = [], [], []
    for path in files:
        with np.load(path) as npz:
            chis.append(float(npz["chi"]))
            dchis.append(float(npz["dchi"]))
            scale_factors.append(float(npz["a"]))

    edges = uniform_chi_edges(args.chi_min, args.chi_max, args.n_lensplanes)
    with np.load(files[0]) as npz0:
        od_shape = npz0["od"].shape

    od_int = [np.zeros(od_shape, dtype=np.float64) for _ in range(args.n_lensplanes)]
    od_weighted = [np.zeros(od_shape, dtype=np.float64) for _ in range(args.n_lensplanes)]
    dchi_tot = np.zeros(args.n_lensplanes, dtype=np.float64)
    wsum = np.zeros(args.n_lensplanes, dtype=np.float64)
    w_chi_sum = np.zeros(args.n_lensplanes, dtype=np.float64)
    w_inva_sum = np.zeros(args.n_lensplanes, dtype=np.float64)

    for path, chi_i, dchi_i, a_i in tqdm(
        zip(files, chis, dchis, scale_factors), total=len(files), leave=False, desc=f"ph{ph:03d} lc{lcid}"
    ):
        with np.load(path) as npz:
            od_i = clean_shell_map(npz["od"], shell_mask, args.shell_clip_sigma, args.shell_hard_clip, args.per_shell_demean)

        shell_overlap = 0.0
        geo = chi_i / max(a_i, 1e-12)
        for k, overlap in split_assign_for_shell(chi_i, dchi_i, edges):
            od_int[k] += od_i * overlap
            od_weighted[k] += od_i * overlap * geo
            dchi_tot[k] += overlap
            w = overlap * geo
            wsum[k] += w
            w_chi_sum[k] += w * chi_i
            w_inva_sum[k] += w / max(a_i, 1e-12)
            shell_overlap += overlap
        if args.assert_overlaps and not np.isclose(shell_overlap, dchi_i, rtol=1e-10, atol=1e-8):
            raise AssertionError(f"Shell overlap {shell_overlap} != dchi {dchi_i} for {path}")

    chi_eff = np.zeros(args.n_lensplanes, dtype=np.float64)
    a_eff = np.ones(args.n_lensplanes, dtype=np.float64)
    for k in range(args.n_lensplanes):
        if wsum[k] > 0:
            chi_eff[k] = w_chi_sum[k] / wsum[k]
            a_eff[k] = 1.0 / max(w_inva_sum[k] / wsum[k], 1e-30)
        else:
            chi_eff[k] = 0.5 * (edges[k] + edges[k + 1])

    for k in range(args.n_lensplanes):
        if args.per_plane_demean:
            if shell_mask is None:
                od_int[k] -= od_int[k].mean()
                od_weighted[k] -= od_weighted[k].mean()
            else:
                w = shell_mask.astype(np.float64)
                od_int[k] -= (od_int[k] * w).sum() / max(w.sum(), 1e-30)
                od_weighted[k] -= (od_weighted[k] * w).sum() / max(w.sum(), 1e-30)
        if args.plane_winsor_q is not None:
            od_int[k] = winsorize_plane(od_int[k], args.plane_winsor_q)
            od_weighted[k] = winsorize_plane(od_weighted[k], args.plane_winsor_q)

    out_dir = args.lensplane_root / f"ph{ph:03d}_lc{lcid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    g_factor = (chi_eff / np.maximum(a_eff, 1e-30)) * dchi_tot
    np.savez_compressed(
        out_dir / "lensplanes_index.npz",
        edges=edges,
        chi_eff=chi_eff,
        a_eff=a_eff,
        dchi_tot=dchi_tot,
        G=g_factor,
        ph=ph,
        lcid=lcid,
    )
    for k in range(args.n_lensplanes):
        od_int_f32 = od_int[k].astype(np.float32)
        od_weighted_f32 = od_weighted[k].astype(np.float32)
        od_eff = od_int_f32.astype(np.float64) / dchi_tot[k]
        np.savez_compressed(
            out_dir / f"lensplane_{k:02d}.npz",
            od_int=od_int_f32,
            od_weighted=od_weighted_f32,
            od_eff=od_eff,
            chi_eff=chi_eff[k],
            a_eff=a_eff[k],
            dchi_tot=dchi_tot[k],
            G=g_factor[k],
            bin_left=edges[k],
            bin_right=edges[k + 1],
        )


def build_lensplanes(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    shell_mask = np.load(args.shell_mask) if args.shell_mask else None
    for ph in tqdm(phases, desc="lensplane phases"):
        for lcid in tqdm(lightcones, leave=False, desc=f"ph{ph:03d} lightcones"):
            index_path = args.lensplane_root / f"ph{ph:03d}_lc{lcid}" / "lensplanes_index.npz"
            if args.skip_existing and index_path.exists():
                continue
            build_lensplanes_for_phase(args, ph, lcid, shell_mask)


def get_patch(arr: np.ndarray, idx: int, patch_size: int, patches_per_side: int) -> np.ndarray:
    row = idx // patches_per_side
    col = idx % patches_per_side
    r0, r1 = row * patch_size, (row + 1) * patch_size
    c0, c1 = col * patch_size, (col + 1) * patch_size
    return arr[r0:r1, c0:c1][None, ...]


def patchify_lensplanes(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    patches_per_side = args.xsize // args.patch_size
    n_patches = patches_per_side * patches_per_side
    for plane in range(args.n_lensplanes):
        (args.patch_root / f"{plane:02d}").mkdir(parents=True, exist_ok=True)

    for ph in tqdm(phases, desc="patch phases"):
        for lcid in tqdm(lightcones, leave=False, desc=f"ph{ph:03d} lightcones"):
            for plane in tqdm(range(args.n_lensplanes), leave=False, desc="planes"):
                data_path = args.lensplane_root / f"ph{ph:03d}_lc{lcid}" / f"lensplane_{plane:02d}.npz"
                data_full = np.load(data_path)["od_eff"]
                for patch_idx in range(n_patches):
                    out_path = args.patch_root / f"{plane:02d}" / f"ph{ph:03d}_lc{lcid}_{patch_idx:03d}.npy"
                    if args.skip_existing and out_path.exists():
                        continue
                    np.save(out_path, get_patch(data_full, patch_idx, args.patch_size, patches_per_side))


def all_patches_exist(args: argparse.Namespace, ph: int, lcid: int) -> bool:
    patches_per_side = args.xsize // args.patch_size
    n_patches = patches_per_side * patches_per_side
    for plane in range(args.n_lensplanes):
        for patch_idx in range(n_patches):
            path = args.patch_root / f"{plane:02d}" / f"ph{ph:03d}_lc{lcid}_{patch_idx:03d}.npy"
            if not path.exists():
                return False
    return True


def write_patch_dataset_from_planes(
    args: argparse.Namespace,
    ph: int,
    lcid: int,
    od_int: list[np.ndarray],
    dchi_tot: np.ndarray,
    shell_mask: np.ndarray | None,
) -> None:
    patches_per_side = args.xsize // args.patch_size
    n_patches = patches_per_side * patches_per_side
    for plane in range(args.n_lensplanes):
        if dchi_tot[plane] <= 0:
            raise ValueError(f"ph{ph:03d} lc{lcid} plane{plane:02d} has zero accumulated dchi")

        if args.per_plane_demean:
            if shell_mask is None:
                od_int[plane] -= od_int[plane].mean()
            else:
                w = shell_mask.astype(np.float64)
                od_int[plane] -= (od_int[plane] * w).sum() / max(w.sum(), 1e-30)
        if args.plane_winsor_q is not None:
            od_int[plane] = winsorize_plane(od_int[plane], args.plane_winsor_q)

        # Match the historical final patches: od_eff came from a stored
        # float32 od_int plane divided in float64 by dchi_tot.
        od_eff = od_int[plane].astype(np.float32).astype(np.float64) / dchi_tot[plane]
        out_dir = args.patch_root / f"{plane:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for patch_idx in range(n_patches):
            out_path = out_dir / f"ph{ph:03d}_lc{lcid}_{patch_idx:03d}.npy"
            if args.skip_existing and out_path.exists():
                continue
            np.save(out_path, get_patch(od_eff, patch_idx, args.patch_size, patches_per_side))


def metadata_path(args: argparse.Namespace, ph: int) -> Path:
    return args.metadata_root / args.dataset_name / "lensplanes_index" / f"ph{ph:03d}_lensplanes_index.npz"


def phase_lensplane_metadata(
    args: argparse.Namespace,
    ph: int,
    lcid: int,
) -> dict[str, np.ndarray | float | int]:
    edges = uniform_chi_edges(args.chi_min, args.chi_max, args.n_lensplanes)
    dchi_tot = np.zeros(args.n_lensplanes, dtype=np.float64)
    wsum = np.zeros(args.n_lensplanes, dtype=np.float64)
    w_chi_sum = np.zeros(args.n_lensplanes, dtype=np.float64)
    w_inva_sum = np.zeros(args.n_lensplanes, dtype=np.float64)
    steps = selected_steps(args, ph)
    if not steps:
        raise FileNotFoundError(f"No ASDF steps found for ph={ph:03d} under {phase_heal_dir(args.raw_root, args.phase_prefix, ph)}")

    asdf = require_asdf()
    last_dist = None
    for step in steps:
        meta_path = existing_heal_path(args.raw_root, args.phase_prefix, ph, step)
        if meta_path is None:
            continue
        with asdf.open(meta_path) as handle:
            a = float(handle["header"]["ScaleFactor"])
            z = (1.0 / a) - 1.0
            current_dist = float(handle["header"]["CoordinateDistanceHMpc"])

        if z > args.z_max:
            break
        if last_dist is None:
            last_dist = current_dist
            continue

        dchi = current_dist - last_dist
        chi = 0.5 * (current_dist + last_dist)
        overlaps = list(split_assign_for_shell(chi, dchi, edges))
        shell_overlap = sum(overlap for _, overlap in overlaps)
        if args.assert_overlaps and not np.isclose(shell_overlap, dchi, rtol=1e-10, atol=1e-8):
            raise AssertionError(f"Shell overlap {shell_overlap} != dchi {dchi} for ph{ph:03d} step{step:04d}")

        geo = chi / max(a, 1e-12)
        for plane, overlap in overlaps:
            dchi_tot[plane] += overlap
            w = overlap * geo
            wsum[plane] += w
            w_chi_sum[plane] += w * chi
            w_inva_sum[plane] += w / max(a, 1e-12)
        last_dist = current_dist

    if not np.any(dchi_tot > 0):
        raise ValueError(f"ph{ph:03d} has zero accumulated lensplane thickness")

    chi_eff = np.zeros(args.n_lensplanes, dtype=np.float64)
    a_eff = np.ones(args.n_lensplanes, dtype=np.float64)
    for plane in range(args.n_lensplanes):
        if wsum[plane] > 0:
            chi_eff[plane] = w_chi_sum[plane] / wsum[plane]
            a_eff[plane] = 1.0 / max(w_inva_sum[plane] / wsum[plane], 1e-30)
        else:
            chi_eff[plane] = 0.5 * (edges[plane] + edges[plane + 1])

    g_factor = (chi_eff / np.maximum(a_eff, 1e-30)) * dchi_tot
    return {
        "edges": edges,
        "chi_eff": chi_eff,
        "a_eff": a_eff,
        "dchi_tot": dchi_tot,
        "G": g_factor,
        "ph": ph,
        "lcid": lcid,
    }


def write_phase_metadata(args: argparse.Namespace, ph: int, lightcones: list[int]) -> None:
    out_path = metadata_path(args, ph)
    if args.skip_existing and out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lcid = lightcones[0] if lightcones else 1
    np.savez_compressed(out_path, **phase_lensplane_metadata(args, ph, lcid))


def build_metadata(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    for ph in tqdm(phases, desc="metadata phases"):
        write_phase_metadata(args, ph, lightcones)


def build_patches_direct_from_asdf(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    asdf = require_asdf()
    shell_mask = np.load(args.shell_mask) if args.shell_mask else None
    edges = uniform_chi_edges(args.chi_min, args.chi_max, args.n_lensplanes)
    od_shape = (args.xsize, args.xsize)
    if shell_mask is not None and shell_mask.shape != od_shape:
        raise ValueError(f"shell_mask shape {shell_mask.shape} != projected map shape {od_shape}")

    for ph in tqdm(phases, desc="direct patch phases"):
        if args.write_metadata:
            write_phase_metadata(args, ph, lightcones)
        active_lightcones = [
            lcid for lcid in lightcones if not (args.skip_existing and all_patches_exist(args, ph, lcid))
        ]
        if not active_lightcones:
            continue

        od_int_by_lc = {
            lcid: [np.zeros(od_shape, dtype=np.float64) for _ in range(args.n_lensplanes)]
            for lcid in active_lightcones
        }
        dchi_tot = np.zeros(args.n_lensplanes, dtype=np.float64)
        steps = selected_steps(args, ph)
        last_dist = None

        for step in tqdm(steps, leave=False, desc=f"ph{ph:03d} ASDF -> patches"):
            meta_path = existing_heal_path(args.raw_root, args.phase_prefix, ph, step)
            if meta_path is None:
                continue
            with asdf.open(meta_path) as handle:
                a = float(handle["header"]["ScaleFactor"])
                z = (1.0 / a) - 1.0
                current_dist = float(handle["header"]["CoordinateDistanceHMpc"])
                nbar = float(handle["header"]["NP"]) / (float(handle["header"]["BoxSizeHMpc"]) ** 3)

            if z > args.z_max:
                break
            if last_dist is None:
                last_dist = current_dist
                continue

            dchi = current_dist - last_dist
            chi = 0.5 * (current_dist + last_dist)
            expected_per_pix = nbar * (4.0 * np.pi / NPIX) * (chi**2) * dchi
            overlaps = list(split_assign_for_shell(chi, dchi, edges))
            shell_overlap = sum(overlap for _, overlap in overlaps)
            if args.assert_overlaps and not np.isclose(shell_overlap, dchi, rtol=1e-10, atol=1e-8):
                raise AssertionError(f"Shell overlap {shell_overlap} != dchi {dchi} for ph{ph:03d} step{step:04d}")

            if not overlaps:
                last_dist = current_dist
                continue

            maps, _, _ = projected_count_maps(
                ph=ph,
                step=step,
                raw_root=args.raw_root,
                phase_prefix=args.phase_prefix,
                xsize=args.xsize,
                reso_arcmin=args.reso_arcmin,
                projected_lightcones=active_lightcones,
            )
            for lcid, counts in maps.items():
                od = (counts / expected_per_pix) - 1.0
                od = clean_shell_map(
                    od,
                    shell_mask,
                    args.shell_clip_sigma,
                    args.shell_hard_clip,
                    args.per_shell_demean,
                )
                for plane, overlap in overlaps:
                    od_int_by_lc[lcid][plane] += od * overlap
            for plane, overlap in overlaps:
                dchi_tot[plane] += overlap
            last_dist = current_dist

        for lcid in active_lightcones:
            write_patch_dataset_from_planes(args, ph, lcid, od_int_by_lc[lcid], dchi_tot, shell_mask)


def compare_arrays(label: str, actual: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape mismatch {actual.shape} != {expected.shape}")
    diff = np.asarray(actual) - np.asarray(expected)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    ok = np.allclose(actual, expected, rtol=rtol, atol=atol)
    print(f"{label}: max_abs={max_abs:.6g} rms={rms:.6g} allclose={ok}")
    if not ok:
        raise AssertionError(f"{label} failed allclose(rtol={rtol}, atol={atol})")


def validate_direct_od(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    steps_to_check = parse_ints(args.validate_steps)
    if not steps_to_check:
        raise ValueError("--validate-steps is required for validate-od")

    for ph in phases:
        steps = selected_steps(args, ph)
        step_to_outer = {steps[i]: steps[i - 1] for i in range(1, len(steps))}
        for step in steps_to_check:
            if step not in step_to_outer:
                raise ValueError(f"ph{ph:03d} step {step:04d} has no outer boundary step in the selected step grid")

            current = read_shell_metadata(args.raw_root, args.phase_prefix, ph, step)
            outer = read_shell_metadata(args.raw_root, args.phase_prefix, ph, step_to_outer[step])
            if current is None or outer is None:
                raise FileNotFoundError(f"Missing ASDF metadata for ph{ph:03d} step {step:04d}")
            if current["z"] > args.z_max:
                print(f"ph{ph:03d} step {step:04d}: skipped because z={current['z']:.4f} > z_max={args.z_max}")
                continue

            maps, _, _ = projected_count_maps(
                ph=ph,
                step=step,
                raw_root=args.raw_root,
                phase_prefix=args.phase_prefix,
                xsize=args.xsize,
                reso_arcmin=args.reso_arcmin,
                projected_lightcones=lightcones,
            )
            dchi = current["dist"] - outer["dist"]
            chi = 0.5 * (current["dist"] + outer["dist"])
            expected_per_pix = current["nbar"] * (4.0 * np.pi / NPIX) * (chi**2) * dchi
            for lcid, counts in maps.items():
                od = (counts / expected_per_pix) - 1.0
                existing_path = args.od_root / f"ph{ph:03d}_lc{lcid}_step{step:04d}.npz"
                with np.load(existing_path) as existing:
                    compare_arrays(
                        f"od ph{ph:03d} lc{lcid} step{step:04d}",
                        od,
                        existing["od"],
                        args.validate_rtol,
                        args.validate_atol,
                    )


def validate_lensplanes(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    planes = parse_ints(args.validate_planes)
    if not planes:
        raise ValueError("--validate-planes is required for validate-lensplanes")
    shell_mask = np.load(args.shell_mask) if args.shell_mask else None
    edges = uniform_chi_edges(args.chi_min, args.chi_max, args.n_lensplanes)

    for ph in phases:
        for lcid in lightcones:
            files = sorted(args.od_root.glob(f"ph{ph:03d}_lc{lcid}_step*.npz"))
            if not files:
                raise FileNotFoundError(f"No shells found for ph={ph:03d}, lc={lcid} under {args.od_root}")

            with np.load(files[0]) as npz0:
                od_shape = npz0["od"].shape
            od_int = {plane: np.zeros(od_shape, dtype=np.float64) for plane in planes}
            dchi_tot = {plane: 0.0 for plane in planes}

            for path in tqdm(files, leave=False, desc=f"validate ph{ph:03d} lc{lcid}"):
                with np.load(path) as npz:
                    chi_i = float(npz["chi"])
                    dchi_i = float(npz["dchi"])
                    overlaps = [(k, ov) for k, ov in split_assign_for_shell(chi_i, dchi_i, edges) if k in od_int]
                    if not overlaps:
                        continue
                    od_i = clean_shell_map(
                        npz["od"],
                        shell_mask,
                        args.shell_clip_sigma,
                        args.shell_hard_clip,
                        args.per_shell_demean,
                    )
                for k, overlap in overlaps:
                    od_int[k] += od_i * overlap
                    dchi_tot[k] += overlap

            for plane in planes:
                if args.per_plane_demean:
                    if shell_mask is None:
                        od_int[plane] -= od_int[plane].mean()
                    else:
                        w = shell_mask.astype(np.float64)
                        od_int[plane] -= (od_int[plane] * w).sum() / max(w.sum(), 1e-30)
                if args.plane_winsor_q is not None:
                    od_int[plane] = winsorize_plane(od_int[plane], args.plane_winsor_q)
                # Existing datasets store od_int as float32, then od_eff as
                # float64 division of that stored float32 plane by dchi_tot.
                rebuilt = od_int[plane].astype(np.float32).astype(np.float64) / dchi_tot[plane]
                existing_path = args.lensplane_root / f"ph{ph:03d}_lc{lcid}" / f"lensplane_{plane:02d}.npz"
                with np.load(existing_path) as existing:
                    compare_arrays(
                        f"lensplane ph{ph:03d} lc{lcid} plane{plane:02d}",
                        rebuilt,
                        existing["od_eff"],
                        args.validate_rtol,
                        args.validate_atol,
                    )


def validate_patches(args: argparse.Namespace, phases: list[int], lightcones: list[int]) -> None:
    planes = parse_ints(args.validate_planes)
    patches = parse_ints(args.validate_patches)
    if not planes:
        raise ValueError("--validate-planes is required for validate-patches")
    if not patches:
        raise ValueError("--validate-patches is required for validate-patches")
    patches_per_side = args.xsize // args.patch_size

    for ph in phases:
        for lcid in lightcones:
            for plane in planes:
                lensplane_path = args.lensplane_root / f"ph{ph:03d}_lc{lcid}" / f"lensplane_{plane:02d}.npz"
                with np.load(lensplane_path) as lensplane:
                    od_eff = lensplane["od_eff"]
                    for patch_idx in patches:
                        expected = get_patch(od_eff, patch_idx, args.patch_size, patches_per_side)
                        patch_path = args.patch_root / f"{plane:02d}" / f"ph{ph:03d}_lc{lcid}_{patch_idx:03d}.npy"
                        actual = np.load(patch_path)
                        compare_arrays(
                            f"patch ph{ph:03d} lc{lcid} plane{plane:02d} patch{patch_idx:03d}",
                            actual,
                            expected,
                            args.validate_rtol,
                            args.validate_atol,
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="direct-patches",
        help=(
            "Comma-separated stages. 'direct-patches' writes final patches, and metadata unless disabled, from ASDF inputs. "
            "'metadata' writes per-phase lensplane metadata from ASDF headers. "
            "'od,lensplanes,patches' keeps saved OD and full-lensplane intermediates. "
            "Use 'counts,od-from-counts' only if you explicitly want saved count-map intermediates."
        ),
    )
    parser.add_argument("--phases", default="auto", help="Comma/range list, e.g. 0 or 10-24, or 'auto'.")
    parser.add_argument("--lightcones", default="1,2", help="Comma/range list of projected footprints. Existing rotations define 1 and 2.")
    parser.add_argument("--raw-root", type=Path, default=Path("../abacus_lightcones"))
    parser.add_argument("--phase-prefix", default="", help="Use c009_ for ../abacus_lightcones/c009_ph000.")
    parser.add_argument("--counts-root", type=Path, default=Path("../abacus_lightcones_npy"))
    parser.add_argument("--od-root", type=Path, default=Path("../abacus_lightcones_od"))
    parser.add_argument("--lensplane-root", type=Path, default=Path("./lp_dataset_test"))
    parser.add_argument("--patch-root", type=Path, default=Path("./lp_dataset_patched"))
    parser.add_argument("--metadata-root", type=Path, default=Path("./metadata"))
    parser.add_argument("--dataset-name", default=None, help="Dataset name used under --metadata-root. Defaults to --patch-root name.")
    parser.add_argument("--write-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument(
        "--step-source",
        choices=("lightcone0", "any"),
        default="lightcone0",
        help="Which ASDF files define the step grid. 'lightcone0' matches the old scripts; 'any' is more permissive.",
    )
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--xsize", type=int, default=2560)
    parser.add_argument("--reso-arcmin", type=float, default=27.2 * 60.0 / 2560.0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--n-lensplanes", type=int, default=20)
    parser.add_argument("--chi-min", type=float, default=290.0)
    parser.add_argument("--chi-max", type=float, default=3580.0)
    parser.add_argument("--shell-mask", type=Path, default=None)
    parser.add_argument("--per-shell-demean", action="store_true")
    parser.add_argument("--per-plane-demean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shell-clip-sigma", type=none_or_float, default=None)
    parser.add_argument("--shell-hard-clip", type=none_or_float, default=None)
    parser.add_argument("--plane-winsor-q", type=none_or_float, default=None)
    parser.add_argument("--assert-overlaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validate-steps", default="", help="Comma/range list of shell steps for the validate-od stage.")
    parser.add_argument("--validate-planes", default="", help="Comma/range list of lensplane ids for validation stages.")
    parser.add_argument("--validate-patches", default="", help="Comma/range list of patch ids for validate-patches.")
    parser.add_argument("--validate-rtol", type=float, default=1e-5)
    parser.add_argument("--validate-atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_name is None:
        args.dataset_name = args.patch_root.name
    phases = available_phases(args.raw_root, args.phase_prefix) if args.phases == "auto" else parse_ints(args.phases)
    lightcones = parse_ints(args.lightcones)
    bad_lightcones = [lcid for lcid in lightcones if lcid < 1 or lcid > len(DEFAULT_ROTATIONS)]
    if bad_lightcones:
        raise ValueError(f"No rotations are configured for projected lightcones: {bad_lightcones}")
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}

    if "direct-patches" in stages:
        build_patches_direct_from_asdf(args, phases, lightcones)
    if "metadata" in stages:
        build_metadata(args, phases, lightcones)
    if "counts" in stages:
        build_projected_counts(args, phases, lightcones)
    if "od" in stages:
        build_overdensity_shells_direct(args, phases, lightcones)
    if "od-from-counts" in stages:
        build_overdensity_shells(args, phases, lightcones)
    if "lensplanes" in stages:
        build_lensplanes(args, phases, lightcones)
    if "patches" in stages:
        patchify_lensplanes(args, phases, lightcones)
    if "validate-od" in stages:
        validate_direct_od(args, phases, lightcones)
    if "validate-lensplanes" in stages:
        validate_lensplanes(args, phases, lightcones)
    if "validate-patches" in stages:
        validate_patches(args, phases, lightcones)


if __name__ == "__main__":
    main()
