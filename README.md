# Conicus Dataset Processing

Minimal scripts for processing AbacusSummit HEALPix particle lightcone products into patched projected-overdensity lensplane datasets used by [Generative Diffusion Priors for 3D Mapping of the Dark Universe](https://imaging.cms.caltech.edu/gdpdm/), Zhao et al., CVPR, 2026.

The pipeline reads raw ASDF lightcone products and writes final `128 x 128` patch files plus per-phase lensplane metadata. It does not save the older projected-count, overdensity-shell, or full-lensplane intermediates.

## Inputs

This code takes the public AbacusSummit lightcone catalog data products as input. For instructions on how to download these, please see [this webpage](https://abacussummit.readthedocs.io/en/latest/data-access.html).

Expected raw input layout:

```text
abacus_lightcones/
  ph000/heal/LightCone*_heal_Step*.asdf
  ...
  ph024/heal/LightCone*_heal_Step*.asdf
  c001_ph000/heal/LightCone*_heal_Step*.asdf
  ...
```

The script uses ASDF header fields `ScaleFactor`, `CoordinateDistanceHMpc`, `NP`, and `BoxSizeHMpc`, and particle HEALPix ids from `data/heal`.

## Outputs

Patch files:

```text
{patch_root}/{lensplane_id}/ph{phase_id}_lc{lightcone_id}_{patch_id}.npy
```

Metadata files:

```text
{metadata_root}/{dataset_name}/lensplanes_index/ph{phase_id}_lensplanes_index.npz
```

Metadata fields are `edges`, `chi_eff`, `a_eff`, `dchi_tot`, `G`, `ph`, and `lcid`. Distances are in `Mpc/h`.

## Processing

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild c000:

```bash
python build_lensplane_dataset.py \
  --stages direct-patches \
  --phases 0-24 \
  --raw-root /path/to/abacus_lightcones \
  --patch-root ./lp_dataset_patched \
  --metadata-root ./metadata \
  --dataset-name lp_dataset_patched \
  --step-source any \
  --step-min 408 \
  --step-max 1105 \
  --no-assert-overlaps \
  --skip-existing
```

Rebuild c001 five-phase dataset:

```bash
python build_lensplane_dataset.py \
  --stages direct-patches \
  --phases 0-4 \
  --phase-prefix c001_ \
  --raw-root /path/to/abacus_lightcones \
  --patch-root ./lp_dataset_patched_c001 \
  --metadata-root ./metadata \
  --dataset-name lp_dataset_patched_c001 \
  --step-source any \
  --no-assert-overlaps \
  --skip-existing
```

To write metadata only:

```bash
python build_lensplane_dataset.py \
  --stages metadata \
  --phases 0-24 \
  --raw-root /path/to/abacus_lightcones \
  --metadata-root ./metadata \
  --dataset-name lp_dataset_patched \
  --step-source any \
  --step-min 408 \
  --step-max 1105 \
  --no-assert-overlaps \
  --skip-existing
```

For exact c000 reproduction, the raw ASDF inputs should include the same step window used by the original processing path, roughly steps `0408` through `1105`. If the farthest high-redshift steps are absent, the last lensplane metadata and patches will differ.

## Dataset Geometry

- HEALPix resolution: `NSIDE=16384`
- projected map size: `2560 x 2560`
- footprint: `27.2 deg x 27.2 deg`
- pixel scale: `0.6375 arcmin/pixel`
- patch size: `128 x 128`
- patches per full map: `20 x 20`
- lensplanes: 20 uniform comoving-distance bins from `290.0` to `3580.0 Mpc/h`
- lightcone rotations:
  - `lc1`: `rot=[45., 64.9, 45.]`
  - `lc2`: `rot=[71.7, 17.4]`

Stored patch values are `od_eff`, a dimensionless thickness-averaged projected overdensity:

```text
od_int[k](x, y) = sum_i delta_i(x, y) * overlap_dchi_i
od_eff[k](x, y) = od_int[k](x, y) / dchi_tot[k]
```

## Citation

If you use this processing code or the released dataset, please cite:

```bibtex
@InProceedings{Zhao_2026_CVPR,
    author    = {Zhao, Brandon and Scognamiglio, Diana and Dor\'e, Olivier and Bouman, Katherine L.},
    title     = {Generative Diffusion Priors for 3D Mapping of the Dark Universe},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {23581-23590}
}
```

The raw AbacusSummit lightcone products are described by [Hadzhiyska et al. (2022)](https://arxiv.org/abs/2110.11413).
