# FOG_GOD_MR_Project

**Does video defogging make visual SLAM work in fog?** We add physically based synthetic fog to real and simulated video, remove it with a temporal defogging network, and measure whether SLAM trajectories improve. Trajectory accuracy (ATE), not PSNR, is the main measure of success.



## Repository layout

| Folder | What it does |
|---|---|
| `defogging/` | Training launchers, fine-tuning data builder, image-quality and ORB-feature evaluation |
| `slam/droid/` | DROID-SLAM install, per-clip run and scoring, full experiment matrix, summary |
| `results/` | Result tables from the mid-evaluation |

### `defogging/`

| File | Purpose |
|---|---|
| `run_drivedefog_6mor.sh` | Base training on driving fog (MOR 50–300 m, 40 epochs, 4 GPUs) and test evaluation. |
| `run_ft_eow_mf_all.sh` | Inference before fine-tuning, 5-epoch fine-tune on EOW + MF with driving replay, inference after, before/after report. |
| `build_ft_domains.py` | Builds the fine-tuning data view (symlinks): EOW/MF frame ranges with gaps between train/val/test, plus driving replay clips. |
| `ft_lr.py` | Computes the learning rate that gives the intended fine-tuning LR after the scheduler resumes. |
| `compute_defog_class_weights.py` | Class weights for the segmentation loss. |
| `eval_edfog_test.py` | PSNR/SSIM of the defogger on held-out frames; optionally saves the defogged frames that SLAM consumes. |
| `eval_orb_proxy.py` | Fast feature-level check: ORB matches surviving RANSAC between frame pairs (clean, foggy, defogged). Not ORB-SLAM. |
| `foggy_baseline.py` | PSNR/SSIM of the undefogged foggy input on the same frames, as the "no defogging" baseline. |

**Not included:** the fog synthesis code and the foggy datasets are part of a separate private research codebase and are not released here. The DriveDefog network (FogPromptNetV2), its trainer, dataset loader and losses come from a separate private research codebase. The launchers here show the exact training and fine-tuning configuration. `eval_edfog_test.py`, `eval_orb_proxy.py` and `ft_lr.py` import from that codebase and run only inside it.

### `slam/droid/`

| File | Purpose |
|---|---|
| `install_droid.sh`, `install_droid2.sh` | Build DROID-SLAM (CUDA extensions) in a virtualenv and fetch `droid.pth`. The second script finishes the build without pip build isolation. |
| `droid_clip.py` | Runs monocular DROID-SLAM on one image folder and scores it with `evo` (APE, translation, Sim(3) alignment with scale correction). MF fisheye frames are rectified to a 640×480 pinhole view first. Writes `metrics.json` and the TUM trajectory. |
| `run_droid_all.py` | Full matrix: 2 sequences × {clean, foggy, defogged-base, defogged-fine-tuned} × 4 fog levels × 3 runs = 78 runs, one worker per GPU. Finished runs are skipped. |
| `summarize_droid.py` | Collects all runs into `droid_runs.csv` and `droid_summary.csv` (median ATE over runs). |

### `results/`

| File | Contents |
|---|---|
| `droid_summary.csv` | DROID-SLAM ATE RMSE per sequence, condition and fog level (median, min, max of 3 runs). |
| `droid_runs.csv` | Every individual DROID-SLAM run. |
| `before_after_report.txt` | Defogger PSNR/SSIM on EOW and MF, before vs after fine-tuning. |
| `foggy_baseline.csv` | PSNR/SSIM of foggy, base-defogged and fine-tuned-defogged frames against clean. |

## Main result so far (DROID-SLAM, ATE RMSE in m, median of 3 runs)

| MOR | EOW foggy | EOW base | EOW fine-tuned | MF foggy | MF base | MF fine-tuned |
|---|---:|---:|---:|---:|---:|---:|
| 50 m | 0.056 | 0.043 | 0.045 | 3.98 | 6.75 | 5.15 |
| 75 m | 0.055 | 0.053 | 0.041 | 6.02 | 6.81 | 6.63 |
| 100 m | 0.047 | 0.057 | 0.045 | 4.97 | 4.23 | 6.49 |
| 150 m | 0.047 | 0.043 | 0.043 | 4.52 | 4.59 | 6.60 |
| Clean | 0.043 | | | 3.76 | | |

Fine-tuning raises MF PSNR well above the foggy input, yet MF trajectories get worse at every fog level. On EOW the differences are millimetres. In 12 of 13 MF conditions the three runs gave identical trajectories, so these numbers have no run-to-run spread yet. ORB-SLAM3 and Rover-SLAM have not been run on this fog data.

## External dependencies

| Dependency | Version used | Role |
|---|---|---|
| [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | commit `2dfd39f`, unmodified | Learned dense SLAM |
| [evo](https://github.com/MichaelGrupp/evo) | pip | Trajectory evaluation |

Data: TartanAir Endofworld and SubT-MRS Multi_Floor, both with ground-truth poses.

## Running

The scripts were written for a SLURM cluster with 4 GPUs per node. Paths such as `/scratch/Ananya_Kulkarni/...` and the conda env `visia` are hard-coded at the top of each script and need changing elsewhere.

```bash
# 1. Defogging (needs the private DriveDefog codebase)
sbatch defogging/run_drivedefog_6mor.sh                     # base model
sbatch defogging/run_ft_eow_mf_all.sh                       # fine-tune + defogged EOW/MF frames

# 2. SLAM
bash   slam/droid/install_droid.sh                          # then install_droid2.sh if lietorch fails
python slam/droid/run_droid_all.py                          # 78 runs, about 40 min on 4 GPUs
python slam/droid/summarize_droid.py                        # -> droid_runs.csv, droid_summary.csv
```

Single clip example:

```bash
python slam/droid/droid_clip.py --images <frame folder> --domain multifloor --out <out dir> \
    [--frames-from <folder whose frame numbers to use>]
```
