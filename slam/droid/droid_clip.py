#!/usr/bin/env python
"""Run DROID-SLAM (monocular) on one clip and score it the way the repo's test_euroc.py does:
evo APE, translation part, Sim(3) alignment with scale correction. Frame numbers in the file names
pick the timestamp and GT pose, so the uneven MF frame gaps are handled."""
import argparse, csv, glob, json, os, sys, time
REPO = '/scratch/Ananya_Kulkarni/slam/DROID-SLAM'
sys.path.insert(0, f'{REPO}/droid_slam')
import cv2, numpy as np, torch
from droid import Droid
from scipy.spatial.transform import Rotation as Rot
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape

R = '/scratch/Ananya_Kulkarni'
T_B_C_MF = np.array([[0.00373945, 0.00244390, 0.99999002, 0.17409098],
                     [-0.99989234, -0.01418010, 0.00377374, 0.04962189],
                     [0.01418918, -0.99989647, 0.00239061, -0.02152957],
                     [0, 0, 0, 1.0]])
MEI = dict(xi=1.5967783801524584, k1=-0.04701957438555236, k2=0.28058825271927323,
           p1=0.0004063627333225579, p2=-0.00015239154341021015,
           g1=771.6611713881042, g2=688.7864250675316, u0=317.73333375994946, v0=233.53737010585292)
PIN_MF = (280.0, 280.0, 320.0, 240.0)   # pinhole view MF frames are rectified to, 640x480


def mei_maps(w=640, h=480):
    """Remap from the MEI fisheye model in Multi_Floor/calibration/intrinsics.yaml to pinhole PIN_MF."""
    fx, fy, cx, cy = PIN_MF; m = MEI
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x, y, z = (u - cx) / fx, (v - cy) / fy, np.ones_like(u)
    n = np.sqrt(x * x + y * y + z * z); x, y, z = x / n, y / n, z / n
    mx, my = x / (z + m['xi']), y / (z + m['xi'])
    r2 = mx * mx + my * my; rad = 1 + m['k1'] * r2 + m['k2'] * r2 * r2
    dx = mx * rad + 2 * m['p1'] * mx * my + m['p2'] * (r2 + 2 * mx * mx)
    dy = my * rad + m['p1'] * (r2 + 2 * my * my) + 2 * m['p2'] * mx * my
    return (m['g1'] * dx + m['u0']).astype(np.float32), (m['g2'] * dy + m['v0']).astype(np.float32)


def load_gt(domain):
    """Return (image_times_by_frame_number, gt_times, gt_xyz, gt_quat_wxyz); frame 000001 is line 1."""
    if domain == 'endofworld':
        ts = np.loadtxt(f'{R}/Endofworld/data/visual_endofworld/image_lcam_front/timestamps.txt')
        g = np.loadtxt(f'{R}/Endofworld/ground_truth/trajectory_camera.txt')
        return ts, g[:, 0], g[:, 1:4], g[:, [7, 4, 5, 6]]
    ts = np.loadtxt(f'{R}/Multi_Floor/data/Multi_Floor_Folder/cam_0/timestamps.txt', dtype=np.int64)
    rows = list(csv.DictReader(open(f'{R}/Multi_Floor/ground_truth/trajectory_imu.csv')))
    t0 = int(ts[0])
    tg = np.array([(int(r['timestamp']) - t0) * 1e-9 for r in rows])
    pos = np.array([[float(r[k]) for k in ('p_w_b_x', 'p_w_b_y', 'p_w_b_z')] for r in rows])
    q = np.array([[float(r[k]) for k in ('q_w_b_x', 'q_w_b_y', 'q_w_b_z', 'q_w_b_w')] for r in rows])
    Rwb = Rot.from_quat(q)                                    # IMU pose -> camera pose
    pc = pos + Rwb.apply(T_B_C_MF[:3, 3])
    qc = (Rwb * Rot.from_matrix(T_B_C_MF[:3, :3])).as_quat()
    return (ts - t0) * 1e-9, tg, pc, qc[:, [3, 0, 1, 2]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True); p.add_argument('--domain', required=True, choices=['endofworld', 'multifloor'])
    p.add_argument('--out', required=True); p.add_argument('--frames-from'); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--weights', default=f'{REPO}/droid.pth'); p.add_argument('--buffer', type=int, default=768)
    p.add_argument('--image_size', default=[240, 320]); p.add_argument('--disable_vis', action='store_true', default=True)
    p.add_argument('--stereo', action='store_true'); p.add_argument('--upsample', action='store_true')
    p.add_argument('--beta', type=float, default=0.3); p.add_argument('--filter_thresh', type=float, default=2.4)
    p.add_argument('--warmup', type=int, default=8); p.add_argument('--keyframe_thresh', type=float, default=4.0)
    p.add_argument('--frontend_thresh', type=float, default=16.0); p.add_argument('--frontend_window', type=int, default=25)
    p.add_argument('--frontend_radius', type=int, default=2); p.add_argument('--frontend_nms', type=int, default=1)
    p.add_argument('--backend_thresh', type=float, default=22.0); p.add_argument('--backend_radius', type=int, default=2)
    p.add_argument('--backend_nms', type=int, default=3); p.add_argument('--asynchronous', action='store_true')
    p.add_argument('--frontend_device', default='cuda'); p.add_argument('--backend_device', default='cuda')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn', force=True)

    src = a.frames_from or a.images                 # same frame numbers for every condition
    names = sorted(os.path.basename(f) for f in glob.glob(f'{src}/*.png'))
    if a.limit: names = names[:a.limit]
    nums = [int(n[:-4]) for n in names]
    if a.domain == 'endofworld': K0 = (320., 320., 320., 320.); maps = None
    else: K0 = PIN_MF; maps = mei_maps()

    def stream():
        for num, n in zip(nums, names):
            im = cv2.imread(f'{a.images}/{n}')
            if maps is not None: im = cv2.remap(im, maps[0], maps[1], cv2.INTER_LINEAR)
            h0, w0 = im.shape[:2]; s = np.sqrt((384 * 512) / (h0 * w0))
            h1, w1 = int(h0 * s), int(w0 * s)
            im = cv2.resize(im, (w1, h1))[:h1 - h1 % 8, :w1 - w1 % 8]
            K = torch.as_tensor(K0, dtype=torch.float32).clone()
            K[0::2] *= w1 / w0; K[1::2] *= h1 / h0
            yield num, torch.as_tensor(im).permute(2, 0, 1)[None], K

    t_start = time.time(); droid = None
    for t, im, K in stream():
        if droid is None:
            a.image_size = [im.shape[2], im.shape[3]]; droid = Droid(a)
        droid.track(t, im, intrinsics=K)
    n_kf = int(droid.video.counter.value); t_track = time.time() - t_start
    traj = droid.terminate(stream()); t_total = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 2**30

    img_t, gt_t, gt_xyz, gt_q = load_gt(a.domain)
    t_est = np.array([img_t[n - 1] for n in nums], dtype=np.float64)
    np.savetxt(f'{a.out}/traj_est_tum.txt', np.c_[t_est, traj[:, :3], traj[:, 3:7]], fmt='%.6f')
    est = PoseTrajectory3D(positions_xyz=traj[:, :3], orientations_quat_wxyz=traj[:, [6, 3, 4, 5]], timestamps=t_est)
    ref = PoseTrajectory3D(positions_xyz=gt_xyz, orientations_quat_wxyz=gt_q, timestamps=gt_t)
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.1 if a.domain == 'multifloor' else 0.01)
    res = main_ape.ape(ref_s, est_s, pose_relation=PoseRelation.translation_part,
                       align=True, correct_scale=True)
    st = res.stats
    scale = float(res.np_arrays.get('alignment_scale_correction', [np.nan])[0]) if 'alignment_scale_correction' in res.np_arrays else float(res.info.get('alignment_transformation_scale', np.nan)) if hasattr(res, 'info') else np.nan
    seg = np.linalg.norm(np.diff(ref_s.positions_xyz, axis=0), axis=1).sum()
    out = dict(images=a.images, domain=a.domain, frames=len(nums), keyframes=n_kf, gt_matched=int(ref_s.num_poses),
               gt_path_m=round(float(seg), 3), ate_rmse_m=float(st['rmse']), ate_mean_m=float(st['mean']),
               ate_median_m=float(st['median']), ate_std_m=float(st['std']), ate_min_m=float(st['min']), ate_max_m=float(st['max']),
               sim3_scale=scale, track_s=round(t_track, 1), total_s=round(t_total, 1),
               fps=round(len(nums) / t_total, 2), peak_gpu_gb=round(peak, 2))
    json.dump(out, open(f'{a.out}/metrics.json', 'w'), indent=1); print('METRICS ' + json.dumps(out), flush=True)
    os._exit(0)                                     # DROID leaves worker state behind; metrics are saved


if __name__ == '__main__':
    main()
