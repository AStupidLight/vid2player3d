import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw AMASS npz files into the joblib format used by convert_amass_isaac.py."
    )
    parser.add_argument(
        "--amass_root",
        type=str,
        required=True,
        help="Root directory of the raw AMASS dataset.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/amass/amass_from_raw.pkl",
        help="Output joblib file.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="Target frame rate for the exported motions.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        help="Skip sequences shorter than this after resampling.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many npz files to process, useful for smoke tests.",
    )
    return parser.parse_args()


def _normalize_gender(gender):
    if isinstance(gender, np.ndarray):
        if gender.shape == ():
            gender = gender.item()
        elif gender.size == 1:
            gender = gender.reshape(-1)[0]
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    gender = str(gender).strip().lower()
    if gender in {"male", "female", "neutral"}:
        return gender
    return "neutral"


def _extract_fps(data):
    for key in ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate"):
        if key in data:
            value = data[key]
            if isinstance(value, np.ndarray):
                value = value.item()
            return float(value)
    raise KeyError("Cannot find frame rate key in AMASS npz.")


def _resample_translation(trans, src_times, dst_times):
    return np.stack(
        [np.interp(dst_times, src_times, trans[:, axis]) for axis in range(trans.shape[1])],
        axis=1,
    ).astype(np.float32)


def _resample_pose_aa(pose_aa, src_times, dst_times):
    if len(src_times) == len(dst_times) and np.allclose(src_times, dst_times):
        return pose_aa.astype(np.float32)

    num_frames, pose_dim = pose_aa.shape
    if num_frames < 2:
        return pose_aa.astype(np.float32)

    num_joints = pose_dim // 3
    pose_rotvec = pose_aa.reshape(num_frames, num_joints, 3)
    out = np.empty((len(dst_times), num_joints, 3), dtype=np.float32)

    for joint_idx in range(num_joints):
        joint_rots = Rotation.from_rotvec(pose_rotvec[:, joint_idx, :])
        slerp = Slerp(src_times, joint_rots)
        out[:, joint_idx, :] = slerp(dst_times).as_rotvec().astype(np.float32)

    return out.reshape(len(dst_times), pose_dim)


def _build_dst_times(num_frames, src_fps, target_fps):
    if num_frames < 2:
        return np.array([0.0], dtype=np.float64)
    duration = (num_frames - 1) / src_fps
    num_dst_frames = int(np.floor(duration * target_fps + 1e-8)) + 1
    return np.arange(num_dst_frames, dtype=np.float64) / target_fps


def _convert_one(npz_path, amass_root, target_fps, min_frames):
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = {"poses", "trans", "betas"}
        if not required_keys.issubset(data.files):
            return None, "missing_keys"

        pose_aa = data["poses"]
        trans = data["trans"]
        betas = data["betas"]

        if pose_aa.ndim != 2 or trans.ndim != 2 or pose_aa.shape[0] != trans.shape[0]:
            return None, "bad_shape"
        if pose_aa.shape[0] < 2:
            return None, "too_short"

        fps = _extract_fps(data)
        if fps <= 0:
            return None, "bad_fps"

        src_times = np.arange(pose_aa.shape[0], dtype=np.float64) / fps
        dst_times = _build_dst_times(pose_aa.shape[0], fps, target_fps)

        pose_aa = _resample_pose_aa(pose_aa, src_times, dst_times)
        trans = _resample_translation(trans, src_times, dst_times)

        if pose_aa.shape[0] < min_frames:
            return None, "too_short_after_resample"

        rel_path = npz_path.relative_to(amass_root).with_suffix("")
        key_name = rel_path.as_posix()

        entry = {
            "pose_aa": pose_aa.astype(np.float32),
            "trans": trans.astype(np.float32),
            "beta": np.asarray(betas, dtype=np.float32),
            "gender": _normalize_gender(data["gender"]) if "gender" in data else "neutral",
            "fps": float(target_fps),
        }
        return key_name, entry


def main():
    args = parse_args()
    amass_root = Path(args.amass_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(amass_root.rglob("*.npz"))
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    converted = {}
    skipped = {}

    for npz_path in tqdm(npz_files, desc="Preparing raw AMASS"):
        key_name, entry_or_reason = _convert_one(npz_path, amass_root, args.target_fps, args.min_frames)
        if key_name is None:
            skipped[entry_or_reason] = skipped.get(entry_or_reason, 0) + 1
            continue
        converted[key_name] = entry_or_reason

    joblib.dump(converted, out_path)

    print(f"Saved {len(converted)} sequences to {out_path}")
    if skipped:
        print("Skipped files:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
