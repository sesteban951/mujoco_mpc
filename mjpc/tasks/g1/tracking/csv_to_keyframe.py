"""Convert a G1 23-DOF qpos-trajectory CSV into a tracking-task keyframe XML.

Input CSV: N rows x 30 cols = [base_x, base_y, base_z, qx, qy, qz, qw, 23 joints]
in the same joint order as g1_23dof_hydrax.xml. (Quaternion is (qx,qy,qz,qw);
this script reorders to MuJoCo's (qw,qx,qy,qz).)

Output: <keyframe> block with one <key> per frame. Every key carries `mpos`
(14 markers x 3 = 42 floats) and `mquat` (14 quats x 4 = 56 floats), computed
via forward kinematics on g1_23dof_hydrax.xml. The FIRST key additionally
carries `qpos` (30) and `qvel` (29) so the robot teleports to the clip start
when the task loads.

Usage (run from mjpc/tasks/g1/tracking/):
  python csv_to_keyframe.py csv/srb_ik_jump_23dof.csv --fps 50
  -> writes keyframes/srb_ik_jump_23dof_poses.xml (override with --out / --name).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import mujoco

# Order MUST match body_names[] in tracking.cc and the mocap[<name>] body
# declarations in tracking/task.xml (which fix body_mocapid -> key_mpos layout).
TRACKED_BODIES = [
    ("pelvis",    "pelvis"),
    ("torso",     "torso_link"),
    ("lknee",     "left_knee_link"),
    ("rknee",     "right_knee_link"),
    ("lhand",     "left_wrist_yaw_link"),
    ("rhand",     "right_wrist_yaw_link"),
    ("lelbow",    "left_elbow_link"),
    ("relbow",    "right_elbow_link"),
    ("lshoulder", "left_shoulder_pitch_link"),
    ("rshoulder", "right_shoulder_pitch_link"),
    ("lhip",      "left_hip_pitch_link"),
    ("rhip",      "right_hip_pitch_link"),
    ("lfoot",     "left_ankle_roll_link"),
    ("rfoot",     "right_ankle_roll_link"),
]

def load_qpos_23dof(csv_path: Path) -> np.ndarray:
    """Load 23-DOF CSV (N x 30) and return qpos in MuJoCo quat order (w,x,y,z)."""
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 30:
        raise ValueError(
            f"Expected (N, 30) 23-DOF CSV; got shape {data.shape}. "
            "Layout must be [base_xyz(3), base_quat_xyzw(4), joints(23)]."
        )

    # CSV base quat is (qx, qy, qz, qw); MuJoCo expects (qw, qx, qy, qz).
    qpos = data.copy()
    qx, qy, qz, qw = qpos[:, 3].copy(), qpos[:, 4].copy(), qpos[:, 5].copy(), qpos[:, 6].copy()
    qpos[:, 3] = qw
    qpos[:, 4] = qx
    qpos[:, 5] = qy
    qpos[:, 6] = qz
    return qpos


def compute_mpos_mquat(model, data, qpos_traj: np.ndarray,
                       body_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Run FK on each frame; return (mpos (N, 14*3), mquat (N, 14*4))."""
    N = qpos_traj.shape[0]
    nb = len(body_ids)
    mpos = np.zeros((N, nb * 3))
    mquat = np.zeros((N, nb * 4))
    for i in range(N):
        data.qpos[:] = qpos_traj[i]
        mujoco.mj_kinematics(model, data)
        for k, bid in enumerate(body_ids):
            mpos[i, 3 * k:3 * k + 3] = data.xpos[bid]
            mquat[i, 4 * k:4 * k + 4] = data.xquat[bid]
    return mpos, mquat


def compute_initial_qvel(model, qpos_traj: np.ndarray, dt: float) -> np.ndarray:
    """qvel[0] via finite difference between frame 0 and frame 1 (proper quat diff)."""
    qvel = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qvel, dt, qpos_traj[0], qpos_traj[1])
    return qvel


def fmt_row(arr: np.ndarray, per_line: int = 3, indent: str = "               ") -> str:
    """Format a 1-D float array as space-separated chunks, one chunk per line."""
    vals = [f"{x:.6g}" for x in arr]
    lines = [" ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return ("\n" + indent).join(lines)


def emit_keyframe_xml(
    qpos0: np.ndarray, qvel0: np.ndarray,
    mpos: np.ndarray, mquat: np.ndarray, name: str,
) -> str:
    out = ["<mujoco>", "  <keyframe>"]
    for i in range(mpos.shape[0]):
        key_name = f"{name}_{i}"
        if i == 0:
            out.append(f'    <key name="{key_name}"')
            out.append(f'         qpos="{fmt_row(qpos0, per_line=7)}"')
            out.append(f'         qvel="{fmt_row(qvel0, per_line=6)}"')
            out.append(f'         mpos="{fmt_row(mpos[i], per_line=3)}"')
            out.append(f'         mquat="{fmt_row(mquat[i], per_line=4)}"/>')
        else:
            out.append(f'    <key name="{key_name}"')
            out.append(f'         mpos="{fmt_row(mpos[i], per_line=3)}"')
            out.append(f'         mquat="{fmt_row(mquat[i], per_line=4)}"/>')
    out += ["  </keyframe>", "</mujoco>", ""]
    return "\n".join(out)


def main():
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Path to qpos CSV (30 or 36 cols).")
    p.add_argument("--fps", type=float, required=True,
                   help="Sample rate of the CSV in Hz (also set kFps in tracking.cc).")
    p.add_argument("--model", type=Path,
                   default=script_dir.parent / "g1_23dof_hydrax.xml",
                   help="G1 model XML used for forward kinematics.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output keyframe XML path. "
                        "Defaults to <script_dir>/keyframes/<csv_stem>_poses.xml.")
    p.add_argument("--name", type=str, default=None,
                   help="Clip name used as <key> name prefix; defaults to CSV stem.")
    args = p.parse_args()

    clip_name = args.name or args.csv.stem
    out_path = args.out or (script_dir / "keyframes" / f"{args.csv.stem}_poses.xml")

    qpos_traj = load_qpos_23dof(args.csv)
    N = qpos_traj.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 frames to compute initial qvel.")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nq != 30 or model.nv != 29:
        raise ValueError(
            f"Model shape mismatch: nq={model.nq}, nv={model.nv} (expected 30/29)."
        )
    data = mujoco.MjData(model)

    body_ids = []
    for _, mj_name in TRACKED_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_name)
        if bid < 0:
            raise ValueError(f"Body '{mj_name}' not found in {args.model}")
        body_ids.append(bid)

    dt = 1.0 / args.fps
    mpos, mquat = compute_mpos_mquat(model, data, qpos_traj, body_ids)
    qvel0 = compute_initial_qvel(model, qpos_traj, dt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        emit_keyframe_xml(qpos_traj[0], qvel0, mpos, mquat, clip_name))

    print(f"Loaded {N} frames from {args.csv} (fps={args.fps}, dt={dt:.4f}s)")
    print(f"Wrote {out_path}")
    print()
    print("Next steps:")
    print(f"  1. In task.xml: <include file=\"./keyframes/{out_path.name}\"/>")
    print(f"     (replace or add to placeholder include)")
    print(f"  2. In tracking.cc: set kFps = {args.fps}")
    print(f"     and append {N} to kMotionLengths[]")
    print(f"     and add \"{clip_name}\" to the task_transition text in task.xml")


if __name__ == "__main__":
    main()
