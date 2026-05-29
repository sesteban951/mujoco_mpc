#include "mjpc/tasks/g1/tracking/tracking.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace {

// Linear interpolation between two adjacent mocap keyframes.
// Returns (idx_0, idx_1, w_0, w_1) such that interp = w_0 * frame[idx_0]
//                                                   + w_1 * frame[idx_1].
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// Sample rate of every loaded clip. Must match all hz.csv values used when
// generating the keyframe XMLs via csv_to_keyframe.py (--fps).
constexpr double kFps = 50.0;

// Per-motion frame counts. The total across all entries must equal the number
// of <key> elements MuJoCo loads (i.e. the sum of all included keyframe files).
// Add a new entry whenever you include another keyframe file in task.xml, and
// update the "task_transition" dropdown there to add a corresponding name.
constexpr int kMotionLengths[] = {
    264,  // Jump Up  (g1/tracking/keyframes/srb_ik_jump_23dof_poses.xml)
    253,  // Jump Fwd (g1/tracking/keyframes/srb_ik_jump_fwd_23dof_poses.xml)
};

int MotionLength(int id) { return kMotionLengths[id]; }

int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) start += MotionLength(i);
  return start;
}

// Tracked-body names. THIS ORDER IS LOAD-BEARING:
//   1. It must match the declaration order of the mocap[<name>] bodies in
//      task.xml (which determines body_mocapid and the layout of key_mpos).
//   2. It must match the dims of the BodyPos/BodyOri/BodyLinVel/BodyAngVel
//      <user> residual sensors in task.xml (each is dim = 14 * 3 = 42).
const std::array<std::string, 14> body_names = {
    "pelvis",    "torso",     "lknee",  "rknee",
    "lhand",     "rhand",     "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",   "rhip",
    "lfoot",     "rfoot",
};

// Anchor body for the BeyondMimic relative formulation: per-body pose is
// tracked in a frame attached to this body, yaw-aligned to the reference.
// Must be one of body_names. (whole_body_tracking / unitree_rl_mjlab use
// "torso_link"; here that maps to the "torso" entry.)
constexpr const char* kAnchorBody = "torso";

// Quaternion (w,x,y,z) keeping only the yaw (rotation about world z).
void YawQuat(double res[4], const double q[4]) {
  double yaw = std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
  res[0] = std::cos(0.5 * yaw);
  res[1] = 0.0;
  res[2] = 0.0;
  res[3] = std::sin(0.5 * yaw);
}

}  // namespace

namespace mjpc::g1 {

std::string Tracking::XmlPath() const {
  return GetModelPath("g1/tracking/task.xml");
}
std::string Tracking::Name() const { return "G1 Track"; }

// ----------------- Residuals for G1 tracking task -----------------
// BeyondMimic-style motion tracking (whole_body_tracking / unitree_rl_mjlab).
// All terms are normed with the Gaussian loss kGaussianLoss (1 - exp(-x^2/p^2))
// so each maps to the exp(-err^2/std^2) tracking reward.
//   Residual (0): AnchorPos   (3)  torso global position error
//   Residual (1): AnchorOri   (3)  torso global orientation error
//   Residual (2): BodyPos     (42) 14 bodies, position in the yaw-aligned
//                                  anchor frame (relative pose-shape)
//   Residual (3): BodyOri     (42) 14 bodies, orientation in the anchor frame
//   Residual (4): BodyLinVel  (42) 14 bodies, GLOBAL linear velocity error
//   Residual (5): BodyAngVel  (42) 14 bodies, GLOBAL angular velocity error
//   Number of parameters: 0   (per-term std lives in the <user> norm params)
// -----------------------------------------------------------------
void Tracking::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // Map sim time -> fractional frame index within the active clip.
  int start = MotionStartIndex(current_mode_);
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- per-body lookups against the interpolated reference clip ----- //
  auto mocapid = [&](const std::string& body_name) {
    int id = mj_name2id(model, mjOBJ_BODY, ("mocap[" + body_name + "]").c_str());
    assert(0 <= id);
    int mid = model->body_mocapid[id];
    assert(0 <= mid);
    return mid;
  };
  // Interpolated reference position of a mocap body (world frame).
  auto ref_pos = [&](int mid, double out[3]) {
    mju_scl3(out, model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * mid,
             weight_0);
    mju_addToScl3(out,
                  model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * mid,
                  weight_1);
  };
  // Interpolated reference orientation of a mocap body (world frame).
  auto ref_quat = [&](int mid, double out[4]) {
    mju_scl(out, model->key_mquat + model->nmocap * 4 * key_index_0 + 4 * mid,
            weight_0, 4);
    mju_addToScl(out, model->key_mquat + model->nmocap * 4 * key_index_1 + 4 * mid,
                 weight_1, 4);
    mju_normalize4(out);
  };
  auto robot_pos = [&](const std::string& body_name) {
    return SensorByName(model, data, ("tracking_pos[" + body_name + "]").c_str());
  };
  auto robot_quat = [&](const std::string& body_name) {
    return SensorByName(model, data,
                        ("tracking_quat[" + body_name + "]").c_str());
  };

  // ----- anchor (torso) world pose, reference + robot ----- //
  int anchor_mid = mocapid(kAnchorBody);
  double ref_anchor_pos[3], ref_anchor_quat[4];
  ref_pos(anchor_mid, ref_anchor_pos);
  ref_quat(anchor_mid, ref_anchor_quat);
  double* robot_anchor_pos = robot_pos(kAnchorBody);
  double* robot_anchor_quat = robot_quat(kAnchorBody);

  // delta orientation: yaw-only misalignment between robot and reference anchor
  //   delta_ori = yaw( robot_anchor_quat * inv(ref_anchor_quat) )
  double inv_ref_anchor_quat[4], tmp_quat[4], delta_ori[4];
  mju_negQuat(inv_ref_anchor_quat, ref_anchor_quat);
  mju_mulQuat(tmp_quat, robot_anchor_quat, inv_ref_anchor_quat);
  YawQuat(delta_ori, tmp_quat);

  // delta position: robot anchor xy, reference anchor z (height comes from clip)
  double delta_pos[3] = {robot_anchor_pos[0], robot_anchor_pos[1],
                         ref_anchor_pos[2]};

  int counter = 0;

  // ----- (0) AnchorPos: torso global position error ----- //
  mju_sub3(&residual[counter], robot_anchor_pos, ref_anchor_pos);
  counter += 3;

  // ----- (1) AnchorOri: torso global orientation error ----- //
  mju_subQuat(&residual[counter], robot_anchor_quat, ref_anchor_quat);
  counter += 3;

  // ----- (2) BodyPos: per-body position in the yaw-aligned anchor frame ----- //
  // target_pos = delta_pos + delta_ori * (ref_body_pos - ref_anchor_pos)
  for (const auto& body_name : body_names) {
    double rbp[3];
    ref_pos(mocapid(body_name), rbp);
    mju_subFrom3(rbp, ref_anchor_pos);           // ref body relative to anchor
    double target[3];
    mju_rotVecQuat(target, rbp, delta_ori);      // into robot's yaw frame
    mju_addTo3(target, delta_pos);               // re-anchor at robot xy / ref z
    mju_sub3(&residual[counter], robot_pos(body_name), target);
    counter += 3;
  }

  // ----- (3) BodyOri: per-body orientation in the anchor frame ----- //
  // target_quat = delta_ori * ref_body_quat
  for (const auto& body_name : body_names) {
    double rbq[4], target_quat[4];
    ref_quat(mocapid(body_name), rbq);
    mju_mulQuat(target_quat, delta_ori, rbq);
    mju_subQuat(&residual[counter], robot_quat(body_name), target_quat);
    counter += 3;
  }

  // ----- (4) BodyLinVel: per-body GLOBAL linear velocity error ----- //
  // Reference velocity = finite-difference of mocap position between frames.
  for (const auto& body_name : body_names) {
    int mid = mocapid(body_name);
    mju_copy3(&residual[counter],
              model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * mid);
    mju_subFrom3(&residual[counter],
                 model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * mid);
    mju_scl3(&residual[counter], &residual[counter], kFps);
    double* sensor_linvel =
        SensorByName(model, data, ("tracking_linvel[" + body_name + "]").c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);
    counter += 3;
  }

  // ----- (5) BodyAngVel: per-body GLOBAL angular velocity error ----- //
  // Reference angvel from finite-differenced mocap quats (q0 body frame),
  // rotated into the world frame to match the frameangvel sensor.
  for (const auto& body_name : body_names) {
    int mid = mocapid(body_name);
    const mjtNum* q0 =
        model->key_mquat + model->nmocap * 4 * key_index_0 + 4 * mid;
    const mjtNum* q1 =
        model->key_mquat + model->nmocap * 4 * key_index_1 + 4 * mid;
    double dvel[3];
    mju_subQuat(dvel, q1, q0);
    mju_scl3(dvel, dvel, kFps);
    double ref_angvel[3];
    mju_rotVecQuat(ref_angvel, dvel, q0);
    double* sensor_angvel =
        SensorByName(model, data, ("tracking_angvel[" + body_name + "]").c_str());
    mju_sub3(&residual[counter], ref_angvel, sensor_angvel);
    counter += 3;
  }

  CheckSensorDim(model, counter);
}

// -------------- Transition for G1 tracking task -----------------
//   Called every physics step by MJPC.
//   - On mode change (or t=0): teleport robot to the active clip's first
//     keyframe qpos/qvel and reset the clip-local reference time.
//   - Every step: write interpolated marker positions into data->mocap_pos
//     so the ghost markers visualize the moving target.
// ---------------------------------------------------------------
void Tracking::TransitionLocked(mjModel* model, mjData* d) {
  int start = MotionStartIndex(mode);
  int length = MotionLength(mode);

  if (residual_.current_mode_ != mode || d->time == 0.0) {
    residual_.current_mode_ = mode;
    residual_.reference_time_ = d->time;

    // Teleport robot to the clip's starting state.
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  double current_index = (d->time - residual_.reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum* mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum* mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);
  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  // Same blend for orientation; LERP + renormalize per mocap body. Adjacent
  // frames at kFps are close enough that this matches slerp to high accuracy.
  mjtNum* mocap_quat_0 = mj_stackAllocNum(d, 4 * model->nmocap);
  mjtNum* mocap_quat_1 = mj_stackAllocNum(d, 4 * model->nmocap);
  mju_scl(mocap_quat_0, model->key_mquat + model->nmocap * 4 * key_index_0,
          weight_0, model->nmocap * 4);
  mju_scl(mocap_quat_1, model->key_mquat + model->nmocap * 4 * key_index_1,
          weight_1, model->nmocap * 4);
  mju_copy(d->mocap_quat, mocap_quat_0, model->nmocap * 4);
  mju_addTo(d->mocap_quat, mocap_quat_1, model->nmocap * 4);
  for (int i = 0; i < model->nmocap; i++) {
    mju_normalize4(d->mocap_quat + 4 * i);
  }

  mj_freeStack(d);
}

}  // namespace mjpc::g1
