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

// TODO: set to match the FPS of the mocap clips loaded below.
constexpr double kFps = 30.0;

// Per-motion frame counts. The total across all entries must equal the number
// of <key> elements MuJoCo loads (i.e. the sum of all included keyframe files).
// Add a new entry whenever you include another keyframe file in task.xml, and
// update the "task_transition" dropdown there to add a corresponding name.
constexpr int kMotionLengths[] = {
    2,  // Placeholder (g1/tracking/keyframes/placeholder_poses.xml)
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
//   2. It must match how the per-body <user> residual sensors are grouped
//      in task.xml (Pos[knee] dim=6 covers lknee+rknee back-to-back, etc.).
const std::array<std::string, 14> body_names = {
    "pelvis",    "torso",     "lknee",  "rknee",
    "lhand",     "rhand",     "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",   "rhip",
    "lfoot",     "rfoot",
};

}  // namespace

namespace mjpc::g1 {

std::string Tracking::XmlPath() const {
  return GetModelPath("g1/tracking/task.xml");
}
std::string Tracking::Name() const { return "G1 Track"; }

// ----------------- Residuals for G1 tracking task -----------------
//   Number of residuals:
//     Residual (0): joint velocity (nv - 6)
//     Residual (1): control (nu)
//     Residual (2): Pos[avg]   - mean(robot_sites) - mean(mocap_markers)
//     Residual (3..16): per-body position (each 3D), robot relative to
//                       its mean, minus mocap relative to its mean.
//     Residual (17..30): per-body linear velocity (each 3D), mocap
//                        finite-difference minus robot framelinvel.
//   Number of parameters: 0
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

  int counter = 0;

  // ----- joint velocity (skip floating-base 6 DoFs) ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- per-body mocap position (linearly interpolated) ----- //
  auto get_body_mpos = [&](const std::string& body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    mju_scl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);
    mju_addToScl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  // ----- per-body robot position (from framepos sensor on the body) ----- //
  auto get_body_sensor_pos = [&](const std::string& body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double* sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  // Compute centroid of mocap markers and centroid of robot tracking sites.
  // We track the average first, then per-body deviations from it. This lets
  // the user weight global translation tracking independently from per-limb
  // tracking (Pos[avg] vs Pos[*] in the user sensors).
  double avg_mpos[3] = {0};
  double avg_sensor_pos[3] = {0};
  int num_body = 0;
  for (const auto& body_name : body_names) {
    double body_mpos[3];
    double body_sensor_pos[3];
    get_body_mpos(body_name, body_mpos);
    mju_addTo3(avg_mpos, body_mpos);
    get_body_sensor_pos(body_name, body_sensor_pos);
    mju_addTo3(avg_sensor_pos, body_sensor_pos);
    num_body++;
  }
  mju_scl3(avg_mpos, avg_mpos, 1.0 / num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0 / num_body);

  // residual: average position (3 dims)
  mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
  counter += 3;

  // residuals: per-body position relative to the centroid (3 dims each)
  for (const auto& body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, avg_mpos);
    mju_subFrom3(body_sensor_pos, avg_sensor_pos);

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);
    counter += 3;
  }

  // ----- per-body linear velocity tracking ----- //
  // Reference velocity is computed by finite-differencing the mocap marker
  // between the two surrounding frames (so it's noisier but contains no qpos
  // information). Compared against the body's framelinvel sensor.
  for (const auto& body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    mju_copy3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
    mju_subFrom3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
    mju_scl3(&residual[counter], &residual[counter], kFps);

    double* sensor_linvel =
        SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);

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

  mj_freeStack(d);
}

}  // namespace mjpc::g1
