#include "mjpc/tasks/g1/stand/stand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::g1 {

// convinient accessor for model path
std::string Stand::XmlPath() const {
  return GetModelPath("g1/stand/task.xml");
}

//convenient accessor for task name
std::string Stand::Name() const { return "G1 Stand"; }

// ------------------ Residuals for G1 stand task ------------------
//   Number of residuals: 6
//     Residual (0): pelvis height above feet - height_goal
//     Residual (1): balance: capture point vs. average feet position (xy)
//     Residual (2): CoM xy velocity (should be 0)
//     Residual (3): ctrl - nominal pose (position actuators -> home keyframe)
//     Residual (4): upright: pelvis z-axis should point up
//     Residual (5): joint velocity (should be 0)
//   Number of parameters: 1
//     Parameter (0): height_goal
// -----------------------------------------------------------------
void Stand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;

  // ----- inputs ----- //
  double* pelvis_pos = SensorByName(model, data, "pelvis_position");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* lfoot = SensorByName(model, data, "left_foot_position");
  double* rfoot = SensorByName(model, data, "right_foot_position");
  double* com = SensorByName(model, data, "com");
  double* com_vel = SensorByName(model, data, "com_vel");

  // ----- (0) Height: pelvis above mean feet, minus goal ----- //
  double feet_z = 0.5 * (lfoot[2] + rfoot[2]);
  residual[counter++] = (pelvis_pos[2] - feet_z) - parameters_[0];

  // ----- (1) Balance: capture point vs. average feet xy ----- //
  double kFallTime = 0.2;
  double capture[3] = {com[0], com[1], com[2]};
  mju_addToScl3(capture, com_vel, kFallTime);

  double fxy[2] = {0.0};
  mju_addTo(fxy, lfoot, 2);
  mju_addTo(fxy, rfoot, 2);
  mju_scl(fxy, fxy, 0.5, 2);

  mju_subFrom(fxy, capture, 2);
  residual[counter++] = mju_norm(fxy, 2);

  // ----- (2) CoM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_vel, 2);
  counter += 2;

  // ----- (3) Ctrl - nominal (home keyframe), skip free-joint qpos (7) ----- //
  mju_sub(residual + counter, data->ctrl, model->key_qpos + 7, model->nu);
  counter += model->nu;

  // ----- (4) Upright: pelvis z-axis should point up ----- //
  residual[counter++] = pelvis_up[2] - 1.0;

  // ----- (5) Joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

}  // namespace mjpc::g1
