// G1 Tracking task: follow a reference motion clip by matching the world-frame
// positions/velocities of selected G1 bodies to "mocap" target markers driven
// from interpolated keyframes. Mirrors mjpc/tasks/humanoid/tracking/.

#ifndef MJPC_TASKS_G1_TRACKING_TRACKING_H_
#define MJPC_TASKS_G1_TRACKING_TRACKING_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc::g1 {

class Tracking : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Tracking* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Tracking;
    int current_mode_;
    double reference_time_;
  };

  Tracking() : residual_(this) {}

  // Advances time through the active mocap clip and writes interpolated
  // marker positions into data->mocap_pos every step. On mode switch (or t=0)
  // it also teleports the robot to the clip's first-frame qpos/qvel.
  void TransitionLocked(mjModel* model, mjData* data) override;

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc::g1

#endif  // MJPC_TASKS_G1_TRACKING_TRACKING_H_
