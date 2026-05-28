#ifndef MJPC_TASKS_G1_STAND_STAND_H_
#define MJPC_TASKS_G1_STAND_STAND_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc::g1 {

class Stand : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Stand() : residual_(this) {}

  // convenience accessors
  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc::g1

#endif  // MJPC_TASKS_G1_STAND_STAND_H_
