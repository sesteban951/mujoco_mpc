// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_CARTPOLE_WALLS_CARTPOLE_WALLS_H_
#define MJPC_TASKS_CARTPOLE_WALLS_CARTPOLE_WALLS_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class CartpoleWalls : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const CartpoleWalls* task) : BaseResidualFn(task) {}
    // Same residual as the vanilla Cartpole task:
    //   (0) cos(pole_angle) - 1  (upright)
    //   (1) cart_pos - goal      (centered)
    //   (2) pole_angular_vel     (velocity)
    //   (3) ctrl[0]              (effort)
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  CartpoleWalls() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_CARTPOLE_WALLS_CARTPOLE_WALLS_H_
