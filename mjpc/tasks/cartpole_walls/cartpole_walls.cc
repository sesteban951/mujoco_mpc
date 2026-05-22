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

#include "mjpc/tasks/cartpole_walls/cartpole_walls.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string CartpoleWalls::XmlPath() const {
  return GetModelPath("cartpole_walls/task.xml");
}
std::string CartpoleWalls::Name() const { return "Cartpole Walls"; }

void CartpoleWalls::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  residual[0] = std::cos(data->qpos[1]) - 1;
  residual[1] = data->qpos[0] - parameters_[0];
  residual[2] = data->qvel[1];
  residual[3] = data->ctrl[0];
}

}  // namespace mjpc
