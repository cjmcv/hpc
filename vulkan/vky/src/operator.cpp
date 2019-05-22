#include "operator.h"

namespace vky {

/////////////////////////
// Operator <protected
void Operator::GetOptimalLocalSizeXYZ(const uint32_t *max_workgroup_size,
                                      const uint32_t max_workgroup_invocations,
                                      const uint32_t width,
                                      const uint32_t height,
                                      const uint32_t channels) {
  if (channels > 0) {
    local_size_xyz_[2] = max_workgroup_size[2];
    while (channels < local_size_xyz_[2]) {
      local_size_xyz_[2] /= 2;
    }
  }
  else {
    local_size_xyz_[2] = std::min((uint32_t)128, max_workgroup_size[2]);
  }

  uint32_t max_local_size_xy = max_workgroup_invocations / local_size_xyz_[2];

  if (height == width || (height < 0 && width < 0)) {
    uint32_t local_size_xy = std::sqrt(max_local_size_xy);
    uint32_t local_size_xy_prefer = 128;
    while (local_size_xy < local_size_xy_prefer) {
      local_size_xy_prefer /= 2;
    }
    local_size_xyz_[0] = local_size_xy_prefer;
    local_size_xyz_[1] = local_size_xy_prefer;
  }

  if (height > 0 && width > 0) {
    if (height > width) {
      float ps = height / (float)width;
      float local_size_xy = sqrt(max_local_size_xy / ps);
      local_size_xyz_[1] = local_size_xy * ps;
      local_size_xyz_[0] = std::max((uint32_t)local_size_xy, (uint32_t)1);
    }
    else {
      float ps = width / (float)height;
      float local_size_xy = sqrt(max_local_size_xy / ps);
      local_size_xyz_[1] = std::max((uint32_t)local_size_xy, (uint32_t)1);
      local_size_xyz_[0] = local_size_xy * ps;
    }

    uint32_t local_size_y_prefer = std::min((uint32_t)128, max_workgroup_size[1]);
    while (local_size_xyz_[1] < local_size_y_prefer) {
      local_size_y_prefer /= 2;
    }

    uint32_t local_size_x_prefer = std::min((uint32_t)128, max_workgroup_size[0]);
    while (local_size_xyz_[0] < local_size_x_prefer) {
      local_size_x_prefer /= 2;
    }

    local_size_xyz_[1] = local_size_y_prefer;
    local_size_xyz_[0] = local_size_x_prefer;
  }
  else if (height > 0) {
    local_size_xyz_[1] = std::min(max_local_size_xy, (uint32_t)max_workgroup_size[1]);
    while ((uint32_t)height < local_size_xyz_[1]) {
      local_size_xyz_[1] /= 2;
    }

    uint32_t max_local_size_x = max_local_size_xy / local_size_xyz_[1];
    local_size_xyz_[0] = std::min(max_local_size_x, (uint32_t)max_workgroup_size[0]);
  }
  else if (width > 0) {
    local_size_xyz_[0] = std::min(max_local_size_xy, (uint32_t)max_workgroup_size[0]);
    while ((uint32_t)width < local_size_xyz_[0]) {
      local_size_xyz_[0] /= 2;
    }

    uint32_t max_local_size_y = max_local_size_xy / local_size_xyz_[0];
    local_size_xyz_[1] = std::min(max_local_size_y, (uint32_t)max_workgroup_size[1]);
  }
}

void Operator::GetGroupCount(const uint32_t width, const uint32_t height, const uint32_t channels) {
  group_count_xyz_[0] = (width + local_size_xyz_[0] - 1) / local_size_xyz_[0];
  group_count_xyz_[1] = (height + local_size_xyz_[1] - 1) / local_size_xyz_[1];
  group_count_xyz_[2] = (channels + local_size_xyz_[2] - 1) / local_size_xyz_[2];
}

} // namespace vky