#ifndef HCS_COMMON_H_
#define HCS_COMMON_H_

namespace hcs {

enum MemoryMode {
  ON_HOST = 0,
  ON_DEVICE
};

enum TypeFlag {
  FLOAT32 = 0,
  INT32 = 1,
  INT8 = 2,
  TYPES_NUM = 3  // Used to mark the total number of elements in TypeFlag.
};

enum ProfilerMode {
  VIEW_NODE = 0x01,
  VIEW_STATUS = 0x02
};

}  // namespace hcs.

#endif // HCS_COMMON_H_