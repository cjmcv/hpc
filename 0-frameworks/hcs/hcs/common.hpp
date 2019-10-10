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
  LOCK_RUN_TO_SERIAL = 0x01,
  VIEW_NODE = 0x02,
  VIEW_STATUS = 0x04,
  VIEW_TIMER = 0x08,

};

}  // namespace hcs.

#endif // HCS_COMMON_H_