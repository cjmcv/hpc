#ifndef HCS_COMMON_H_
#define HCS_COMMON_H_

namespace hcs {

enum ExecutorMode {
  SERIAL = 0,
  PARALLEL,
  PARALLEL_USE_GPU
};

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
  VIEW_NODE_RUN_TIME = 0x08,
  VIEW_STATUS_RUN_TIME = 0x16,
};

class TaskAssistor {
public:
  TaskAssistor() {}
  ~TaskAssistor() {}

  int ass_a;

  inline int id() const {
    return thread_var().id;
  }

  struct ThreadVar {
    int id = -1;
  };

  inline ThreadVar &thread_var() const {
    thread_local ThreadVar thread_var;
    return thread_var;
  }
};

}  // namespace hcs.

#endif // HCS_COMMON_H_