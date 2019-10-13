#ifndef HCS_COMMON_H_
#define HCS_COMMON_H_

namespace hcs {

enum ExecutorMode {
  // Single thread.
  SERIAL = 0,
  // Multiple threads.
  PARALLEL,
  // Multiple threads & multiple streams.
  PARALLEL_MULTI_STREAMS
};

enum MemoryMode {
  ON_HOST = 0,
  ON_DEVICE
};

enum TypeFlag {
  FLOAT32 = 0,
  INT32 = 1,
  INT8 = 2,
  // Used to mark the total number of elements in TypeFlag.
  TYPES_NUM = 3  
};

enum ProfilerMode {
  LOCK_RUN_TO_SERIAL = 0x01,
  VIEW_NODE = 0x02,
  VIEW_STATUS = 0x04,
  VIEW_NODE_RUN_TIME = 0x08,
  VIEW_STATUS_RUN_TIME = 0x16,
};

// The main function is to bring in additional parameters to each task.
// Usage: 1. You need to customize a class and then inherit TaskAssistor.
//        2. Populate the class with the data you need to use in the tasks.
//        3. Use Bind() to pass the class into executor, and call it in tasks.
class TaskAssistor {
public:
  TaskAssistor() :streams_(nullptr) {}
  ~TaskAssistor() {
    if (streams_ != nullptr) {
      for (int i = 0; i < num_streams_; i++) {
        cudaStreamDestroy(streams_[i]);
      }
      //free(streams_);
      streams_ = nullptr;
    }
  }
  
  // Thread local variables.
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

  // Initialization for GPU side. 
  // Create streams for each thread.
  void Init4GPU(int num_streams) {
    num_streams_ = num_streams;
    streams_ = (cudaStream_t *)malloc(num_streams_ * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams_; i++) {
      cudaStreamCreate(&(streams_[i]));
    }
  }
  inline cudaStream_t stream() const {
    if (streams_ != nullptr)
      return streams_[id()];
    else
      return nullptr;
  }

private:
  int num_streams_;
  cudaStream_t *streams_;
};

}  // namespace hcs.

#endif // HCS_COMMON_H_