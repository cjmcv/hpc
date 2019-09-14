/*!
* \brief Internal Thread.
*        Mainly implemented by thread.
*/

#ifndef HCS_INTERNAL_THREAD_H_
#define HCS_INTERNAL_THREAD_H_

#include <thread>

namespace hcs {

class InternalThread {
public:
  InternalThread() : thread_(), interrupt_flag_(false) {}
  virtual ~InternalThread() { StopInnerThread(); }

  // To chech wether the internal thread has been started. 
  inline bool is_started() const { return thread_ && thread_->joinable(); }

  bool StartInnerThread();
  void StopInnerThread();

protected:
  // Virtual function, should be override by the classes
  // which needs a internal thread to assist.
  virtual void EntryInnerThread() {}
  bool IsMustStop();

private:
  bool interrupt_flag_;
  std::shared_ptr<std::thread> thread_;
};

bool InternalThread::IsMustStop() {
  if (thread_ && interrupt_flag_) {
    interrupt_flag_ = false;
    return true;
  }
  return false;
}

bool InternalThread::StartInnerThread() {
  if (is_started()) {
    printf("Threads should persist and not be restarted.");
    return false;
  }
  try {
    thread_.reset(new std::thread(&InternalThread::EntryInnerThread, this));
  }
  catch (std::exception& e) {
    printf("Thread exception: %s", e.what());
  }

  return true;
}

void InternalThread::StopInnerThread() {
  if (is_started()) {
    // This flag will work in must_stop.
    interrupt_flag_ = true;
    try {
      thread_->join();
    }
    catch (std::exception& e) {
      printf("Thread exception: %s", e.what());
    }
  }
}

} // namespace hcs

#endif // HCS_INTERNAL_THREAD_H_
