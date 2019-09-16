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
  virtual ~InternalThread() { Stop(); }

  // To chech wether the internal thread has been started. 
  inline bool is_started() const { return thread_ && thread_->joinable(); }
  inline int function_id() const { return function_id_; }
  inline int ms() const { return ms_; }

  bool Start(int func_id, int ms);
  void Stop();

protected:
  // Virtual function, should be override by the classes
  // which needs a internal thread to assist.
  virtual void Entry() {}
  bool IsMustStop();

private:
  bool interrupt_flag_;
  std::shared_ptr<std::thread> thread_;
  int function_id_;
  int ms_;
};

bool InternalThread::IsMustStop() {
  if (thread_ && interrupt_flag_) {
    interrupt_flag_ = false;
    return true;
  }
  return false;
}

bool InternalThread::Start(int func_id, int ms) {
  function_id_ = func_id;
  ms_ = ms;
  if (is_started()) {
    printf("Threads should persist and not be restarted.");
    return false;
  }
  try {
    thread_.reset(new std::thread(&InternalThread::Entry, this));
  }
  catch (std::exception& e) {
    printf("Thread exception: %s", e.what());
  }

  return true;
}

void InternalThread::Stop() {
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
