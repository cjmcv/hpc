/*!
* \brief Internal Thread.
*        Mainly implemented by thread.
*/

#include <iostream>
#include <thread>

class InternalThread {
public:
  InternalThread();
  virtual ~InternalThread();

  bool StartInnerThread();
  void StopInnerThread();

  // To chech wether the internal thread has been started. 
  bool is_started() const;

protected:
  // Virtual function, should be override by the classes
  // which needs a internal thread to assist.
  virtual void EntryInnerThread() {}
  bool must_stop();

private:
  bool interrupt_flag_;
  std::shared_ptr<std::thread> thread_;
};

InternalThread::InternalThread()
  : thread_(), interrupt_flag_(false) {}

InternalThread::~InternalThread() {
  StopInnerThread();
}

inline bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  if (thread_ && interrupt_flag_) {
    interrupt_flag_ = false;
    return true;
  }
  else
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

////////////////////// Test //////////////////////////////
class InternalThreadTest : public InternalThread {
private:
  void EntryInnerThread() {
    while (!must_stop()) {
      std::cout << "0";
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }
};

int main() {
  InternalThreadTest test;
  bool ret = test.StartInnerThread();
  std::cout << "Start: " << ((ret == true) ? "Succeeded" : "Failed") << std::endl;

  for (int i = 0; i < 100; i++) {
    std::cout << "1";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::cout << std::endl << "End." << std::endl;
  test.StopInnerThread();

  return 0;
}