/*!
* \brief timer.
*/

#ifndef HCS_TIMER_H_
#define HCS_TIMER_H_

#include <iostream>
#include <chrono>

namespace hcs {

// Timer for cpu.
class CpuTimer {
public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;

  inline void Start() { start_time_ = clock::now(); }
  inline void Stop() { stop_time_ = clock::now(); }
  inline float NanoSeconds() {
    return (float)std::chrono::duration_cast<ns>(stop_time_ - start_time_).count();
  }

  // Returns the elapsed time in milliseconds.
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }

  // Returns the elapsed time in microseconds.
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }

  // Returns the elapsed time in seconds.
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

protected:
  std::chrono::time_point<clock> start_time_;
  std::chrono::time_point<clock> stop_time_;
};

/////////////////////////////////////////////////
//  auto func = [&]()
//  -> float {
//    timer.Start();
//    hcs::QueryDevices();
//    timer.Stop();
//    return timer.MilliSeconds();
//  };
//  ret = func();
#define GET_TIME_DIFF(timer, ...)     \
  [&]() -> float {                    \
    timer.Start();                    \
    {__VA_ARGS__}                     \
    timer.Stop();                     \
    return timer.MilliSeconds();      \
  }();

// TODO: 3. ¼æÈÝGPUºËº¯Êý£¿
class Timer {
public:
  Timer(int node_idx, std::string name) :
    node_idx_(node_idx), 
    node_name_(name),
    count_(0), ave_(0),
    min_(FLT_MAX), max_(0) {}

  inline std::string node_name() const { return node_name_; }
  inline int node_idx() const { return node_idx_; }
  inline float min() const { return min_; }
  inline float max() const { return max_; }
  inline float ave() const { return ave_; }
  inline int count() const { return count_; }

  inline void Start() {
    if (is_record_)
      timer_.Start();
  }
  inline void Stop() {
    if (is_record_) {
      timer_.Stop();
      float time = timer_.MilliSeconds();
      if (time > max_) {
        max_ = time;
      }
      else if (time < min_) {
        min_ = time;
      }
      ave_ = (count_ / (count_ + 1.0)) * ave_ + time / (count_ + 1.0);
      count_++;
    }
  }

public:
  static bool is_record_;
  static bool lock2serial_;

private:
  CpuTimer timer_;
  int node_idx_;
  std::string node_name_;

  float min_;
  float max_;
  float ave_;
  int count_;
};

bool Timer::is_record_ = false;
bool Timer::lock2serial_ = false;

#define TIMER_PROFILER(timer, unique_lock, ...)    \
  [&]() -> void {                     \
    if (Timer::lock2serial_) {        \
      unique_lock;                    \
      timer.Start();                  \
      {__VA_ARGS__}                   \
      timer.Stop();                   \
    }                                 \
    else {                            \
      timer.Start();                  \
      {__VA_ARGS__}                   \
      timer.Stop();                   \
    }                                 \
  }();

} // hcs.
#endif //HCS_TIMER_H_
