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

// TODO: 1. 时间记录，标志位由profiler控制，if则执行，else为空。数据记录到自身的列表中，由profiler去获取
//       2. 每个线程都会有一个Timer对象，即一个Timer对应一个node
//       3. 兼容GPU核函数？
class Timer {
public:
  Timer() {}

  void Start() {
    if (is_record_)
      timer_.Start();
  }
  void Stop() {
    if (is_record_) {
      timer_.Stop();
      float time = timer_.MilliSeconds();
    }
  }

public:
  static bool is_record_;

private:
  CpuTimer timer_;
  float lists_;
};

bool Timer::is_record_ = false;

} // hcs.
#endif //HCS_TIMER_H_
