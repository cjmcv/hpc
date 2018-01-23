/*!
* \brief The basic use of concurrent queue in tbb.
*/
#include <iostream>
#include <tbb/tbb.h>

// A test class for testing the performace of tbb::concurrent_queue in multithreading.
class ConcurrentQueueTest {
public:
  ConcurrentQueueTest(int test_num) : test_num_(test_num) {}
  void push(int thread_id) {
    const int start = thread_id * test_num_;
    const int end = start + test_num_;
    for (int val = start; val < end; val++)
      queue_.push(val);
  }

  void pop(int thread_id) {
    int val;
    while (1) {
      if (queue_.try_pop(val)) {
        std::cout << val << "(" << thread_id << "), ";
      }
      else {
        Sleep(1);
      }
    }
  }

private:
  // A high - performance thread - safe non - blocking concurrent queue.
  // And it is very useful in multi-threads program.
  tbb::concurrent_queue<int> queue_;
  int test_num_;
};

// Only for Creating multiple threads.
class ThreadCreater {
public:
  ThreadCreater(ConcurrentQueueTest *queue_test) : queue_test_(queue_test) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      // Let half of the threads to excute push operation.
      if (i <= (r.size() + 1) / 2) {
        queue_test_->push(i);
      }
      // The rest to excute pop.
      else if (i > r.size() / 2) {
        queue_test_->pop(i);
      }
    }
  }
private:
  ConcurrentQueueTest *queue_test_;
};

void main() {
  ConcurrentQueueTest queue_test(300);
  // Create 3 thread for testing.
  tbb::parallel_for(tbb::blocked_range<size_t>(0, 3, 1), ThreadCreater(&queue_test));
}