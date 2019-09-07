#ifndef HCS_BLOCKING_QUEUE_H_
#define HCS_BLOCKING_QUEUE_H_

#include <queue>
namespace hcs {

template <typename T>
class BlockingQueue {
public:
  BlockingQueue() {};
  ~BlockingQueue() {};

  void push(const T& t) {
    std::unique_lock <std::mutex> lock(mutex_);
    queue_.push(t);
    lock.unlock();
    cond_var_.notify_one();
  }

  bool try_pop(T* t) {
    std::unique_lock <std::mutex> lock(mutex_);
    if (queue_.empty())
      return false;

    *t = queue_.front();
    queue_.pop();
    return true;
  }

  void wait_and_pop(T* t) {
    std::unique_lock <std::mutex> lock(mutex_);
    while (queue_.empty())
      cond_var_.wait(lock);

    *t = queue_.front();
    queue_.pop();
  }

  bool empty() const {
    std::unique_lock <std::mutex> lock(mutex_);
    return queue_.empty();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
  std::queue<T> queue_;
};

}  // namespace hcs

#endif // HCS_BLOCKING_QUEUE_H_