/*!
* \brief Blocking queue.
*        Mainly implemented by thread, queue and condition_variable.
*/

#include <iostream>
#include <thread>
#include <queue>
#include <condition_variable>

template<typename T>
class BlockingQueue {
public:
  BlockingQueue(int capacity) {
    capacity_ = capacity;
  };
  ~BlockingQueue() {};

  int size() const;

  void push(const T& t);
  void wait_and_push(const T& t);
  bool try_pop(T* t);
  void wait_and_pop(T* t);

private:
  int capacity_;
  std::queue<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable pop_cond_var_;
  std::condition_variable push_cond_var_;
};

template<typename T>
int BlockingQueue<T>::size() const {
  std::unique_lock <std::mutex> lock(mutex_);
  return queue_.size();
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  std::unique_lock <std::mutex> lock(mutex_);
  queue_.push(t);
  lock.unlock();

  pop_cond_var_.notify_one();
}

template<typename T>
void BlockingQueue<T>::wait_and_push(const T& t) {
  std::unique_lock <std::mutex> lock(mutex_);
  while(queue_.size() >= capacity_)
    push_cond_var_.wait(lock);

  queue_.push(t);
  pop_cond_var_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  if (queue_.empty())
    return false;

  *t = queue_.front();
  queue_.pop();

  push_cond_var_.notify_one();
  return true;
}

template<typename T>
void BlockingQueue<T>::wait_and_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  while (queue_.empty())
    pop_cond_var_.wait(lock);

  *t = queue_.front();
  queue_.pop();

  push_cond_var_.notify_one();
}
///////////////////////////////////////////////////////////

template<typename T>
class Worker {
public:
  Worker(int capacity) {
    blocking_queue_ = new BlockingQueue<T>(capacity);
    num_ = 0;
  }
  ~Worker() {
    delete blocking_queue_;
  }
  void Produce() {
    while (1) {
      blocking_queue_->wait_and_push(num_++);
    }
  }

  void Consume() {
    while (1) {
      T data;
      blocking_queue_->wait_and_pop(&data);
      std::cout << "(" << data << ", " << blocking_queue_->size() << ")";
    }
  }

private:
  BlockingQueue<T> *blocking_queue_; 
  int num_;
};

int main() {

  auto Func_Produce = [](Worker<int> *worker) {
    worker->Produce();
  };
  auto Func_Consume = [](Worker<int> *worker) {
    worker->Consume();
  };

  Worker<int> worker(2000);
  std::thread t0 = std::thread(Func_Produce, &worker);
  std::thread t1 = std::thread(Func_Consume, &worker);

  t0.join();
  t1.join();

  return 0;
}