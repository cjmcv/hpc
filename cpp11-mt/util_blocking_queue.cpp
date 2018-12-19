/*!
* \brief Blocking queue.
*        Mainly implemented by queue, condition_variable and unique_lock.
*/

#include <iostream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <windows.h>

template<typename T>
class BlockingQueue {
public:
  BlockingQueue(int capacity) {
    capacity_ = capacity;
  };
  ~BlockingQueue() {};

  void push(const T& t);
  void wait_and_push(const T& t);
  bool try_pop(T* t);
  void wait_and_pop(T* t);

private:
  mutable std::mutex mutex_;
  std::condition_variable push_cond_var_;
  std::condition_variable pop_cond_var_;
  std::queue<T> queue_;
  int capacity_;
};

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  std::unique_lock <std::mutex> lock(mutex_);
  queue_.push(t);
  lock.unlock();

  push_cond_var_.notify_one();
}

template<typename T>
void BlockingQueue<T>::wait_and_push(const T& t) {
  std::unique_lock <std::mutex> lock(mutex_);
  while(queue_.size() >= capacity_)
    pop_cond_var_.wait(lock);

  queue_.push(t);
  push_cond_var_.notify_one();
}

//template<typename T>
//bool BlockingQueue<T>::full() const {
//  std::unique_lock <std::mutex> lock(mutex_);
//  if(queue_.size() >= capacity_)
//    return true
//  return false;
//}
//
//template<typename T>
//bool BlockingQueue<T>::empty() const {
//  std::unique_lock <std::mutex> lock(mutex_);
//  return queue_.empty();
//}
//
//template<typename T>
//int BlockingQueue<T>::size() const {
//  std::unique_lock <std::mutex> lock(mutex_);
//  return queue_.size();
//}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  if (queue_.empty())
    return false;

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
void BlockingQueue<T>::wait_and_pop(T* t) {
  std::unique_lock <std::mutex> lock(mutex_);
  while (queue_.empty())
    push_cond_var_.wait(lock);

  *t = queue_.front();
  queue_.pop();
  pop_cond_var_.notify_one();
}

///////////////////////////////////////////////////////////

template<typename T>
class Worker {
public:
  Worker(int capacity, T *src) {
    blocking_queue_ = new BlockingQueue<T>(capacity);
    src_ = src;
    index_ = 0;
  }
  ~Worker() {
    delete blocking_queue_;
  }
  void Produce() {
    while (1) {
      blocking_queue_->push(src_[index_++]);
      //Sleep(1);
    }
  }

  void Consume() {
    while (1) {
      T data;
      blocking_queue_->wait_and_pop(&data);
      //std::cout << "(" << data << ", " << blocking_queue_.size() << ")";
      std::cout << data << ", ";
    }
  } 

private:
  BlockingQueue<T> *blocking_queue_; 
  T *src_;
  int index_;
};

void thread1(Worker<int> *worker) {
  worker->Produce();
}
void thread2(Worker<int> *worker) {
  worker->Consume();
}

int main() {
  int len = 10000000;
  int *src_data = new int[len];
  for (int i = 0; i < len; i++) {
    src_data[i] = i;
  }

  int capacity = 2000;
  Worker<int> worker(capacity, src_data);
  std::thread t0 = std::thread(thread1, &worker);
  std::thread t1 = std::thread(thread2, &worker);

  t0.join();
  t1.join();

  delete[]src_data;
  return 0;
}