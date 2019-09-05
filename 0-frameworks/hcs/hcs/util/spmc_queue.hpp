// Taken from: https://github.com/cpp-taskflow/cpp-taskflow

#ifndef HCS_WORKSTEALING_QUEUE_H_
#define HCS_WORKSTEALING_QUEUE_H_

#include <atomic>
#include <vector>
#include <optional>

namespace hcs {

/**
@class: WorkStealingQueue

@tparam T data type

@brief Lock-free unbounded single-producer multiple-consumer queue.

This class implements the work stealing queue described in the paper, 
"Dynamic Circular Work-stealing Deque," SPAA, 2015.
Only the queue owner can perform pop and push operations,
while others can steal data from the queue.

PPoPP implementation paper
"Correct and Efficient Work-Stealing for Weak Memory Models"
https://www.di.ens.fr/~zappa/readings/ppopp13.pdf

typedef enum memory_order {
    memory_order_relaxed,    //  The call has no effects.
    memory_order_acquire,    //  In this thread, all subsequent reads must be performed after
                             // this atomic operation completes.
    memory_order_release,    //  In this thread, all previous writes are completed before this 
                             // atomic operation can be performed.
    memory_order_acq_rel,    //  It contains both memory_order_acquire and memory_order_release.
    memory_order_consume,    //  All subsequent operations of this atomic type in this thread
                             // must be performed after this atomic operation is completed.
    memory_order_seq_cst     //  All access is sequential.
    } memory_order;
*/

template <typename T>
class WorkStealingQueue {

  //  This class contains the parts that need to be locked
  // when reading or writing by multiple threads.
  class Array {
  public:
    explicit Array(int64_t capacity) :
      capacity_{ capacity },
      circ_mask_{ capacity - 1 },    // Use & to turn the S into a circular queue.
      data_{ new T[static_cast<size_t>(capacity)] } {
    }
    ~Array() { delete[] data_; }

    inline int64_t capacity() const noexcept { return capacity_; }

    template <typename O>
    inline void fill(int64_t i, O&& o) noexcept { data_[i & circ_mask_] = std::forward<O>(o); }

    inline T fetch(int64_t i) noexcept { return data_[i & circ_mask_]; }
     
    inline Array *create(int64_t size, int64_t b, int64_t t) {
      Array *dst = new Array{ size };
      for (int64_t i = t; i != b; ++i) {
        dst->fill(i, this->fetch(i));
      }
      return dst;
    }

    inline void clear(Array *arr) { delete arr; }

  private:
    int64_t capacity_;
    int64_t circ_mask_;
    T* data_;
  };

public:
  // constructs the queue with a given capacity
  explicit WorkStealingQueue(int64_t capacity = 256) {
    assert(capacity && (!(capacity & (capacity - 1))));
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
    array_.store(new Array{ capacity }, std::memory_order_relaxed);
  }

  ~WorkStealingQueue() { delete array_.load(); }

  // queries if the queue is empty at the time of this call
  bool empty() const {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return b <= t; 
  }

  // queries the number of items at the time of this call
  size_t size() const { 
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return static_cast<size_t>(b >= t ? b - t : 0); 
  }

  // queries the capacity of the queue
  inline int64_t capacity() const { 
    return array_.load(std::memory_order_relaxed)->capacity(); 
  }

  // inserts an item to the queue
  // Only the owner thread can insert an item to the queue. 
  // The operation can trigger the queue to resize its capacity 
  // if more space is required.
  template <typename O>
  void push(O&& item);

  // pops out an item from the queue
  // Only the owner thread can pop out an item from the queue. 
  // The return can be a @std_nullopt if this operation failed (empty queue).
  std::optional<T> pop();

  // Pops out an item from the queue without synchronization with thieves.
  // Only the onwer thread can pop out an item from the queue, 
  // given no other threads trying to steal an item at the same time
  // The return can be a @std_nullopt if this operation failed (empty queue).
  std::optional<T> unsync_pop();

  // Steals an item from the queue
  // Any threads can try to steal an item from the queue.
  // The return can be a @std_nullopt if this operation failed (not necessary empty).
  std::optional<T> steal();

private:
  std::atomic<int64_t> top_;
  std::atomic<int64_t> bottom_;
  std::atomic<Array*> array_;
};

template <typename T>
template <typename O>
void WorkStealingQueue<T>::push(O&& o) {
  int64_t b = bottom_.load(std::memory_order_relaxed);
  int64_t t = top_.load(std::memory_order_acquire);
  Array* a = array_.load(std::memory_order_relaxed);

  // queue is full
  if(a->capacity() - 1 < (b - t)) {
    Array *new_ptr = a->create(a->capacity() * 2, b, t);
    new_ptr->clear(a);

    array_.store(new_ptr, std::memory_order_relaxed);
    a = new_ptr;
  }

  a->fill(b, std::forward<O>(o));
  std::atomic_thread_fence(std::memory_order_release);
  bottom_.store(b + 1, std::memory_order_relaxed);
}

// Function: pop
template <typename T>
std::optional<T> WorkStealingQueue<T>::pop() {
  int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
  Array* a = array_.load(std::memory_order_relaxed);
  bottom_.store(b, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = top_.load(std::memory_order_relaxed);

  std::optional<T> item;

  if(t <= b) {
    item = a->fetch(b);
    if(t == b) {
      // the last item just got stolen
      if(!top_.compare_exchange_strong(t, t+1, 
                                       std::memory_order_seq_cst, 
                                       std::memory_order_relaxed)) {
        item = std::nullopt;
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
    }
  }
  else {
    bottom_.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: unsync_pop
template <typename T>
std::optional<T> WorkStealingQueue<T>::unsync_pop() {

  int64_t t = top_.load(std::memory_order_relaxed);
  int64_t b = bottom_.fetch_sub(1, std::memory_order_relaxed) - 1;
  Array* a = array_.load(std::memory_order_relaxed);

  std::optional<T> item;

  if(t <= b) {
    item = a->fetch(b);
  }
  else {
    bottom_.store(b + 1, std::memory_order_relaxed);
  }

  return item;
}

// Function: steal
template <typename T>
std::optional<T> WorkStealingQueue<T>::steal() {
  int64_t t = top_.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = bottom_.load(std::memory_order_acquire);
  
  std::optional<T> item;

  if(t < b) {
    Array* a = array_.load(std::memory_order_consume);
    item = a->fetch(t);
    if(!top_.compare_exchange_strong(t, t+1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      return std::nullopt;
    }
  }

  return item;
}

}  // end of namespace hcs

#endif // HCS_WORKSTEALING_QUEUE_H_