/*!
* \brief The basic use of mutex in tbb.
* Mutex            Scalable         Fair      Reentrant   Sleeps       Size
* mutex          OS-dependent   OS-dependent      No       Yes    Three or more words
* spin_mutex          No             No           No        No       One byte
* queuing_mutex      Yes            Yes           No        No       One word
* spin_rw_mutex       No             No           No        No       One word
* queuing_rw_mutex   Yes            Yes           No        No       One word
*/
#include "iostream"
#include "time.h"

#include <tbb/tbb.h>  

void serialTest(int *value, int len) {
  for (int i = 0; i < len; i++) {
    (*value)++;
  }
}

class NormalTest {
public:
  int *value_;
  NormalTest(int *in) : value_(in) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    for (int k = r.begin(); k < r.end(); k++)
      (*value_)++;
  }
};

tbb::mutex mutex_lock;
class NormalMutexTest {
public:
  int *value_;

  NormalMutexTest(int *in) : value_(in) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    for (int k = r.begin(); k < r.end(); k++) {
      mutex_lock.lock();
      (*value_)++;
      mutex_lock.unlock();
    }
  }
};

class NormalMutexTest2 {
public:
  int *value_;

  NormalMutexTest2(int *in) : value_(in) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    tbb::mutex::scoped_lock mylock(mutex_lock);
    for (int k = r.begin(); k < r.end(); k++) {
      (*value_)++;
    }
  }
};

class NormalMutexTest3 {
public:
  int *value_;

  NormalMutexTest3(int *in) : value_(in) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    tbb::mutex::scoped_lock mylock;
    for (int k = r.begin(); k < r.end(); k++) {
      mylock.acquire(mutex_lock);
      (*value_)++;
      mylock.release();
    }
  }

};
/*!
* \Description :
*   A spin_mutex is not scalable, fair, or reentrant. It is ideal when the lock is lightly 
* contended and is held for only a few machine instructions.
*   If a task cannot acquire a spin_mutex when the class is created, it busy - waits, which 
* can degrade system performance if the wait is long.However, if the wait is typically short,
* a spin_mutex significantly improves performance compared to other mutexes.
*/
tbb::spin_mutex spin_lock;

/*!
* \Description :
*   A queuing_mutex is scalable, in the sense that if a task has to wait to acquire the mutex, 
* it spins on its own local cache line.A queuing_mutex is fair, in that tasks acquire the lock 
* in the order they requested it, even if they are later suspended.A queuing_mutex is not reentrant.
*   The current implementation does busy - waiting, so using a queuing_mutex may degrade
* system performance if the wait is long.
*/
tbb::queuing_mutex queue_lock;

/*!
* \Description :
*   A spin_rw_mutex is not scalable, fair, or reentrant.
*   It is ideal when the lock is lightly contended and is held for only a few machine instructions.
*   If a task cannot acquire a spin_rw_mutex when the class is created, it busy-waits, which can
* degrade system performance if the wait is long. However, if the wait is typically short,
* a spin_rw_mutex significantly improves performance compared to other ReaderWriterMutex mutexes.
*/
tbb::spin_rw_mutex spin_rw_lock;

/*!
* \Description :
*   A queuing_rw_mutex is scalable, in the sense that if a task has to wait to acquire the mutex,
* it spins on its own local cache line. A queuing_rw_mutex is fair, in that tasks acquire the
* lock in the order they requested it, even if they later are suspended.
*   A queuing_rw_mutex is not reentrant.
*/
tbb::queuing_rw_mutex queuing_rw_lock;

class MutexTest {
public:
  int *value_;

  MutexTest(int *in) : value_(in) {}
  void operator () (const tbb::blocked_range<size_t> & r) const {
    //tbb::spin_mutex::scoped_lock mylock(spin_lock);
    //tbb::queuing_mutex::scoped_lock mylock(queue_lock);
    //tbb::spin_rw_mutex::scoped_lock mylock(spin_rw_lock);
    tbb::queuing_rw_mutex::scoped_lock mylock(queuing_rw_lock);

    for (int k = r.begin(); k < r.end(); k++) {
      (*value_)++;
    }
  }
};

int main() {
  int countSize = 100000000;
  int a[1];
  *a = 0;

  time_t begin, end;	//CPU¼ÆÊ±
  begin = clock();

  //serialTest(a, countSize);
  //tbb::parallel_for(tbb::blocked_range<size_t>(0, countSize), NormalTest(a)); // Without mutex and get a wrong result.
  //tbb::parallel_for(tbb::blocked_range<size_t>(0, countSize), NormalMutexTest(a));
  tbb::parallel_for(tbb::blocked_range<size_t>(0, countSize), MutexTest(a));

  end = clock();
  printf("result = %d, time: %f \n", *a, double(end - begin));
  return 0;
}