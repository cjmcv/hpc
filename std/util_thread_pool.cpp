/*!
* \brief Thread Pool.
*        Mainly implemented by thread, queue, future and condition_variable.
*/

#include <iostream>
#include <queue>
#include <thread>
#include <future>
#include <condition_variable>

class ThreadPool {
public:
  ThreadPool() : is_created_(false) {}
  ~ThreadPool() {};
  void CreateThreads(int thread_num);
  void ClearPool(); 

  template<class F, class... Args>
  auto TaskEnqueue(F&& f, Args&&... args)
    ->std::future<typename std::result_of<F(Args...)>::type>;

  void ParallelFor(std::function<void(const int, const int)> func, const int number);

private:
  // The Threads generated in this thread pool.
  std::vector< std::thread > workers_;
  // Stores the functional tasks that need to be run.
  std::queue< std::function<void()> > tasks_;
  // Gets the result from the asynchronous task.
  std::vector<std::future<void>> futures_;
  // Synchronization.
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool is_stop_;
  bool is_created_;
};

void ThreadPool::CreateThreads(int thread_num) {
  is_stop_ = false;
  if (is_created_ == true) {
    if (workers_.size() == thread_num)
      return;
    else {
      ClearPool();
      is_created_ = false;
    }
  }

  for (int i = 0; i < thread_num; ++i) {
    workers_.emplace_back([this] {
      // Threads live in this loop.
      while (1) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          // Waiting to be activated.
          this->condition_.wait(lock,
            [this] { return this->is_stop_ || !this->tasks_.empty(); });
          // If the thread pool is closed and the task queue is empty.
          if (this->is_stop_ && this->tasks_.empty())
            return;
          // Get a task from the tasks queue.
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        // Execute the task.
        task();
      }
    });
  }

  is_created_ = true;
}

void ThreadPool::ClearPool() {
  if (is_created_ == false)
    return;
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    is_stop_ = true;
  }

  // Activates all threads in the thread pool and
  // waits for all threads to complete their work.
  condition_.notify_all();
  for (std::thread &worker : workers_)
    worker.join();

  workers_.clear();
  tasks_ = decltype(tasks_)();

  is_created_ = false;
}

// Add a new task to the pool.
// If there is an inactive thread, the task will be executed immediately.
template<class F, class... Args>
auto ThreadPool::TaskEnqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type> {
  if (is_created_ == false)
    std::cout << "Error: Please create a Thread Pool first." << std::endl;

  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared< std::packaged_task<return_type()> >(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (is_stop_)
      throw std::runtime_error("TaskEnqueue on stopped ThreadPool");
    // Add a task to tasks queue.
    tasks_.emplace([task]() { (*task)(); });
  }
  // Activate/notify a thread.
  condition_.notify_one();
  return res;
}

// Use all threads in the thread pool to run the task.
// The workload per thread is split evenly.
// Note: The entire thread pool will serve this task 
//       and no other tasks should be inserted.
void ThreadPool::ParallelFor(std::function<void(const int, const int)> func, const int number) {
  if (number <= 0) {
    printf("[ ThreadPool::ParallelFor ]: number <= 0\n");
    return;
  }
  const int threads_num = workers_.size();
  if (threads_num <= 1 || number <= 1) {
    func(0, number);
  }
  else {
    const int datum_per_thread = number / threads_num;
    const int datum_remainder = number - datum_per_thread * threads_num;

    int start_num_idx = 0;
    futures_.clear();
    for (int i = 0; i < threads_num; i++) {
      int stop_num_idx = start_num_idx + datum_per_thread;
      if (i < datum_remainder)
        stop_num_idx = stop_num_idx + 1;

      futures_.emplace_back(TaskEnqueue(func, start_num_idx, stop_num_idx));

      start_num_idx = stop_num_idx;
      if (stop_num_idx >= number)
        break;
    }

    for (int i = 0; i < futures_.size(); i++)
      futures_[i].wait();
  }
}

void CompTest(int *a, int *b, int len) {
  printf("%d, ", a[0]);
  for (int i = 0; i < len; i++) {
    b[i] = a[i] + b[i];
  }
}

int main() {
  /////////////    Create      /////////////
  int len = 10;// 1000000;
  int num = 50;

  int **a = (int **)malloc(sizeof(int *) * num);
  int **b = (int **)malloc(sizeof(int *) * num);
  for (int i = 0; i < num; i++) {
    a[i] = (int *)malloc(sizeof(int) * len);
    b[i] = (int *)malloc(sizeof(int) * len);
  }

  for (int i = 0; i < num; i++) {
    for (int j = 0; j < len; j++) {
      a[i][j] = b[i][j] = i;// i*len + j;
    }
  }

  int thread_num = 15;
  ThreadPool thread_pool;
  thread_pool.CreateThreads(thread_num);

  /////////////    Process      /////////////
  bool is_test_task_enqueue = false;
  if(is_test_task_enqueue) { 
    // Test TaskEnqueue.
    auto func = [&](const int start, const int end) {
      printf("start(%d),end(%d)\n", start, end);
      for (int idx = start; idx < end; idx++)
        CompTest(*(a + idx), *(b + idx), len);
    };

    thread_pool.TaskEnqueue(func, 0, num);
    thread_pool.TaskEnqueue(func, 0, num);
  }
  else {  
    // Test ParallelFor.
    auto func = [&](const int start, const int end) {
      printf("start(%d),end(%d)\n", start, end);
      for (int idx = start; idx < end; idx++)
        CompTest(*(a + idx), *(b + idx), len);
    };

    thread_pool.ParallelFor(func, num);
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < len; j++) {
        printf("%d, ", b[i][j]);
      }
    }

    thread_pool.ParallelFor(func, num);
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < len; j++) {
        printf("%d, ", b[i][j]);
      }
    }
  }

  /////////////    Clear      /////////////
  thread_pool.ClearPool();
  for (int i = 0; i < num; i++) {
    free(a[i]);
    free(b[i]);
  }
  free(a);
  free(b);

  return 0;
}