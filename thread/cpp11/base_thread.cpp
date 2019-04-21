/*!
* \brief Multitasking.
*/

#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;
void Function(int index) {
  int num_iter = 5;
  for (int i = 0; i < num_iter; i++) {
    mtx.lock();
    std::cout << "(" << index << "," << std::this_thread::get_id() 
      << "," << i << ")" << std::endl;
    mtx.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

int main() {
  int num_threads = 5;

  std::cout << "(index, thread_id, iter)" << std::endl;

  std::thread *threads = new std::thread[num_threads];
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread(Function, i);
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  return 0;
}