/*!
* \brief Record the basic usage of std::async.
*/

#include <iostream>

#include <chrono>
#include <functional>
#include <future>

// Parallel computing.
void TestAsync1() {

  auto Calculate = [](int ms, int num) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    return num;
  };

  clock_t time = clock();

  // Asynchronous.
  std::future<int> f1 = std::async(Calculate, 10, 1);
  std::future<int> f2 = std::async(Calculate, 20, 2);
  std::future<int> f3 = std::async(Calculate, 30, 3);

  // The get() blocks until the result is returned. 
  int sum = f1.get() + f2.get() + f3.get();
  std::cout << "test_feture_time result is :" << sum 
    << ", " << clock() - time << std::endl;
}

void PrintChar(char character) {
  for (int i = 0; i < 5; i++)
    std::cout << character;
};

void TestAsync2() {

  std::future<void> f1 = std::async(PrintChar, 'A');
  std::future<void> f2 = std::async(PrintChar, 'B');
  std::future<void> f3 = std::async(PrintChar, 'C');

  for (int i = 0; i < 5; i++) {
    std::cout << "D ";
  }

  f1.get(), f2.get(), f3.get();
  std::cout << std::endl;
}

void TestAsync3() {

  std::future<void> f1 = std::async(std::launch::async, PrintChar, 'A');
  std::future<void> f2 = std::async(std::launch::async, PrintChar, 'B');
  std::future<void> f3 = std::async(std::launch::async, PrintChar, 'C');

  for (int i = 0; i < 5; i++) {
    std::cout << "D ";
  }

  f1.get(), f2.get(), f3.get();
  std::cout << std::endl;
}

void TestAsync4() {

  std::future<void> f1 = std::async(std::launch::deferred, PrintChar, 'A');
  std::future<void> f2 = std::async(std::launch::deferred, PrintChar, 'B');
  std::future<void> f3 = std::async(std::launch::deferred, PrintChar, 'C');

  for (int i = 0; i < 5; i++) {
    std::cout << "D ";
  }

  // In std::launch::deferred.
  // The new thread was not immediately started to execute 
  // when the std::async function was created.
  // It only executes when get() is called.
  f1.get(), f2.get(), f3.get();
  std::cout << std::endl;
}

int main() {
  std::cout << "Test 1 : " << std::endl;
  TestAsync1();

  std::cout << "Test 2 <Default>: " << std::endl;
  TestAsync2();

  std::cout << "Test 3 <std::launch::async>: " << std::endl;
  TestAsync3();

  std::cout << "Test 4 <std::launch::deferred>: " << std::endl;
  TestAsync4();

  return 0;
}