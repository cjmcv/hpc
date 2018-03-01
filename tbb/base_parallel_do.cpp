/*!
* \brief The basic use of parallel_do.
* \note  The function of parallel_do is nearly the same as
*      parallel_while, but parallel_while is deprecated.
*/

#include <iostream>
#include <tbb/tbb.h>
#include <time.h>

using Item = int;

void Foo(Item elem) {
  //Sleep(10);
  //printf("%d, ", elem);

  int tt = 0;
  for (int i = 0; i < elem*100; i++) {
    if (i % 123 == 0)
      tt++;
  }
  printf("%d, ", tt);
}

// It is an option to choose std:list, std::vector or others.
void SerialApplyFooToList(const std::list<Item>& list) {
  for (std::list<Item>::const_iterator i = list.begin();
    i != list.end(); ++i)
    Foo(*i);
}

class ApplyFoo {
public:
  // operator's entry must be Item,
  // which is the type of elements in list.
  void operator()(Item item) const {
    Foo(item);
  }
};

void ParallelApplyFooToList(const std::list<Item>& list) {
  tbb::parallel_do(list.begin(), list.end(), ApplyFoo());
}

int main() {
  // Initialize the list.
  std::list<Item> list;
  for (int i = 0; i < 1000; i++) {
    list.push_back(i);
  }

  time_t stime;
  stime = clock();
  SerialApplyFooToList(list);
  std::cout << "\n Serial ->  time: " << clock() - stime << std::endl;

  stime = clock();
  ParallelApplyFooToList(list);
  std::cout << "\n TBB Parallel ->  time: " << clock() - stime << std::endl;
  return 0;
}