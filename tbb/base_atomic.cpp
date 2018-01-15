/*!
* \brief The basic use of atomic.
*/

#include <iostream>
#include <tbb/tbb.h>

class BodyTest {
public:
	BodyTest(tbb::atomic<int> *a, int *b) : atomic_value_(a), normal_value_(b) {}

	void operator () (const tbb::blocked_range<size_t> & r) const	{
		for(int i=0; i<10000; i++) {
			//(*atomic_value_).fetch_and_add(1);
			(*atomic_value_)++;		// ++ Operator overloading, it is the same as fetch_and_add.
			(*normal_value_)++;		// Can not get the right result.
		}	
	}

private:
  tbb::atomic<int> *atomic_value_;
  int *normal_value_;
};

int main() {
	tbb::atomic<int> atomic_value[1];
  atomic_value[0] = 0;

	int normal_value[1];
  normal_value[0] = 0;

  // ps: tbb::atomic can not get the right answer when the number of threads is too many. (over 500?)
	tbb::parallel_for(tbb::blocked_range<size_t>(0,500), BodyTest(atomic_value, normal_value));

	std::cout << "atomic: " << *atomic_value << std::endl;
  std::cout << "normal: " << *normal_value << std::endl;

  return 0;
}