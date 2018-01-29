/*!
* \brief count strings. This code is adapted from a example 
*        of the book - Intel Threading Building Blocks
* \tbb functions concurrent_hash_map, parallel_for
*/
#include <iostream>

#include "tbb/concurrent_hash_map.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include "tbb/task_scheduler_init.h"

// Structure that defines hashing and comparison operations for user's type.
struct MyHashCompare {
  // Hash function. The name of this function can not be changed, 
  // and it will be called in concurrent_hash_map.h
  static size_t hash(const std::string& x) {
    size_t h = 0;
    for (const char* s = x.c_str(); *s; s++)
      h = (h * 17) ^ *s;
    return h;
  }
  // True if strings are equal.
  // This function will also be called in concurrent_hash_map.h,
  // to check whether there has an element is equal to the candidate.
  // ps: Can not change the function name either.
  static bool equal(const std::string& x, const std::string& y) {
    return x == y;
    //return 0; // Then insert all of the strings independently.
  }
};

// A concurrent hash table that maps strings to ints.
typedef tbb::concurrent_hash_map<std::string, int, MyHashCompare> StringTable;
// Function object for counting occurrences of strings.
struct Tally {
  StringTable& table_;
  Tally(StringTable& table) : table_(table) {}
  void operator( )(const tbb::blocked_range<std::string*> range) const {
    for (std::string* p = range.begin(); p != range.end(); ++p) {
      StringTable::accessor a; 
      // Insert an element through the accessor.
      // It will call hash and equal functions in MyHashCompare.
      table_.insert(a, *p);
      // Record how many times the same element is inserted.
      a->second += 1;
    }
  }
};

static const std::string Adjective[] =
{ "sour", "sweet", "bitter", "salty", "big", "small" };
static const std::string Noun[] =
{ "apple", "banana", "cherry", "date", "eggplant", "fig", "grape", "honeydew", "icao", "jujube" };
static void CreateInputData(std::string *str_container, const int str_container_size) {
  srand(2);
  size_t n_adjective = sizeof(Adjective) / sizeof(Adjective[0]);
  size_t n_noun = sizeof(Noun) / sizeof(Noun[0]);
  for (int i = 0; i < str_container_size; ++i) {
    str_container[i] = Adjective[rand() % n_adjective];
    str_container[i] += " ";
    str_container[i] += Noun[rand() % n_noun];
  }
}

int main(int argc, char* argv[]) {
  // Set to true to counts.
  static bool ver_bose = true;
  // Working threads count
  static int num_thread = 10;
  // Problem size
  const size_t len = 1000000;

  // TBB Initialize. This affects the number of threads.
  // Each thread must initialize the Threading Building Blocks library using
  // tbb::task_scheduler_init before using an algorithm template or the task scheduler.
  tbb::task_scheduler_init init(num_thread);

  static std::string *str_container = new std::string[len];
  CreateInputData(str_container, len);

  StringTable table;
  tbb::tick_count t0 = tbb::tick_count::now();
  // Insert data with multi-threads.
  tbb::parallel_for(tbb::blocked_range<std::string*>(str_container, str_container + len, 1000), Tally(table));
  tbb::tick_count t1 = tbb::tick_count::now();

  // Check the total number of strings in map.
  int n = 0;
  for (StringTable::iterator i = table.begin(); i != table.end(); ++i) {
    if (ver_bose)
      std::cout << i->first.c_str() << " -> " << i->second << std::endl;
    n += i->second;
  }
  std::cout << "threads = "<< num_thread << ", total = " << n << ", time = " << (t1 - t0).seconds() << std::endl;

  delete[] str_container;
  return 0;
}