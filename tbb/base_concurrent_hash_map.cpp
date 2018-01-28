/*!
* \brief The basic use of concurrent_hash_map in tbb.
*        The STL map is not safe for concurrent use.
*/

#include <iostream>
#include <tbb/tbb.h>

typedef tbb::concurrent_hash_map<std::string, int> NameTable;

void MapInsert(NameTable& table, std::string& key, int value) {
  // Use an accessor to make sure it is safe for concurrent use.
  NameTable::accessor acc;
  // Manipulate the data through the accessor.
  table.insert(acc, key);
  acc->second = value;

  acc.release();
}

int main(int argc, char* argv[]) {
  NameTable table;
  NameTable::iterator table_iter;
  std::string name_a = "zhangsan";
  std::string name_b = "lisi";
  MapInsert(table, name_a, 24);
  MapInsert(table, name_b, 25);

  for (table_iter = table.begin(); table_iter != table.end(); table_iter++) {
    std::cout << "(" << table_iter->first << "," << table_iter->second << ")" << std::endl;
  }
  return 0;
}