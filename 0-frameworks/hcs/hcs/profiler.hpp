#ifndef HCS_PROFILER_H_
#define HCS_PROFILER_H_

#include <iostream>

#include "util/internal_thread.hpp"
#include "graph.hpp"

namespace hcs {

class Profiler {
public:
  static void StatusView(Graph& g) {
    for (int i = 0; i < g.nodes().size(); i++) {
      Node *n = &(*g.nodes()[i]);
      printf("(%s: %d, %d)", n->name().c_str(), n->outs_full_.size(), n->atomic_run_count_.load());
    }
    printf("\n");
  }
};

}  // namespace hcs.

#endif // HCS_PROFILER_H_