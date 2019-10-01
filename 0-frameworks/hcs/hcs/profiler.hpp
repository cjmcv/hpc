#ifndef HCS_PROFILER_H_
#define HCS_PROFILER_H_

#include <iostream>

#include "util/internal_thread.hpp"
#include "executor.hpp"
#include "graph.hpp"

// TODO: 添加调试打印控制，设置后所有信息都打印，否则所有信息都不打印。

namespace hcs {

class Profiler :public InternalThread {
public:
  Profiler(Executor *exec, Graph *graph) { exec_ = exec; graph_ = graph; }

private:
  void ViewNode() {
    for (int i = 0; i < graph_->nodes().size(); i++) {
      Node *n = &(*(graph_->nodes()[i]));
      printf("(%s: ", n->name().c_str());
      printf(" %d ", n->num_cached_buf(0));
      for (int si = 1; si < n->num_successors(); si++) {
        printf(" %d ", n->num_cached_buf(si));
      }
      printf("<%d>", n->num_empty_buf());
      printf(", %d)", n->run_count());
    }
    printf("\n");
  }

  void ViewStatus() {
    std::vector<Executor::Status*> list = exec_->status_list_;

    for (int i = 0; i < list.size(); i++) {
      if (list[i] == nullptr) {
        printf("%d> nullptr.\n", i);
        continue;
      }
      else
        printf("%d> %d -> ", i, list[i]->num_incomplete_out_nodes);

      printf("\n");
    }
  }

  void Entry() {
    while (!IsMustStop()) {
      switch (function_id()) {
      case 0:
        ViewNode();
        break;
      case 1:
        ViewStatus();
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(ms()));
    }
  }

private:
  Executor *exec_;
  Graph *graph_;
};

}  // namespace hcs.

#endif // HCS_PROFILER_H_