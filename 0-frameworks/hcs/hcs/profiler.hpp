#ifndef HCS_PROFILER_H_
#define HCS_PROFILER_H_

#include <iostream>

#include "util/internal_thread.hpp"
#include "executor.hpp"
#include "graph.hpp"

namespace hcs {

class Profiler :public InternalThread {
public:
  Profiler(Executor *exec, Graph *graph) { exec_ = exec; graph_ = graph; }

private:
  void ViewNode() {
    for (int i = 0; i < graph_->nodes().size(); i++) {
      Node *n = &(*(graph_->nodes()[i]));
      printf("(%s: %d, %d)", n->name().c_str(), n->outs_full_.size(), n->atomic_run_count_.load());
    }
    printf("\n");
  }

  void ViewStatus() {
    std::vector<Executor::Status*> list = exec_->status_list();

    for (int i = 0; i < list.size(); i++) {
      if (list[i] == nullptr) {
        printf("%d> nullptr.\n", i);
        continue;
      }
      else
        printf("%d> %d -> ", i, list[i]->num_incomplete_out_nodes);

      std::map<std::string, std::atomic<int>>::iterator iter;
      iter = list[i]->depends.begin();
      while (iter != list[i]->depends.end()) {
        printf("%s: %d, ", iter->first.c_str(), iter->second.load());
        iter++;
      }
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