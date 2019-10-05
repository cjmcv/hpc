#ifndef HCS_PROFILER_H_
#define HCS_PROFILER_H_

#include <iostream>

#include "util/internal_thread.hpp"
#include "executor.hpp"
#include "graph.hpp"

namespace hcs {

// TODO: 1. 使用 位与 选择功能;  2.时间测试，在内部调用，在demo中将不能使用timer。
class Profiler :public InternalThread {
public:
  Profiler(Executor *exec, Graph *graph) { exec_ = exec; graph_ = graph; }
  void Config(int func_id, int ms) {
    function_id_ = func_id;
    interval_ms_ = ms;

    LogMessage::min_log_level_ = INFO;
  }

private:
  void ViewNode() {
    std::ostringstream stream;
    for (int i = 0; i < graph_->nodes().size(); i++) {
      Node *n = &(*(graph_->nodes()[i]));
      stream << "(" << n->name().c_str() << ":";
      stream << " " << n->num_cached_buf(0) << " ";
      for (int si = 1; si < n->num_successors(); si++) {
        stream << " " << n->num_cached_buf(si) << " ";
      }
      stream << "<" << n->num_empty_buf() << ">";
      stream << ", " << n->run_count() << ")";
    }
    LOG(INFO) << stream.str();
  }

  void ViewStatus() {
    std::vector<Executor::Status*> list = exec_->status_list_;
    for (int i = 0; i < list.size(); i++) {
      if (list[i] == nullptr || list[i]->num_incomplete_out_nodes == 0) {
        continue;
      }
      else {
        printf("<%d, %d>", i, list[i]->num_incomplete_out_nodes);
      }
    }
    printf("\n");
  }

  void Entry() {
    while (!IsMustStop()) {
      switch (function_id_) {
      case 0:
        ViewNode();
        break;
      case 1:
        ViewStatus();
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
  }

private:
  Executor *exec_;
  Graph *graph_;

  int function_id_;
  int interval_ms_;
};

}  // namespace hcs.

#endif // HCS_PROFILER_H_