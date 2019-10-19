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
  void Config(int mode, int ms) {
    mode_ = mode;
    interval_ms_ = ms;

    if (mode != 0)
      LogMessage::min_log_level_ = INFO;
  }
  ~Profiler() { 
    LogMessage::min_log_level_ = WARNING; 
    Timer::is_record_ = false;
  }

private:
  // Show the information of each node.
  // var - cached output for branch1, cached output for branch2..
  // (node name: var <Free cache space>, How many times has the node run)
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
    LOG(INFO) << "Profiler->Node: " << stream.str();
  }

  // Show the progress of Run().
  void ViewStatus() {
    bool flag = false;
    std::ostringstream stream;
    std::vector<Executor::Status*> list = exec_->status_list_;
    for (int i = 0; i < list.size(); i++) {
      if (list[i] == nullptr || list[i]->num_incomplete_out_nodes == 0) {
        continue;
      }
      else {
        flag = true;
        stream << "<" << i << ", " << list[i]->num_incomplete_out_nodes << ">";
      }
    }
    if (flag)
      LOG(INFO) << stream.str();
    else
      LOG(INFO) << "Profiler->Status: There's no active Status";
  }

  // Show the running time of task in each node.
  // a - node name.  b - How many times has the node run.
  // c - The minimum time consuming.   
  // d - The maximum time consuming.  
  // e - average.
  // (a: count-b, min-c, max-d, ave-e)
  void ViewNodeRunTime() {
    Timer::is_record_ = true;
    std::ostringstream stream;
    for (int i = 0; i < exec_->node_timers_.size(); i++) {
      Timer *timer = exec_->node_timers_[i];
      stream << "(" << timer->name().c_str() << ": ";
      stream << "count-" << timer->count() << ", ";
      stream << "min-" << timer->min() << ", ";
      stream << "max-" << timer->max() << ", ";
      stream << "ave-" << timer->ave() << "). ";
    }
    LOG(INFO) << "Profiler->NodeRunTime: " << stream.str();
  }

  // Show how long it takes to run() in an executor.
  void ViewStatusRunTime() {
    Timer::is_record_ = true;
    std::ostringstream stream;
    std::vector<Executor::Status*> list = exec_->status_list_;
    for (int i = 0; i < list.size(); i++) {
      Timer *timer = &(list[i]->timer);
      if (timer->count() == 0)
        continue;
      stream << "(" << i << ": ";
      stream << "count-" << timer->count() << ", ";
      stream << "ave-" << timer->ave() << "). ";
    }
    LOG(INFO) << "Profiler->StatusRunTime: " << stream.str();
  }

  void Entry() {
    while (!IsMustStop()) {
      // Note: "==" has a higher priority than "&".
      if ((mode_ & VIEW_NODE) == VIEW_NODE) {
        ViewNode();
      }
      if ((mode_ & VIEW_STATUS) == VIEW_STATUS) {
        ViewStatus();
      }
      if ((mode_ & VIEW_NODE_RUN_TIME) == VIEW_NODE_RUN_TIME) {
        ViewNodeRunTime();
      }
      if ((mode_ & VIEW_STATUS_RUN_TIME) == VIEW_STATUS_RUN_TIME) {
        ViewStatusRunTime();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
  }

private:
  Executor *exec_;
  Graph *graph_;

  int mode_;
  // Query interval time.
  int interval_ms_;
};

}  // namespace hcs.

#endif // HCS_PROFILER_H_