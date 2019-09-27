#ifndef HCS_EXECUTOR_H_
#define HCS_EXECUTOR_H_

#include <iostream>
#include <random>
#include <atomic>
#include <thread>
#include <cassert>
#include <future>
#include <map>

#include "graph.hpp"

namespace hcs {

// The executor class to run a graph.
class Executor {
  // Just for debugging.
  friend class Profiler;

  struct PerThread {
    bool is_worker = false;
    int worker_id = -1;
  };

  struct Promise {
    std::promise<void> promise;
  };
  struct Status { 
    int num_incomplete_out_nodes;
    int num_nodes;
    Promise *p = nullptr;

    ~Status() {
      if (p != nullptr) {
        delete p;
        p = nullptr;
      }
    }
    void CopyTo(Status *s) {
      s->num_incomplete_out_nodes = this->num_incomplete_out_nodes;
      s->num_nodes = this->num_nodes;  
      if (s->p == nullptr)
        s->p = new Promise;
    }
  };

public:
  std::string name_;

  // constructs the executor with N worker threads
  explicit Executor(unsigned N = std::thread::hardware_concurrency()) :
    base_status_{ nullptr }, 
    graph_{ nullptr },
    run_count_{ 0 } {}

  ~Executor() {
    if (base_status_ != nullptr) {
      delete base_status_;
      base_status_ = nullptr;
    }
    if (status_list_.size() > 0) {
      for (int i = 0; i < status_list_.size(); i++) {
        delete status_list_[i];
      }
      status_list_.clear();
    }
    // shut down the scheduler
    done_ = true;
    for (auto& t : threads_) {
      t.join();
    }
  }

  // runs the taskflow once
  // return a std::future to access the execution status.
  std::future<void> Run();
  void Bind(Graph *g);

private:  
  PerThread& per_thread() const {
    thread_local PerThread pt;
    return pt;
  }

  void Spawn(std::vector<std::unique_ptr<Node>> &nodes);

  void PushSuccessors(Node *node);
  void Stop(int id);

private:
  Graph *graph_;
  Status *base_status_;
  std::vector<Status*> status_list_;
  int run_count_;

  std::vector<std::thread> threads_;

  std::atomic<size_t> num_actives_{ 0 };
  std::atomic<size_t> num_thieves_{ 0 };
  std::atomic<bool> done_{ 0 };

  std::mutex mutex_;
};

void Executor::Spawn(std::vector<std::unique_ptr<Node>> &nodes) {

  // Lock to synchronize all workers before creating _worker_maps
  for (unsigned i = 0; i < nodes.size(); ++i) {
    Node *node = &(*nodes[i]);
    if (node->num_dependents() == 0) {
      continue;
    }

    threads_.emplace_back([this, i, node]() -> void {

      while (1) {
        node->Run();
        PushSuccessors(node);
        // 在wait里卡住了，join不了，需要唤醒来判断是否done.
        //if (done_) {
        //  break;
        //}
      }
    });
  }
}

void Executor::PushSuccessors(Node* node) {

  const auto num_successors = node->num_successors();
  for (int i = 0; i < num_successors; ++i) {
    Node *successor = node->successor(i);

    for (int j = 0; j < successor->num_dependents(); ++j) {
      // TODO: 多输入的情况，brach.
      if (successor->dependents(j)->outs_full_.size() <= 0) {
        break;
      }
    }

    successor->cond_.notify_one();
    // 删除后，其他线程无法马上唤醒？
    // 因为BorrowOut等地方也用了std::unique_lock<std::mutex> locker(mutex_)，跟wait里调动该锁有冲突
    //std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
  }
  int status_id = (node->atomic_run_count_.load() - 1) % status_list_.size(); 
  Status *status = status_list_[status_id];
  // A node without any successor should check the termination of this run.
  if (num_successors == 0) {
    if (--(status->num_incomplete_out_nodes) == 0) {
      // It means that all of the output nodes have been completed.
      Stop(status_id);   // Finishing this Run.
    }
  }
}

void Executor::Stop(int id) {
  auto p{ std::move(status_list_[id]->p->promise) };

  delete status_list_[id]->p;
  status_list_[id]->p = nullptr;

  // Recover.
  base_status_->CopyTo(status_list_[id]);

  // We set the promise in the end to response the std::future in Run().
  p.set_value();
}

void Executor::Bind(Graph *g) {
  if (base_status_ != nullptr) {
    delete base_status_;
    base_status_ = nullptr;
  }

  base_status_ = new Status;
  base_status_->num_incomplete_out_nodes = g->GetOutputNodes().size();
  base_status_->num_nodes = g->nodes().size();

  for (int i = 0; i < g->buffer_queue_size(); i++) {
    Status *stat = new Status;
    base_status_->CopyTo(stat);
    status_list_.push_back(stat);
  }

  graph_ = g;

  Spawn(graph_->nodes());
}

std::future<void> Executor::Run() {
  Status *stat = status_list_[run_count_ % status_list_.size()];
  run_count_++;

  std::vector<Node*> input_nodes = graph_->GetInputNodes();
  for (auto node : input_nodes) {
    for (size_t i = 0; i < node->num_successors(); ++i) {
      node->successor(i)->cond_.notify_one();
    }
  }

  std::future<void> future = stat->p->promise.get_future();
  return future;
}

// TODO: 2. 多输入的节点有输出方案，但没有写对应输入的方式。

}  // end of namespace hcs
#endif // HCS_EXECUTOR_H_