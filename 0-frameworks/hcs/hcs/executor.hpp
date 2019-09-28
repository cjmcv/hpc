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

    // Nodtify and join
    printf("notify and join.\n");
    for (auto &node : graph_->nodes()) {
      node->cond_.notify_all();
    }
    for (auto& t : threads_) {
      t.join();
    }
  }

  void Bind(Graph *g);  
  // TODO: Match wait with all task.
  std::future<void> Run();
  void NotifyAll();

private:
  PerThread& per_thread() const {
    thread_local PerThread pt;
    return pt;
  }

  void Spawn(std::vector<std::unique_ptr<Node>> &nodes);

  bool WaitBorrowOut(Node *node, std::vector<Blob *> &inputs);

  void NotifySuccessors(Node *node);

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
    if (node->work_ == nullptr) {
      printf("Error: No work can be invoked in %s.\n", name_.c_str());
    }

    threads_.emplace_back([this, i, node]() -> void {

      while (!done_) {
        // Wait and borrow intputs from depends.
        std::vector<Blob *> inputs;
        bool is_pass = WaitBorrowOut(node, inputs);
        if (!is_pass) { break; }

        node->Run(inputs);

        // Recycle inputs.
        for (int i = 0; i < node->dependents_.size(); i++) {
          node->RecycleDependsOut(i, inputs[i]);
        }
        
        NotifySuccessors(node);
      }
    });
  }
}

bool Executor::WaitBorrowOut(Node *node, std::vector<Blob *> &inputs) {

  std::unique_lock<std::mutex> locker(node->mutex_);
  inputs.clear();

  for (int i = 0; i < node->dependents_.size(); i++) {
    // If the output from the dependent node cannot be extracted, wait.
    Blob *in = node->BorrowDependsOut(i);
    while (in == nullptr) {
      printf("Info: %s-waiting (%d).\n", node->name().c_str(), i);
      node->cond_.wait(locker);

      if (done_) return false;
      in = node->BorrowDependsOut(i);
    }
    inputs.push_back(in);
  }

  return true;
}

void Executor::NotifySuccessors(Node* node) {

  const auto num_successors = node->num_successors();
 
  mutex_.lock();
  for (int i = 0; i < num_successors; ++i) {
    // Check that all dependent nodes have cached output data.
    Node *successor = node->successor(i);
    bool is_ready = true;
    for (int j = 0; j < successor->num_dependents(); ++j) {
      if (successor->GetDependsOutSize(j) <= 0) {
        is_ready = false;
        break;
      }
    }

    // If the dependents are ready, wake up it.
    // Note: mutex_ in node is a member variable of node. 
    // So std::unique_lock<std::mutex> locker(mutex_) should only be use for 
    // std::condition_variable to prevent multiple locks from being invoked 
    // at the same time and causing unawakes
    printf("Notify %s.", successor->name().c_str());
    if (is_ready)
      successor->cond_.notify_one();
  }
  mutex_.unlock();

  int status_id = (node->atomic_run_count_.load() - 1) % status_list_.size(); 
  Status *status = status_list_[status_id];
  printf("Check %d.", status_id);
  // A node without any successor should check the termination of this run.
  if (num_successors == 0) {
    if (--(status->num_incomplete_out_nodes) == 0) {
      // It means that all of the output nodes have been completed.
      printf("Stop %d.", status_id);
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

void Executor::NotifyAll() {
  for (auto &node : graph_->nodes()) {
    node->cond_.notify_all();
  }
}

}  // end of namespace hcs
#endif // HCS_EXECUTOR_H_