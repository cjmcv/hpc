#ifndef HCS_EXECUTOR_H_
#define HCS_EXECUTOR_H_

#include <iostream>
#include <random>
#include <atomic>
#include <optional>
#include <thread>
#include <cassert>
#include <future>
#include <map>

#include "util/spmc_queue.hpp"
#include "util/notifier.hpp"
#include "graph.hpp"

namespace hcs {

// The executor class to run a graph.
class Executor {
  // Just for debugging.
  friend class Profiler;

  struct Worker {
    std::mt19937 rdgen{ std::random_device{}() };
    WorkStealingQueue<Node*> queue;
  };
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
    int *depends = nullptr;
    Promise *p = nullptr;

    ~Status() {
      if (p != nullptr) {
        delete p;
        p = nullptr;
      }
      if (depends != nullptr) {
        delete depends;
        depends = nullptr;
      }
    }
    void CopyTo(Status *s) {
      s->num_incomplete_out_nodes = this->num_incomplete_out_nodes;
      s->num_nodes = this->num_nodes;
      if (s->depends == nullptr)
        s->depends = new int[s->num_nodes];
      memcpy(s->depends, this->depends, sizeof(int) * s->num_nodes);    
      if (s->p == nullptr)
        s->p = new Promise;
    }
  };

public:
  std::string name_;

  // constructs the executor with N worker threads
  explicit Executor(unsigned N = std::thread::hardware_concurrency()) :
    workers_{ N },
    waiters_{ N },
    notifier_{ waiters_ },
    base_status_{ nullptr }, 
    graph_{ nullptr },
    run_count_{ 0 } {

    Spawn(N);
  }
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
    notifier_.notify(true);
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

  unsigned FindVictim(unsigned);
  bool Wait4Tasks(unsigned, std::optional<Node*>&);
  void Spawn(unsigned);
  void ExploitTask(unsigned, std::optional<Node*>&);
  void ExploreTask(unsigned, std::optional<Node*>&);
  void Schedule(Node*);

  void PushSuccessors(Node *node);
  void Stop(int id);

private:
  Graph *graph_;
  Status *base_status_;
  std::vector<Status*> status_list_;
  int run_count_;

  std::vector<Worker> workers_;
  std::vector<Notifier::Waiter> waiters_;
  std::vector<std::thread> threads_;

  // This queue is prepared for master thread.
  WorkStealingQueue<Node*> queue_;  

  std::atomic<size_t> num_actives_{ 0 };
  std::atomic<size_t> num_thieves_{ 0 };
  std::atomic<bool> done_{ 0 };

  Notifier notifier_;
  std::mutex mutex_;
};

void Executor::Spawn(unsigned N) {

  // Lock to synchronize all workers before creating _worker_maps
  for (unsigned i = 0; i < N; ++i) {
    threads_.emplace_back([this, i]() -> void {

      PerThread& pt = per_thread();
      pt.is_worker = true;
      pt.worker_id = i;

      std::optional<Node*> t;

      // must use 1 as condition instead of !done
      while (1) {
        // If this thread has a task in t, execute it.
        ExploitTask(i, t);
        // After finishing the current task, try to steal one.
        ExploreTask(i, t);
        // If it can not steal any task, just wait for its.
        if (!t) {
          if (Wait4Tasks(i, t) == false) {
            break;
          }
        }
      }
    });
  }
}

unsigned Executor::FindVictim(unsigned thief) {
  // try to look for a task from other workers
  for (unsigned vtm = 0; vtm < workers_.size(); ++vtm) {
    if ((thief == vtm && !queue_.empty()) ||
      (thief != vtm && !workers_[vtm].queue.empty())) {
      return vtm;
    }
  }

  return workers_.size();
}

void Executor::ExploreTask(unsigned thief, std::optional<Node*>& t) {

  assert(!t);

  const unsigned l = 0;
  const unsigned r = workers_.size() - 1;

  const size_t F = (workers_.size() + 1) << 1;
  const size_t Y = 100;

steal_loop:

  size_t f = 0;
  size_t y = 0;

  ++num_thieves_;

  // explore
  while (!done_) {

    unsigned vtm = std::uniform_int_distribution<unsigned>{ l, r }(
      workers_[thief].rdgen
      );
    t = (vtm == thief) ? queue_.steal() : workers_[vtm].queue.steal();
    if (t) { break; }

    if (f++ > F) {
      if (std::this_thread::yield(); y++ > Y) {
        break;
      }
    }
  }

  // We need to ensure at least one thieve if there is an
  // active worker
  if (auto N = --num_thieves_; N == 0) {
    if (t != std::nullopt) {
      notifier_.notify(false);
      return;
    }
    else if (num_actives_ > 0) {
      goto steal_loop;
    }
  }
}

void Executor::ExploitTask(unsigned i, std::optional<Node*>& t) {
  if (t) {
    auto& worker = workers_[i];

    if (++num_actives_; num_thieves_ == 0) {
      notifier_.notify(false);
    }

    do {
      Blob *p = nullptr;
      Node *n = *t;
      // Locked by node.
      // That means the same node can only be invoked by one thread at a time.
      n->lock(); 
      auto &f = n->work_;
      if (f != nullptr) {
        if (n->num_successors() <= 1) {
          if (false == n->outs_free_.try_pop(&p)) {
            printf("Failed to outs_free_.try_pop.\n");
          }
          f(n->dependents_, p);
          n->outs_full_.push(p);
        }
        else {
          // If there are multiple successors, copy the output for each one.
          if (false == n->outs_free_.try_pop(&p)) {
            printf("Failed to outs_free_.try_pop.\n");
          }
          f(n->dependents_, p);
          
          n->outs_branch_full_[0].push(p);
          Blob *p2;
          for (int i = 1; i < n->num_successors(); i++) {
            if (false == n->outs_free_.try_pop(&p2)) {
              printf("Failed to outs_free_.try_pop.\n");
            }
            p->CloneTo(p2);
            n->outs_branch_full_[i].push(p2);
          }
        }
      }
      PushSuccessors(n);
      n->atomic_run_count_++;
      n->unlock();

      t = worker.queue.pop();
    } while (t);

    --num_actives_;
  }
}

bool Executor::Wait4Tasks(unsigned me, std::optional<Node*>& t) {

  assert(!t);

  notifier_.prepare_wait(&waiters_[me]);

  if (auto vtm = FindVictim(me); vtm != workers_.size()) {
    notifier_.cancel_wait(&waiters_[me]);
    t = (vtm == me) ? queue_.steal() : workers_[vtm].queue.steal();
    return true;
  }

  if (done_) {
    notifier_.cancel_wait(&waiters_[me]);
    notifier_.notify(true);
    return false;
  }

  // Now I really need to relinguish my self to others
  notifier_.commit_wait(&waiters_[me]);

  return true;
}

void Executor::Schedule(Node* node) {

  // no worker thread available
  if (workers_.size() == 0) {
    queue_.push(node);
    return;
  }

  if (auto& pt = per_thread(); pt.is_worker) {
    // caller is a worker
    workers_[pt.worker_id].queue.push(node);
    return;
  }
  else {
    // master threads
    queue_.push(node);
  }

  notifier_.notify(false);
}

void Executor::PushSuccessors(Node* node) {

  int status_id = node->atomic_run_count_.load() % status_list_.size();
  Status *status = status_list_[status_id];
  //printf("push (%s, %d).\n", node->name().c_str(), status_id);

  const auto num_successors = node->num_successors();
  for (size_t i = 0; i < num_successors; ++i) {
    int depends = --(status->depends[node->successor(i)->id()]);
    if (depends == 0) {
      Schedule(node->successor(i));
      //printf("S_%s.\n", node->successor(i)->name().c_str());
    }
  }

  // A node without any successor should check the termination of this run.
  if (num_successors == 0) {
    if (--(status->num_incomplete_out_nodes) == 0) {
      // It means that all of the output nodes have been completed.
      if (workers_.size() > 0) {   
        Stop(status_id);   // Finishing this Run.
      }
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
  base_status_->depends = new int[g->nodes().size()];

  for (auto& node : g->nodes()) {
    base_status_->depends[node->id()] = node->num_dependents();
  }

  for (int i = 0; i < g->buffer_queue_size(); i++) {
    Status *stat = new Status;
    base_status_->CopyTo(stat);
    status_list_.push_back(stat);
  }

  graph_ = g;
}

std::future<void> Executor::Run() {

  Status *stat = status_list_[run_count_ % status_list_.size()];
  run_count_++;

  std::vector<Node*> input_nodes = graph_->GetInputNodes();
  for (auto node : input_nodes) {
    queue_.push(node);
    notifier_.notify(false);
  }

  std::future<void> future = stat->p->promise.get_future();
  return future;
}

}  // end of namespace hcs
#endif // HCS_EXECUTOR_H_