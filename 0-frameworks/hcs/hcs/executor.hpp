#ifndef HCS_EXECUTOR_H_
#define HCS_EXECUTOR_H_

#include <iostream>
#include <random>
#include <atomic>
#include <optional>
#include <thread>
#include <cassert>
#include <future>

#include "util/spmc_queue.hpp"
#include "util/notifier.hpp"
#include "graph.hpp"

namespace hcs {

// The executor class to run a graph.
class Executor {
  
  struct Worker {
    std::mt19937 rdgen{ std::random_device{}() };
    WorkStealingQueue<Node*> queue;
  };
  struct PerThread {
    bool is_worker = false;
    int worker_id = -1;
  };
  struct Status {
    int num_incomplete_out_nodes;
    std::promise<void> promise;
  };

public:
  // constructs the executor with N worker threads
  explicit Executor(unsigned N = std::thread::hardware_concurrency()) :
    status_{ nullptr },
    workers_{ N },
    waiters_{ N },
    notifier_{ waiters_ } {
    Spawn(N);
  }
  ~Executor() {
    // shut down the scheduler
    done_ = true;
    notifier_.notify(true);
    for (auto& t : threads_) {
      t.join();
    }
  }

  // runs the taskflow once
  // return a std::future to access the execution status.
  std::future<void> Run(Graph& g);

  // queries the number of worker threads (can be zero)
  inline size_t num_workers() const { return workers_.size(); }

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

  void PushSuccessors(Node*);
  void Stop();

private:
  Status *status_;

  std::vector<Worker> workers_;
  std::vector<Notifier::Waiter> waiters_;
  std::vector<std::thread> threads_;

  WorkStealingQueue<Node*> queue_;  // The queue belngs to master thread.

  std::atomic<size_t> num_actives_{ 0 };
  std::atomic<size_t> num_thieves_{ 0 };
  std::atomic<bool> done_{ 0 };

  Notifier notifier_;
};

inline void Executor::Spawn(unsigned N) {

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

inline unsigned Executor::FindVictim(unsigned thief) {
  // try to look for a task from other workers
  for (unsigned vtm = 0; vtm < workers_.size(); ++vtm) {
    if ((thief == vtm && !queue_.empty()) ||
      (thief != vtm && !workers_[vtm].queue.empty())) {
      return vtm;
    }
  }

  return workers_.size();
}

inline void Executor::ExploreTask(unsigned thief, std::optional<Node*>& t) {

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

inline void Executor::ExploitTask(unsigned i, std::optional<Node*>& t) {
  if (t) {
    auto& worker = workers_[i];

    if (++num_actives_; num_thieves_ == 0) {
      notifier_.notify(false);
    }

    do {
      auto &f = (*t)->work_;
      if (f != nullptr)
        std::invoke(f, (*t)->dependents_, &((*t)->out_));

      PushSuccessors(*t);

      t = worker.queue.pop();
    } while (t);

    --num_actives_;
  }
}

inline bool Executor::Wait4Tasks(unsigned me, std::optional<Node*>& t) {

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

inline void Executor::Schedule(Node* node) {

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

inline void Executor::PushSuccessors(Node* node) {

  // Recover for the next Run().
  node->atomic_num_depends_ = static_cast<int>(node->num_dependents());

  const auto num_successors = node->num_successors();
  for (size_t i = 0; i < num_successors; ++i) {
    if (--(node->successor(i)->atomic_num_depends_) == 0) {
      Schedule(node->successor(i));
    }
  }

  // A node without any successor should check the termination of this run.
  if (num_successors == 0) {
    if (--(status_->num_incomplete_out_nodes) == 0) {
      // It means that all of the output nodes have been completed.
      if (workers_.size() > 0) {   
        Stop();   // Finishing this Run.
      }
    }
  }
}

inline void Executor::Stop() {
  auto p{ std::move(status_->promise) };

  delete status_;
  status_ = nullptr;

  // We set the promise in the end to response the std::future in Run().
  p.set_value();
}

std::future<void> Executor::Run(Graph& g) {

  status_ = new Status();
  status_->num_incomplete_out_nodes = g.GetOutputNodes().size();
  std::vector<Node*> input_nodes = g.GetInputNodes();

  if (workers_.size() == 0) {
    // Speical case of zero workers needs.
    for (auto node : input_nodes)
      queue_.push(node);

    auto node = queue_.pop();
    while (node) {
      auto &f = (*node)->work_;
      if (f != nullptr)
        std::invoke(f, (*node)->dependents_, &((*node)->out_));

      PushSuccessors(*node);
      node = queue_.unsync_pop();
    }
    return std::async(std::launch::deferred, []() {});
  }
  else {  
    for (auto node : input_nodes) {
      queue_.push(node);
      notifier_.notify(false);
    }

    std::future<void> future = status_->promise.get_future();
    return future;
  }
}

}  // end of namespace hcs
#endif // HCS_EXECUTOR_H_