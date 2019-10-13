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
#include "util/timer.hpp"

namespace hcs {

// The executor class to run a graph.
class Executor {
  // Just for debugging.
  friend class Profiler;

  // A Run() corresponds to a Promise.
  struct Promise {
    std::promise<void> promise;
  };
  struct Status {
    // The output node that this Run should invoke.
    int num_incomplete_out_nodes;
    Promise *p = nullptr;
    Timer timer;

    ~Status() {
      if (p != nullptr) {
        delete p;
        p = nullptr;
      }
    }
  };

public:

  explicit Executor(std::string name) :
    name_{name},
    graph_{ nullptr },
    run_count_{ 0 },
    finish_count_{ 0 },
    task_assistor_{ nullptr } {}

  ~Executor() {
    // Clear status.
    for (int i = 0; i < status_list_.size(); i++) {
      delete status_list_[i];
    }
    status_list_.clear();
    // Clear Timers.
    for (int i = 0; i < node_timers_.size(); i++) {
      delete node_timers_[i];
    }
    node_timers_.clear();
    // Shut down this scheduler
    done_ = true;
    // Nodtify and join
    NotifyAll();
    for (auto& t : threads_) {
      t.join();
    }
  }

  // Set some important parameters and do some initialization.
  void Bind(Graph *g, ExecutorMode mode, TaskAssistor *task_assistor);
  // Run.
  std::future<void> Run();
  // Notify all of the threads.
  void NotifyAll();

private:
  ///////////////
  // Serial.
  ///////////////
  // Generate serial_nodes_ for serial execution.
  void SerialFreeze();
  // Serial execute according to serial_nodes_.
  void SerialExec();

  ///////////////
  // Parallel. 
  ///////////////
  // Start all of the threads and waiting for input data.
  // A thread is bound to a node
  void Spawn();
  // Wait and check whether the input data is ready or not.
  bool WaitCheckInputs(Node *node);
  // Wake up subsequent nodes to tell them that they have new input data.
  void NotifySuccessors(Node *node);
  // Check to see if the Run() has completed, and if it has, respond the promise in Run().
  bool CheckStop(Node* node);

private:
  std::string name_;
  Graph *graph_;
  // Bring configurable parameters to Task.
  // Memory is controlled from the outside and set via Bind().
  TaskAssistor *task_assistor_; 

  // Record the state of each Run.
  std::vector<Status*> status_list_;
  // The number of times Run() has been called.
  std::atomic<int> run_count_;
  // The number of times the Run() completes.
  std::atomic<int> finish_count_;

  std::mutex mutex_;
  // A thread corresponds to a task node.
  std::vector<std::thread> threads_;
  // A flag bit used to indicate the end of work, and stop the working threads.
  std::atomic<bool> done_{ 0 };
  
  ExecutorMode mode_;
  // For serial mode only. 
  // The order of nodes in serial_nodes_ is the order in which they are called.
  std::vector<Node *> serial_nodes_; 
  // For each node, it marks the elapsed time of the node operation.
  std::vector<Timer *> node_timers_;

  static bool lock2serial_;
};

bool Executor::lock2serial_ = false;

void Executor::SerialFreeze() {
  serial_nodes_.clear();

  // flag_ in here is used to record whether a node has been added to serial_nodes_.
  for (int i = 0; i < graph_->nodes().size(); i++) {
    graph_->nodes()[i]->flag_ = 0;
  }

  // BFS.
  std::queue<Node *> queue;
  std::vector<Node *> input_nodes = graph_->GetInputNodes();
  for (int i = 0; i < input_nodes.size(); i++) {
    input_nodes[i]->flag_ = 2;
    queue.push(input_nodes[i]);
  }
  while (!queue.empty()) {
    Node *front = queue.front();

    // We need to ensure that all the leading nodes of this node 
    // have been added to serial_nodes_.
    bool all_depent_flag = true;
    for (int di = 0; di < front->num_dependents(); di++) {
      bool one_depent_flag = false;
      for (int j = 0; j < serial_nodes_.size(); j++) {
        if (serial_nodes_[j] == front->dependents(di)) {
          one_depent_flag = true;
          break;
        }
      }
      // If not, put the node back at the end of the queue and 
      // wait for the next retrieval.
      if (one_depent_flag == false) {
        all_depent_flag = false;
        queue.pop();
        queue.push(front);
        break;
      }
    }
    if (all_depent_flag) {
      serial_nodes_.push_back(front);
      queue.pop();
      for (int si = 0; si < front->num_successors(); si++) {
        // Skip nodes that have been placed.
        if (front->successor(si)->flag_ == 0) {
          front->successor(si)->flag_ = 1;
          queue.push(front->successor(si));
        }
      }
    }
  }

  // Skip input nodes.
  for (auto it = serial_nodes_.begin(); it != serial_nodes_.end(); it++) {
    if ((*it)->flag_ == 2)
      it = serial_nodes_.erase(it);
  }

  // Shows the node order of serial execution.
  std::ostringstream stream;
  stream << "Serial: ";
  for (int i = 0; i < serial_nodes_.size(); i++) {
    stream << serial_nodes_[i]->name().c_str() << ", ";
  }
  LOG(INFO) << stream.str();
}

void Executor::SerialExec() {
  if (node_timers_.size() != serial_nodes_.size()) {
    node_timers_.clear();
    for (int i = 0; i < serial_nodes_.size(); i++) {
      Timer *timer = new Timer(serial_nodes_[i]->name());
      node_timers_.push_back(timer);
    }
  }

  bool stop = false;
  while (!stop) {
    for (int n = 0; n < serial_nodes_.size(); n++) {
      Node *node = serial_nodes_[n];

      std::vector<Blob *> inputs; 
      Blob *output = nullptr;
      {
        node->BorrowInputs(inputs); 
        node->PrepareOutput(&output);

        TIME_DIFF_RECORD((*node_timers_[n]), node->Run(task_assistor_, inputs, output););

        node->RecycleInputs(inputs);
        node->PushOutput(output);
      }
      stop = CheckStop(node);
    }
  }
}

void Executor::Spawn() {

  std::vector<std::unique_ptr<Node>> &nodes = graph_->nodes();
  for (unsigned i = 0; i < nodes.size(); ++i) {
    Node *node = &(*nodes[i]);
    if (node->num_dependents() == 0) {
      continue;
    }
    LOG(INFO) << "Spawn " << node->name().c_str();
    if (!(node->has_task())) {
      LOG(ERROR) << "No task can be invoked in " << node->name().c_str();
      continue;
    }

    threads_.emplace_back([this, i, node]() -> void {

      TaskAssistor::ThreadVar &tv = task_assistor_->thread_var();
      tv.id = i;

      Timer *timer = new Timer(node->name());
      mutex_.lock();
      node_timers_.push_back(timer);
      mutex_.unlock();

      while (!done_) {
        // Wait and check intputs from depends.
        bool is_pass = WaitCheckInputs(node);
        if (!is_pass) { break; }

        std::vector<Blob *> inputs;
        Blob *output = nullptr;

        { // Borrow input & Prepare output.
          std::unique_lock<std::mutex> locker(mutex_);
          node->BorrowInputs(inputs);
          node->PrepareOutput(&output);
        }

        // Run.
        if (!lock2serial_) {
          TIME_DIFF_RECORD((*timer), node->Run(task_assistor_, inputs, output););
        }
        else {
          std::unique_lock<std::mutex> locker(mutex_);
          TIME_DIFF_RECORD((*timer), node->Run(task_assistor_, inputs, output););
        }

        { // Recycle inputs & Push output.
          std::unique_lock<std::mutex> locker(mutex_);
          node->RecycleInputs(inputs);
          node->PushOutput(output);
        }

        NotifySuccessors(node);

        CheckStop(node);
      }
    });
  }
}

bool Executor::WaitCheckInputs(Node *node) {

  std::unique_lock<std::mutex> locker(node->mutex_);
  bool is_ready = false;
  while (!is_ready) {
    is_ready = true;
    for (int i = 0; i < node->num_dependents(); ++i) {
      if (node->IsDependsOutEmpty(i)) {
        is_ready = false;
        break;
      }
    }

    if (!is_ready) {
      node->cond_.wait(locker);
      if (done_) {
        locker.unlock();
        return false;
      }
      LOG(INFO) << node->name().c_str() << "-waiting";
    }
  }
  return true;
}

void Executor::NotifySuccessors(Node *node) {

  const auto num_successors = node->num_successors();
 
  for (int i = 0; i < num_successors; ++i) {
    // Check that all dependent nodes have cached output data.
    Node *successor = node->successor(i);
    bool is_ready = true;
    for (int j = 0; j < successor->num_dependents(); ++j) {
      if (successor->IsDependsOutEmpty(j)) {
        is_ready = false;
        break;
      }
    }
    // If the dependents are ready, wake up it.
    // Note: mutex_ in node is a member variable of node. 
    // So std::unique_lock<std::mutex> locker(mutex_) should only be use for 
    // std::condition_variable to prevent multiple locks from being invoked 
    // at the same time and causing unawakes.
    if (is_ready)
      successor->cond_.notify_one();
  }
}

bool Executor::CheckStop(Node *node) {
  const auto num_successors = node->num_successors();

  int status_id = finish_count_.load() % status_list_.size();
  Status *status = status_list_[status_id];
  // A node without any successor should check the termination of this run.
  if (num_successors == 0) {
    if (--(status->num_incomplete_out_nodes) == 0) {
      // It means that all of the output nodes have been completed.
      LOG(INFO) << "Stop " << status_id;
      finish_count_++;
      status_list_[status_id]->timer.Stop();

      auto p{ std::move(status_list_[status_id]->p->promise) };

      // Delete Promise & Recover status.
      delete status_list_[status_id]->p;
      status_list_[status_id]->p = new Promise;
      status_list_[status_id]->num_incomplete_out_nodes = 0;

      // We set the promise in the end to response the std::future in Run().
      p.set_value();

      return true;
    }
  }
  return false;
}

void Executor::Bind(Graph *g, ExecutorMode mode, TaskAssistor *task_assistor) {
  graph_ = g;
  mode_ = mode;
  task_assistor_ = task_assistor;

  for (int i = 0; i < g->buffer_queue_size(); i++) {
    Status *stat = new Status;
    stat->p = new Promise;
    stat->num_incomplete_out_nodes = 0;

    status_list_.push_back(stat);
  }

  switch (mode) {
  case PARALLEL_MULTI_STREAMS:
    task_assistor->Init4GPU(g->nodes().size());
  case PARALLEL:
    Spawn();
    break;
  case SERIAL:
    SerialFreeze();
    break;
  default:
    LOG(ERROR) << "mode " << mode << "is not supported.";
  }
}

std::future<void> Executor::Run() {

  std::vector<Node*> input_nodes = graph_->GetInputNodes();
  if (input_nodes.size() <= 0) {
    LOG(ERROR) << "A graph needs at least one input node";
  }
  // Check the dimensions of each input node.
  if (input_nodes.size() > 1) {
    for (int i = 1; i < input_nodes.size(); i++) {
      if (input_nodes[i]->num_cached_buf(0)
        != input_nodes[i - 1]->num_cached_buf(0)) {
        LOG(ERROR) << "The number of input data of each input node should be the same";
      }
    }
  }

  Status *stat = status_list_[run_count_.load() % status_list_.size()];
  if (stat->num_incomplete_out_nodes != 0) {
    LOG(ERROR) << "The Status you chose is busy, please wait.";
  }
  stat->timer.Start();
  run_count_++;
  // TODO: Enable batch size.
  // Set the number of output nodes for this Run.
  stat->num_incomplete_out_nodes 
    = input_nodes[0]->num_cached_buf(0)
    * graph_->GetOutputNodes().size();

  std::future<void> future = stat->p->promise.get_future();

  if (mode_ == PARALLEL || mode_ == PARALLEL_MULTI_STREAMS) {
    // Notify each input nodes.
    for (auto node : input_nodes) {
      for (size_t i = 0; i < node->num_successors(); ++i) {
        node->successor(i)->cond_.notify_one();
      }
    }
  }
  else if (mode_ == SERIAL) {
    SerialExec();
  }
  else {
    LOG(ERROR) << "mode " << mode_ << "is not supported.";
  }

  return future;
}

void Executor::NotifyAll() {
  for (auto &node : graph_->nodes()) {
    node->cond_.notify_all();
  }
}

}  // end of namespace hcs
#endif // HCS_EXECUTOR_H_