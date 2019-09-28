#ifndef HCS_NODE_H_
#define HCS_NODE_H_

#include <atomic>
#include <functional>
#include <string>
#include <condition_variable>

#include "blob.hpp"
#include "util/blocking_queue.hpp"

namespace hcs {

// Forward declaration
class Graph;

// Class: Node
class Node {

  using Work = std::function<void(std::vector<Blob *> inputs, Blob *output)>;

public:
  Node() : atomic_run_count_(0), outs_full_(nullptr), name_("noname") {}
  Node(Work &&c) : work_(c), atomic_run_count_(0), outs_full_(nullptr), name_("noname") {}
  ~Node() {}

  // Querys.
  inline size_t num_successors() const { return successors_.size(); }
  inline size_t num_dependents() const { return dependents_.size(); }
  inline Node *successor(int id) { return successors_[id]; }
  inline Node *dependents(int id) { return dependents_[id]; }
  inline const std::string &name() const { return name_; }

  inline Node *name(const std::string& name) { name_ = name; return this; }

  inline void precede(Node &v) {
    successors_.push_back(&v);
    v.dependents_.push_back(this);
  }

  template <typename... Ns>
  inline void precede(Ns&&... node) {
    (precede(*(node)), ...);
  }

  void Init(int buffer_queue_size);
  void Clean();
  void Run();

  Blob *BorrowOut(int branch_id);
  bool RecycleOut(Blob *out);

  bool PushOutput(Blob *out);
  bool PopOutput(Blob *out, int branch_id);

public:
  std::condition_variable cond_;

  Work work_;
  std::atomic<int> atomic_run_count_;
  std::vector<Node*> successors_;
  std::vector<Node*> dependents_;

  std::vector<Blob *> outs_;
  BlockingQueue<Blob *> outs_free_;
  BlockingQueue<Blob *> *outs_full_;

private:
  std::string name_;  
  mutable std::mutex mutex_;
};

void Node::Init(int buffer_queue_size) {
  for (int i = 0; i < buffer_queue_size; i++) {
    std::string blob_name = name_ + "-" + std::to_string(i);
    Blob *p = new Blob(blob_name);
    if (p != nullptr) {
      p->set_node_name(name_);
      outs_.push_back(p);
      outs_free_.push(p);
    }
  }

  int len = 1;
  if (successors_.size() >= 2) {
    len = successors_.size();
  }
  outs_full_ = new BlockingQueue<Blob *>[len];
}

void Node::Clean() {
  if (outs_.size() > 0) {
    for (int i = 0; i < outs_.size(); i++) {
      if (outs_[i] != nullptr) {
        delete outs_[i];
        outs_[i] = nullptr;
      }
    }
  }
  if (outs_full_ != nullptr) {
    delete[]outs_full_;
  }
}

void Node::Run() {
  std::unique_lock<std::mutex> locker(mutex_);

  Blob *p = nullptr;
  if (work_ != nullptr) {

    // Fetch input.
    std::vector<Blob *> inputs;
    for (int i = 0; i < dependents_.size(); i++) {
      // Check the case of multiple inputs.
      //   When the dependent node has multiple outputs, multiple copies of the 
      // output will be saved in outs_full_, and bound to the successors by index.
      // So we need to find the index and fetch the output in the corresponding branch.
      int branch_id = -1;
      for (int si = 0; si < dependents_[i]->num_successors(); si++) {
        if (dependents_[i]->successors_[si] == this) {
          branch_id = si;
          break;
        }
      }
      if (branch_id == -1) {
        printf("Error: Can not match branch id.\n");
      }

      // If the output from the dependent node cannot be extracted, wait.
      Blob *in = dependents_[i]->BorrowOut(branch_id);
      while (in == nullptr) {
        cond_.wait(locker);

        in = dependents_[i]->BorrowOut(branch_id);
        printf("Info: %s-waiting.\n", name_.c_str());
      }
      inputs.push_back(in);
    }
    // Prepare output.
    if (false == outs_free_.try_pop(&p)) {
      printf("Error: %s-Failed to outs_free_.try_pop.\n", name_.c_str());
    }

    // Run.
    work_(inputs, p);
    atomic_run_count_++;

    // Push to output.
    outs_full_[0].push(p);
    Blob *p2;
    for (int i = 1; i < successors_.size(); i++) {
      if (false == outs_free_.try_pop(&p2)) {
        printf("Error: %s-Failed to outs_free_.try_pop.\n", name_.c_str());
      }
      p->CloneTo(p2);
      outs_full_[i].push(p2);
    }

    // Recycle.
    for (int i = 0; i < dependents_.size(); i++) {
      dependents_[i]->RecycleOut(inputs[i]);
    }
  }
  else {
    printf("Error: No work can be invoked in %s.\n", name_.c_str());
  }
}

Blob* Node::BorrowOut(int branch_id) {
  if (branch_id > successors_.size()) {
    printf("Error: BorrowOut %s -> branch_id > outs_full_.size()", name_.c_str());
  }
  Blob *out = nullptr;
  if (false == outs_full_[branch_id].try_pop(&out)) {
    printf("<%d-%s>BorrowOut outs_full_: No element can be pop.", std::this_thread::get_id(), name_.c_str());
    return nullptr;
  }
  return out;
}

bool Node::RecycleOut(Blob *out) {
  outs_free_.push(out);
  return true;
}

bool Node::PushOutput(Blob *out) {
  Blob *inside_out;
  for (int i = 0; i < successors_.size(); i++) {
    if (false == outs_free_.try_pop(&inside_out)) {
      printf("<%d>PushOutput: failed..", std::this_thread::get_id());
      return false;
    }
    out->CloneTo(inside_out);
    outs_full_[i].push(inside_out);
  }
  return true;
}

bool Node::PopOutput(Blob *out, int branch_id) {
  if (branch_id > successors_.size()) {
    printf("Error: PopOutput %s -> branch_id > outs_full_.size()", name_.c_str());
  }
  Blob *inside_out;
  if (false == outs_full_[branch_id].try_pop(&inside_out)) {
    printf("<%d-%s>PopOutput outs_full_: No element can be pop.", std::this_thread::get_id(), name_.c_str());
    return false;
  }

  inside_out->CloneTo(out);
  outs_free_.push(inside_out);
  return true;
}

}  // end of namespace hcs.

#endif //HCS_NODE_H_