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
  void Run(std::vector<Blob *> &inputs);

  int GetDependsOutSize(int depend_id);

  Blob *BorrowDependsOut(int depend_id);
  bool RecycleDependsOut(int depend_id, Blob *out);

  bool PushOutput(Blob *out);
  bool PopOutput(Blob *out, int branch_id);

public:
  std::condition_variable cond_;
  mutable std::mutex mutex_;

  Work work_;
  std::atomic<int> atomic_run_count_;
  std::vector<Node*> successors_;
  std::vector<Node*> dependents_;
  std::vector<int> depends_branch_id_;

  std::vector<Blob *> outs_;
  BlockingQueue<Blob *> outs_free_;
  BlockingQueue<Blob *> *outs_full_;

private:
  std::string name_;  
};

void Node::Init(int buffer_queue_size) {
  // Initialize the buffer blobs.
  for (int i = 0; i < buffer_queue_size; i++) {
    std::string blob_name = name_ + "-" + std::to_string(i);
    Blob *p = new Blob(blob_name);
    if (p != nullptr) {
      p->set_node_name(name_);
      outs_.push_back(p);
      outs_free_.push(p);
    }
  }

  // Multiple copies for multiple successors.
  int len = 1;
  if (successors_.size() >= 2) {
    len = successors_.size();
  }
  outs_full_ = new BlockingQueue<Blob *>[len];
  if (outs_full_ == nullptr) {
    printf("Error: Failed to new for outs_full_.\n");
  }

  // Set the branch id for the case of multiple inputs.
  //   When the dependent node has multiple outputs, multiple copies of the 
  // output will be saved in outs_full_, and bound to the successors by index.
  // So we need to find the index and fetch the output in the corresponding branch.
  depends_branch_id_.resize(dependents_.size());
  for (int i = 0; i < depends_branch_id_.size(); i++) {
    depends_branch_id_[i] = -1;
  }
  for (int i = 0; i < dependents_.size(); i++) {
    for (int si = 0; si < dependents_[i]->num_successors(); si++) {
      if (dependents_[i]->successors_[si] == this) {
        depends_branch_id_[i] = si;
        break;
      }
    }
  }
  // Check.
  for (int i = 0; i < depends_branch_id_.size(); i++) {
    if (depends_branch_id_[i] == -1) {
      printf("Error: Failed to initialize depends_branch_id_.\n");
    }
  }
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

void Node::Run(std::vector<Blob *> &inputs) {

  // Prepare output.
  Blob *p = nullptr;
  if (false == outs_free_.try_pop(&p)) {
    printf("Error: %s-Failed to outs_free_.try_pop.\n", name_.c_str());
  }

  // Run.
  work_(inputs, p);
  atomic_run_count_++;

  // Push output.
  outs_full_[0].push(p);
  Blob *p2;
  for (int i = 1; i < successors_.size(); i++) {
    if (false == outs_free_.try_pop(&p2)) {
      printf("Error: %s-Failed to outs_free_.try_pop.\n", name_.c_str());
    }
    p->CloneTo(p2);
    outs_full_[i].push(p2);
  }
}

int Node::GetDependsOutSize(int id) {
  return dependents_[id]->outs_full_[depends_branch_id_[id]].size();
}

Blob* Node::BorrowDependsOut(int id) {
  // node->dependents(i)->outs_full_[node->depends_branch_id_[i]].size()
  // node->dependents_[i]->BorrowOut(node->depends_branch_id_[i]);
  Node *depends = dependents_[id];
  Blob *out = nullptr;
  if (false == depends->outs_full_[depends_branch_id_[id]].try_pop(&out)) {
    printf("<%d-%s>BorrowOut outs_full_: No element can be pop.", std::this_thread::get_id(), name_.c_str());
    return nullptr;
  }
  return out;
}

bool Node::RecycleDependsOut(int id, Blob *out) {
  //node->dependents_[i]->RecycleOut(inputs[i]);
  dependents_[id]->outs_free_.push(out);
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
// TODO: update.
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