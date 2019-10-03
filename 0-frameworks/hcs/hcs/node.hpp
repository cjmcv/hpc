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
  Node() : run_count_(0), outs_full_(nullptr), name_("noname") {}
  Node(Work &&c) : task_(c), run_count_(0), outs_full_(nullptr), name_("noname") {}
  ~Node() {}

  // Querys.
  inline bool has_task() const { return task_ != nullptr; }
  inline int run_count() const { return run_count_.load(); }
  inline int num_cached_buf(int id) const { return outs_full_[id].size(); }
  inline int num_empty_buf() const { return outs_free_.size(); }
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

  inline bool IsDependsOutEmpty(int depend_id) {
    return dependents_[depend_id]->outs_full_[depends_branch_id_[depend_id]].empty();
  }

  void Init(int buffer_queue_size);
  void Clean();
  void Run(std::vector<Blob *> &inputs, Blob *output);


  bool BorrowInputs(std::vector<Blob *> &inputs); 
  bool RecycleInputs(std::vector<Blob *> &inputs);

  bool PrepareOutput(Blob **output);
  bool PushOutput(Blob *output);

  bool Enqueue(Blob *input);
  bool Dequeue(Blob *output);

public:
  std::condition_variable cond_;
  mutable std::mutex mutex_;

  BlockingQueue<Blob *> outs_free_;
  BlockingQueue<Blob *> *outs_full_;

private:
  Work task_;
  std::string name_;   

  std::vector<Node*> successors_;
  std::vector<Node*> dependents_;
  std::vector<int> depends_branch_id_; 
  
  std::atomic<int> run_count_;
  std::vector<Blob *> outs_;
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
    LOG(ERROR) << "Node::Init -> Failed to new for outs_full_";
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
      LOG(ERROR) << "Node::Init -> Failed to initialize depends_branch_id_";
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

void Node::Run(std::vector<Blob *> &inputs, Blob *output) {
  task_(inputs, output);
  run_count_++;
}

bool Node::BorrowInputs(std::vector<Blob *> &inputs) {
  inputs.clear();
  for (int i = 0; i < num_dependents(); i++) {
    Blob *in = nullptr;
    if (false == dependents_[i]->outs_full_[depends_branch_id_[i]].try_pop(&in)) {
      LOG(ERROR) << "Node::BorrowInputs -> <" << std::this_thread::get_id() 
        << "-" << name_.c_str() << ">BorrowOut outs_full_: No element can be pop.";
      return false;
    }
    inputs.push_back(in);
  }
  return true;
}

bool Node::RecycleInputs(std::vector<Blob *> &inputs) {
  for (int i = 0; i < num_dependents(); i++) {
    dependents_[i]->outs_free_.push(inputs[i]);
  }
  return true;
}

bool Node::PrepareOutput(Blob **output) {
  if (false == outs_free_.try_pop(output)) {
    LOG(ERROR) << "Node::PrepareOutput -> Failed to outs_free_.try_pop in blob-" << name_.c_str();
    return false;
  }
  return true;
}

bool Node::PushOutput(Blob *output) {
  outs_full_[0].push(output);
  Blob *p2;
  for (int i = 1; i < num_successors(); i++) {
    if (false == outs_free_.try_pop(&p2)) {
      LOG(ERROR) << "Node::PushOutput -> Failed to outs_free_.try_pop in blob-" << name_.c_str();
      return false;
    }
    output->CloneTo(p2);
    outs_full_[i].push(p2);
  }
  return true;
}

bool Node::Enqueue(Blob *input) {
  Blob *inside_out;
  for (int i = 0; i < successors_.size(); i++) {
    if (false == outs_free_.try_pop(&inside_out)) {
      LOG(ERROR) << "Node::Enqueue -> Failed in thread-" << std::this_thread::get_id();
      return false;
    }
    input->CloneTo(inside_out);
    outs_full_[i].push(inside_out);
  }
  return true;
}

bool Node::Dequeue(Blob *output) {
  if (successors_.size() >= 1) {
    LOG(ERROR) << "Node::Dequeue -> You can only pop output from output nodes of the graph";
    return false;
  }
  Blob *inside_out;
  if (false == outs_full_[0].try_pop(&inside_out)) {
    LOG(WARNING) << "Node::Dequeue -> No element can be pop in blob-" << name_.c_str();
    return false;
  }

  inside_out->CloneTo(output);
  outs_free_.push(inside_out);
  return true;
}

}  // end of namespace hcs.

#endif //HCS_NODE_H_