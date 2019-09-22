#ifndef HCS_GRAPH_H_
#define HCS_GRAPH_H_

#include <optional>
#include <atomic>
#include <functional>

#include "blob.hpp"
#include "util/blocking_queue.hpp"

namespace hcs {

// Forward declaration
class Graph;

// Class: Node
class Node {

  using Work = std::function<void(std::vector<Node*> &dependents, Blob *output)>;

public:
  Node() {}
  Node(Work &&c) : work_(c) {}
  ~Node() {}

  void Init(int buffer_queue_size) {
    outs_branch_full_ = nullptr;
    for (int i = 0; i < buffer_queue_size; i++) {
      Blob *p = new Blob;
      if (p != nullptr) {
        outs_.push_back(p);
        outs_free_.push(p);
      }
    }
    if (successors_.size() >= 2) {
      outs_branch_full_ = new BlockingQueue<Blob *>[successors_.size()];
    }
  }
  void Clean() {
    if (outs_.size() > 0) {
      for (int i = 0; i < outs_.size(); i++) {
        if (outs_[i] != nullptr) {
          delete outs_[i];
          outs_[i] = nullptr;
        }
      }
    }
    if (outs_branch_full_ != nullptr) {
      delete[]outs_branch_full_;
      outs_branch_full_ = nullptr;
    }
  }

  void precede(Node &v) {
    successors_.push_back(&v);
    v.dependents_.push_back(this);
  }

  template <typename... Ns>
  void precede(Ns&&... node) {
    (precede(*(node)), ...);
  }

  size_t num_successors() const { return successors_.size(); }
  size_t num_dependents() const { return dependents_.size(); }
  Node *successor(int id) { return successors_[id]; }
  const std::string &name() const { return name_; }
  const int id() const { return id_; }
  const void set_id(int id) { id_ = id; }

  Node *name(const std::string& name) { name_ = name; return this; }

  void lock() { mutex_.lock(); }
  void unlock() { mutex_.unlock(); }

  Blob *BorrowOut(int branch_id = -1) {
    std::unique_lock<std::mutex> lock(mutex_);
    Blob *out = nullptr;
    if (branch_id == -1) {
      if (false == outs_full_.try_pop(&out)) {
        printf("<%d>PopOutput outs_full_: No element can be pop.", std::this_thread::get_id());
        return nullptr;
      }
    }
    else {
      if (false == outs_branch_full_[branch_id].try_pop(&out)) {
        printf("<%d>PopOutput outs_branch_full_: No element can be pop.", std::this_thread::get_id());
        return nullptr;
      }
    }
    return out;
  }

  bool RecycleOut(Blob *out) {
    outs_free_.push(out);
    return true;
  }

  bool PopOutput(Blob *out, int branch_id = -1) {
    std::unique_lock<std::mutex> lock(mutex_);
    Blob *inside_out;
    if (branch_id == -1) {
      if (false == outs_full_.try_pop(&inside_out)) {
        printf("<%d>PopOutput outs_full_: No element can be pop.", std::this_thread::get_id());
        return false;
      }
    }
    else {
      if (false == outs_branch_full_[branch_id].try_pop(&inside_out)) {
        printf("<%d>PopOutput outs_branch_full_: No element can be pop.", std::this_thread::get_id());
        return false;
      }
    }
    inside_out->CloneTo(out);
    outs_free_.push(inside_out);
    return true;
  }

  bool PushOutput(Blob *out) {
    std::unique_lock<std::mutex> lock(mutex_);
    Blob *inside_out;
    if (false == outs_free_.try_pop(&inside_out)) {
      printf("<%d>PushOutput: failed..", std::this_thread::get_id());
      return false;
    }
    out->CloneTo(inside_out);
    outs_full_.push(inside_out);
    return true;
  }

public:
  Work work_;
  std::atomic<int> atomic_run_count_;
  std::vector<Node*> successors_;
  std::vector<Node*> dependents_;

  std::vector<Blob *> outs_;
  BlockingQueue<Blob *> outs_free_;
  BlockingQueue<Blob *> outs_full_;
  BlockingQueue<Blob *> *outs_branch_full_;

private:
  std::string name_;
  mutable std::mutex mutex_;
  int id_;
};
// ----------------------------------------------------------------------------

// Class: Graph
class Graph {

public:
  Graph() = default;
  Graph(const std::string& name) :name_(name) {}

  // Querys.
  inline const std::string& name() const { return name_; }
  inline void clear() { nodes_.clear(); }
  inline bool is_empty() const { return nodes_.empty(); }
  inline size_t size() const { return nodes_.size(); }
  std::vector<std::unique_ptr<Node>>& nodes() { return nodes_; }
  const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }
  inline int buffer_queue_size() const { return buffer_queue_size_; }

  // Collect header nodes that do not need to rely on other nodes
  std::vector<Node*> &GetInputNodes() {
    input_nodes_.clear();
    for (auto& node : nodes_) {
      if (node->num_dependents() == 0) {
        input_nodes_.push_back(node.get());
      }
    }
    return input_nodes_;
  }
  // Collect output nodes.
  std::vector<Node*> &GetOutputNodes() {
    output_nodes_.clear();
    for (auto& node : nodes_) {
      if (node->num_successors() == 0) {
        output_nodes_.push_back(node.get());
      }
    }
    return output_nodes_;
  }

  void Initialize(int buffer_queue_size) {
    buffer_queue_size_ = buffer_queue_size;
    for (int i = 0; i < nodes_.size(); i++) {
      Node *node = &(*nodes_[i]);
      node->set_id(i);

      if (node->num_successors() >= 2) {
        node->Init(buffer_queue_size * node->num_successors());
      }
      else
        node->Init(buffer_queue_size);
    }
  }
  void Clean() {
    for (auto& node : nodes_) {
      node->Clean();
    }
  }
  // create a node from a give argument; constructor is called if necessary
  template <typename C>
  Node *emplace(C &&c) {
    nodes_.push_back(std::make_unique<Node>(std::forward<C>(c)));
    return &(*(nodes_.back()));
  }
  Node *emplace() {
    nodes_.push_back(std::make_unique<Node>());
    return &(*(nodes_.back()));
  }

  // creates multiple tasks from a list of callable objects at one timeS
  // std::enable_if_t<(sizeof...(C) > 1), void>* = nullptr
  template <typename... C>
  auto emplace(C&&... callables) {
    return std::make_tuple(emplace(std::forward<C>(callables))...);
  }

private:
  std::string name_;
  std::vector<std::unique_ptr<Node>> nodes_;
  
  std::vector<Node*> input_nodes_;
  std::vector<Node*> output_nodes_;

  int buffer_queue_size_ = 0;
};

}  // end of namespace hcs.

#endif //HCS_GRAPH_H_