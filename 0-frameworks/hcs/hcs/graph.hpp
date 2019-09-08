#ifndef HCS_GRAPH_H_
#define HCS_GRAPH_H_

#include <optional>
#include <atomic>
#include <functional>

#include "params.hpp"
#include "util/blocking_queue.hpp"

namespace hcs {

// Forward declaration
class Graph;

// Class: Node
class Node {

  using StaticWork = std::function<void(std::vector<Node*> &dependents, IOParams **output)>;
  static const int kMaxBatchSize = 10;

public:
  Node() = delete;
  Node(ParamsMode output_mode) {
    Init(output_mode);
  }
  Node(StaticWork &&c, ParamsMode output_mode) : work_(c) {
    Init(output_mode);
  }

  ~Node() {
    Clean();
  }

  void precede(Node &v) {
    successors_.push_back(&v);
    v.dependents_.push_back(this);
    v.atomic_num_depends_.fetch_add(1, std::memory_order_relaxed);
  }

  template <typename... Ns>
  void precede(Ns&&... node) {
    (precede(*(node)), ...);
  }

  size_t num_successors() const { return successors_.size(); }
  size_t num_dependents() const { return dependents_.size(); }
  Node *successor(int id) { return successors_[id]; }
  const std::string &name() const { return name_; }

  Node *name(const std::string& name) { name_ = name; return this; }

  bool FrontOutput(IOParams *out) {
    IOParams *inside_out;
    if (false == outs_full_.try_front(&inside_out)) {
      std::cout << "FrontOutput: No element can be pop." << std::endl;
      return false;
    }
    Assistor::CopyParams(inside_out, out);
  }

  bool PopOutput(IOParams *out, int branch_id = -1) {
    IOParams *inside_out;
    if (branch_id == -1) {
      if (false == outs_full_.try_pop(&inside_out)) {
        std::cout << "PopOutput: No element can be pop." << std::endl;
        return false;
      }
    }
    else {
      if (false == outs_branch_full_[branch_id].try_pop(&inside_out)) {
        std::cout << "PopOutput: No element can be pop." << std::endl;
        return false;
      }
    }
    Assistor::CopyParams(inside_out, out);
    outs_free_.push(inside_out);
  }

  bool PushOutput(IOParams *out) {
    IOParams *inside_out;
    if (false == outs_free_.try_pop(&inside_out)) {
      std::cout << "PushOutput: failed." << std::endl;
      return false;
    }
    Assistor::CopyParams(out, inside_out);
    outs_full_.push(inside_out);
  }

public:
  StaticWork work_;
  std::atomic<int> atomic_num_depends_{ 0 };
  std::vector<Node*> successors_;
  std::vector<Node*> dependents_;

  IOParams *outs_[kMaxBatchSize];  
  BlockingQueue<IOParams *> outs_free_;
  BlockingQueue<IOParams *> outs_full_;
  BlockingQueue<IOParams *> *outs_branch_full_{ nullptr };

private:
  void Init(ParamsMode output_mode) {
    for (int i = 0; i < kMaxBatchSize; i++) {
      outs_[i] = Assistor::CreateParams(output_mode);
      if (outs_[i] != nullptr) {
        outs_free_.push(outs_[i]);
      }
    }
  }
  void Clean() {
    for (int i = 0; i < kMaxBatchSize; i++) {
      if (outs_[i] != nullptr) {
        delete outs_[i];
      }
    }
  }
private:
  std::string name_;
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
  // create a node from a give argument; constructor is called if necessary
  template <typename C>
  Node *emplace(C &&c, ParamsMode mode) {
    nodes_.push_back(std::make_unique<Node>(std::forward<C>(c), mode));
    return &(*(nodes_.back()));
  }
  Node *emplace(ParamsMode mode) {
    nodes_.push_back(std::make_unique<Node>(mode));
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
};

}  // end of namespace hcs.

#endif //HCS_GRAPH_H_