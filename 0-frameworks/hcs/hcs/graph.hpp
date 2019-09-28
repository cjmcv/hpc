#ifndef HCS_GRAPH_H_
#define HCS_GRAPH_H_

#include <string>
#include "node.hpp"

namespace hcs {

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
      if (node->num_successors() >= 2) {
        buffer_queue_size *= node->num_successors();
      }
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