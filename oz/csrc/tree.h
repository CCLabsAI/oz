#ifndef OZ_TREE_H
#define OZ_TREE_H

#include "node.h"

#include <set>
#include <unordered_map>

namespace oz {

using std::set;
using std::unordered_map;

class tree_t final {
 public:
  using map_t = unordered_map<infoset_t, node_t>;

  struct sample_ret_t {
    action_prob_t ap;
    bool out_of_tree = false;
  };

  void create_node(infoset_t infoset);

  node_t &lookup(const infoset_t &infoset) { return nodes_.at(infoset); }
  const node_t &lookup(const infoset_t &infoset) const { return nodes_.at(infoset); }

  sample_ret_t sample_sigma(const infoset_t &infoset,
                            const set<action_t> &targets,
                            bool targeted,
                            bool average_response,
                            prob_t eps,
                            prob_t gamma,
                            rng_t &rng) const;

  sigma_t sigma_average() const;

  map_t &nodes() { return nodes_; }
  const map_t &nodes() const { return nodes_; }

  map_t::size_type size() const { return nodes_.size(); }

  void clear();

 private:
  map_t nodes_;
};

class sigma_average_t final : public sigma_t::concept_t {
 public:
  explicit sigma_average_t(const tree_t &tree):
      tree_(tree) { };

  prob_t pr(infoset_t infoset, action_t a) const override;

 private:
  const tree_t &tree_;
};

} // namespace oz

#endif // OZ_TREE_H
