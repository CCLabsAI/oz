#ifndef OZ_OOS_SEARCH_H
#define OZ_OOS_SEARCH_H

#include "oos_base.h"

#include "history.h"
#include "target.h"

#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/vector.hpp>

namespace oz {

class tree_t;
class node_t;

using std::move;

static constexpr prob_t NaN = std::numeric_limits<prob_t>::signaling_NaN();

class oos_t final {
 public:
  void search(history_t h, int n_iter, tree_t &tree, rng_t &rng,
              prob_t eps = 0.4,
              prob_t delta = 0.6,
              prob_t gamma = 0.0,
              prob_t beta = 1.0);

  void search_targeted(history_t h, int n_iter, tree_t &tree, rng_t &rng,
                       target_t target, infoset_t target_infoset,
                       prob_t eps = 0.2,
                       prob_t delta = 0.6,
                       prob_t gamma = 0.01,
                       prob_t beta = 0.99);

  prob_t avg_targeting_ratio() const { return avg_targeting_ratio_; }

  void reset_targeting_ratio() {
    avg_targeting_ratio_ = 1.0;
  }

 private:
  void search_iter(history_t h, player_t player, tree_t &tree, rng_t &rng,
                   target_t target, infoset_t target_infoset,
                   void *buffer, size_t buffer_size,
                   prob_t eps, prob_t delta, prob_t gamma, prob_t beta);

  prob_t avg_targeting_ratio_ = 1.0;

  // state machine representing a search
 public:
  class search_t final {
   public:
    using allocator_type = boost::container::pmr::memory_resource*;

    search_t(history_t history, player_t search_player):
        state_(state_t::SELECT),
        history_(move(history)),
        target_(null_target()),
        target_infoset_(null_infoset()),
        search_player_(search_player),
        targeted_(false),
        eps_(0.4),
        delta_(0.0),
        gamma_(0.0),
        eta_(0.0)
        // zeta_(0.0)
    { }

    search_t(history_t history, player_t search_player,
             target_t target, infoset_t target_infoset,
             allocator_type allocator,
             prob_t eps, prob_t delta, prob_t gamma,
             prob_t eta = 0.0, prob_t zeta = 0.0):
        state_(state_t::SELECT),
        history_(move(history)),
        path_(allocator),
        target_(move(target)),
        target_infoset_(move(target_infoset)),
        search_player_(search_player),
        targeted_(false),
        eps_(eps),
        delta_(delta),
        gamma_(gamma),
        eta_(eta)
        // zeta_(zeta)
    { }

    void select(const tree_t& tree, rng_t &rng); // walk from tip to leaf and updating path
    void create(tree_t& tree, rng_t &rng);       // add node to tree with zero values
    void playout_step(action_prob_t ap);         // step playout forward one ply
    void backprop(tree_t& tree);                 // unwind updates along path

    // add node to tree with prior values
    void create_prior(tree_t& tree,
                      action_prob_map_t average_strategy,
                      rng_t &rng);

    const history_t &history() const { return history_; }
    infoset_t infoset() const { return history_.infoset(); }

    // states are mostly sequential
    // PLAYOUT has a self loop
    // SELECT and CREATE may move straight to BACKPROP
    enum class state_t {
      SELECT,   // initial state
      CREATE,   // create node (with prior information)
      PLAYOUT,  // waiting for playout policy evaluation
      BACKPROP, // history is terminal, waiting to apply updates
      FINISHED
    };
    // invariant: CREATE, PLAYOUT => history is not terminal
    // invariant: BACKPROP, FINISHED => history is terminal

    prob_t targeting_ratio() const;
    void set_initial_weight(prob_t w);
    allocator_type get_allocator() const;

    state_t state() const { return state_; };
    player_t search_player() const { return search_player_; }

   private:
    void tree_step(action_prob_t ap); // take one step in-tree and extend path
    void tree_step(action_prob_t ap, const infoset_t& infoset); // take one step in-tree and extend path
    void prepare_suffix_probs();

    sample_ret_t sample_tree(const tree_t &tree,
                             const infoset_t &infoset,
                             rng_t &rng) const;

    void insert_node_step(tree_t &tree,
                          const infoset_t &infoset,
                          const node_t &node,
                          rng_t &rng);

    struct prefix_prob_t {
      prob_t pi_i = 1.0;  // reach probability for search player to current history
      prob_t pi_o = 1.0;  // reach probability for opponent player and chance
      prob_t s1 = 1.0;    // probability of this sample when targeted
      prob_t s2 = 1.0;    // probability of this sample (untargeted)
    };

    struct suffix_prob_t {
      prob_t x = 1.0;     // playout / suffix probability
      prob_t l = NaN;     // total tip-to-tail sample probability (only known at terminal)
      value_t u = NaN;    // value at the terminal (only known at terminal)
    };

    struct path_item_t {
      player_t player;
      infoset_t infoset;
      action_prob_t action_prob;
      prefix_prob_t prefix_prob;
    };

    using path_t = boost::container::pmr::vector<path_item_t>;

    state_t state_;

    history_t history_;
    path_t path_;
    target_t target_;
    infoset_t target_infoset_;

    player_t search_player_;
    prefix_prob_t prefix_prob_;
    suffix_prob_t suffix_prob_;

    bool targeted_; // is this iteration targeted?
    bool average_response_; // is this an average response iteration?

    prob_t eps_;
    prob_t delta_;
    prob_t gamma_;
    prob_t eta_;
    // prob_t zeta_;
  }; // class search_t
}; // class oos_t

} // namespace oz

#endif // OZ_OOS_SEARCH_H
