#ifndef OZ_OSS_H
#define OZ_OSS_H

#include <tuple>
#include <random>
#include <unordered_map>
#include <map>

#include "game.h"
#include "target.h"

namespace oz {

using rng_t = std::mt19937;

using std::move;
using std::vector;
using std::unordered_map;
using std::map;

struct action_prob_t {
  action_t a;
  prob_t pr_a;  // probability action was taken under policy
  prob_t rho1;  // probability of sampling action when targeted
  prob_t rho2;  // probability of sampling action not targeted
};

class history_t final {
 public:
  history_t(const history_t& that) : self_(that.self_->clone()) {};
  history_t(history_t&& that) = default;

  history_t &operator=(history_t &&that) = default;

  void act(action_t a) { self_->act(a); }
  infoset_t infoset() const { return self_->infoset(); }
  player_t player() const { return self_->player(); }
  bool is_terminal() const { return self_->is_terminal(); }
  value_t utility(player_t player) const { return self_->utility(player); }
  map<action_t, prob_t> chance_actions() const
    { return self_->chance_actions(); }

  action_prob_t sample_chance(rng_t& rng) const;
  action_prob_t sample_uniform(rng_t& rng) const;

  history_t operator>>(action_t a) const {
    auto g = self_->clone();
    g->act(a);
    return history_t(move(g));
  }

  template <class T>
  const T &cast() const { return assert_cast<const T&>(*self_.get()); }

 private:
  using ptr_t = std::unique_ptr<game_t>;

  explicit history_t(ptr_t self) : self_(move(self)) {};

  template<class Infoset, typename... Args>
  friend history_t make_history(Args&& ... args);

  ptr_t self_;
};

template<class Game, typename... Args>
auto make_history(Args&& ... args) -> history_t {
  return history_t(std::make_unique<Game>(std::forward<Args>(args)...));
}

class sigma_t final {
 public:
  struct concept_t {
    virtual prob_t pr(infoset_t infoset, action_t a) const = 0;
    virtual action_prob_t sample_pr(infoset_t infoset, rng_t &rng) const;
    virtual ~concept_t() = default;
  };

  prob_t pr(infoset_t infoset, action_t a) const {
    return self_->pr(move(infoset), a);
  };

  action_prob_t sample_pr(infoset_t infoset, rng_t &rng) const {
    return self_->sample_pr(move(infoset), rng);
  }

  action_prob_t sample_eps(infoset_t infoset, prob_t eps, rng_t &rng) const;

  action_prob_t sample_targeted(infoset_t infoset,
                                set<action_t> targets, bool targeted,
                                prob_t eps, prob_t gamma,
                                rng_t &rng) const;

 private:
  using ptr_t = std::shared_ptr<const concept_t>;

  explicit sigma_t(ptr_t self) : self_(move(self)) {};
  template<class Sigma, typename... Args>
  friend sigma_t make_sigma(Args&& ... args);

  ptr_t self_;
};

template<class Sigma, typename... Args>
sigma_t make_sigma(Args&& ... args) {
  return sigma_t(std::make_shared<Sigma>(std::forward<Args>(args)...));
}

class sigma_regret_t;

class node_t final {
 public:
  using regret_map_t = map<action_t, value_t>;
  using avg_map_t = map<action_t, prob_t>;

  explicit node_t(vector<action_t> actions);

  sigma_t sigma_regret_matching() const { return make_sigma<sigma_regret_t>(regrets_); }

  const value_t &regret(action_t a) const { return regrets_.at(a); }
  value_t &regret(action_t a) { return regrets_.at(a); }

  const prob_t &average_strategy(action_t a) const { return average_strategy_.at(a); }
  prob_t &average_strategy(action_t a) { return average_strategy_.at(a); }

  int regret_n() const { return regret_n_; }
  int &regret_n() { return regret_n_; }

 private:
  friend class oos_t;

  regret_map_t regrets_;
  avg_map_t average_strategy_;

  int regret_n_ = 0;

 public:
  // used only in python interface
  regret_map_t &regret_map() { return regrets_; }
  avg_map_t &average_strategy_map() { return average_strategy_; }
  void accumulate_regret(action_t a, value_t r) { regrets_[a] += r; }
  void accumulate_average_strategy(action_t a, prob_t s) { average_strategy_[a] += s; }
};

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

  sample_ret_t sample_sigma(infoset_t infoset,
                            set<action_t> targets, bool targeted,
                            prob_t eps, prob_t gamma,
                            rng_t &rng) const;

  sigma_t sigma_average() const;

  map_t &nodes() { return nodes_; }
  const map_t &nodes() const { return nodes_; }

  map_t::size_type size() const { return nodes_.size(); }

 private:
  map_t nodes_;
};

class sigma_regret_t final : public sigma_t::concept_t {
 public:
  explicit sigma_regret_t(node_t::regret_map_t regrets):
      regrets_(move(regrets)) { };

  prob_t pr(infoset_t infoset, action_t a) const override;
  action_prob_t sample_pr(infoset_t infoset, rng_t &rng) const override;

 private:
  const node_t::regret_map_t regrets_;
};

class sigma_average_t final : public sigma_t::concept_t {
 public:
  explicit sigma_average_t(tree_t tree):
      tree_(move(tree)) { };

  prob_t pr(infoset_t infoset, action_t a) const override;

 private:
  const tree_t tree_;
};

static_assert(std::numeric_limits<prob_t>::has_signaling_NaN);
static constexpr prob_t NaN = std::numeric_limits<prob_t>::signaling_NaN();

class oos_t final {
 public:
  void search(history_t h, int n_iter, tree_t &tree, rng_t &rng,
              target_t target = target_t{},
              prob_t eps = 0.4,
              prob_t delta = 0.2,
              prob_t gamma = 0.01);

  void search_iter(history_t h, player_t player, tree_t &tree, rng_t &rng,
                   target_t target, prob_t eps, prob_t delta, prob_t gamma);

  // state machine representing a search
  class search_t final {
   public:
    search_t(history_t history, player_t search_player):
        state_(state_t::SELECT),
        history_(move(history)),
        search_player_(search_player),
        targeted_(false),
        eps_(0.4),
        delta_(0.2),
        gamma_(0.01)
    { }

    search_t(history_t history, player_t search_player, target_t target,
             prob_t eps = 0.4, prob_t delta = 0.2, prob_t gamma = 0.01):
        state_(state_t::SELECT),
        history_(move(history)),
        target_(move(target)),
        search_player_(search_player),
        targeted_(false),
        eps_(eps),
        delta_(delta),
        gamma_(gamma)
    { }

    void select(const tree_t& tree, rng_t &rng); // walk from tip to leaf and updating path
    void create(tree_t& tree, rng_t &rng);       // add node to tree with zero values
    void playout_step(action_prob_t ap);         // step playout forward one ply
    void backprop(tree_t& tree);                 // unwind updates along path

    // add node to tree with prior values
    void create_prior(tree_t& tree,
                      node_t::regret_map_t regrets,
                      node_t::avg_map_t average_strategy,
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

    state_t state() const { return state_; };
    player_t search_player() const { return search_player_; }

   private:
    void tree_step(action_prob_t ap); // take one step in-tree and extend path
    void prepare_suffix_probs();

    tree_t::sample_ret_t sample_tree(const tree_t &tree,
                                     const infoset_t &infoset,
                                     rng_t &rng) const;

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

    state_t state_;

    history_t history_;
    vector<path_item_t> path_;
    target_t target_;

    player_t search_player_;
    prefix_prob_t prefix_prob_;
    suffix_prob_t suffix_prob_;

    bool targeted_; // is this iteration targeted?

    prob_t eps_;
    prob_t delta_;
    prob_t gamma_;
  }; // class search_t

}; // class oos_t

} // namespace oz

#endif // OZ_OSS_H
