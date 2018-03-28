#ifndef OZ_OSS_H
#define OZ_OSS_H

#include <tuple>
#include <random>

#include "game.h"

namespace oz {

struct action_prob_t {
  action_t a;
  prob_t pr_a;  // probability action was taken under policy
  prob_t rho1;  // probability of sampling action when targeted
  prob_t rho2;  // probability of sampling action
};

class history_t {
 public:
  history_t(const history_t& that): self_(that.self_->clone()) { };
  history_t(history_t&& that) noexcept: self_(move(that.self_)) { };

  void act(action_t a) { self_->act(a); }
  infoset_t infoset() const { return self_->infoset(); }
  player_t player() const { return self_->player(); }
  bool is_terminal() const { return self_->is_terminal(); }
  value_t utility(player_t player) const { return self_->utility(player); }
  action_prob_t sample_chance();

  history_t operator >>(action_t a) const {
    auto g = self_->clone();
    g->act(a);
    return history_t(move(g));
  }

 private:
  using ptr_t = std::unique_ptr<game_t>;

  explicit history_t(ptr_t game): self_(move(game)) { };
  template <class Infoset, typename... Args>
    friend history_t make_history(Args&&... args);

  ptr_t self_;
};

template <class Game, typename... Args>
auto make_history(Args&&... args) -> history_t {
  return history_t(history_t::ptr_t {
      new Game(std::forward<Args>(args)...)
  });
}

//class rng_t { };

class sigma_t {
 public:
  struct concept_t {
    virtual prob_t pr(infoset_t infoset, action_t a) const = 0;
    virtual action_prob_t sample_pr(infoset_t infoset) const;
    virtual ~concept_t() = default;
  };

  prob_t pr(infoset_t infoset, action_t a) const {
    return self_->pr(std::move(infoset), a);
  };

  action_prob_t sample_pr(infoset_t infoset) const {
    return self_->sample_pr(std::move(infoset));
  }

 private:
  using ptr_t = std::shared_ptr<const concept_t>;

  explicit sigma_t(ptr_t self): self_(std::move(self)) { };
  template <class Sigma, typename... Args>
    friend sigma_t make_sigma(Args&&... args);

  ptr_t self_;
};

template <class Sigma, typename... Args>
sigma_t make_sigma(Args&&... args) {
  return sigma_t(std::make_shared<Sigma>(std::forward<Args>(args)...));
};

class node_t {
 public:
  sigma_t sigma_regret_matching();
  void accumulate_regret(action_t a, value_t r);
  void accumulate_average_strategy(action_t a, prob_t s);
};

class tree_t {
 public:
  node_t lookup(infoset_t infoset);
  std::tuple<action_prob_t, bool> sample_sigma(infoset_t infoset);
};

class oss_t {
 public:
  struct prefix_prob_t {
    prob_t pi_i;  // reach probability for search player
    prob_t pi_o;  // reach probability for opponent player and chance
    prob_t s1;    // probability of this sample when targeted
    prob_t s2;    // probability of this sample
  };

  struct suffix_prob_t {
    prob_t x;     // suffix probability
    prob_t l;     // tip-to-tail sample probability
    value_t u;    // value at the terminal
  };

  struct path_item_t {
    player_t player;
    infoset_t infoset;
    action_prob_t action_prob;
    prefix_prob_t prefix_prob;
  };

  class search_t {
   private:
    void step(action_prob_t ap);
    void walk(tree_t tree);
    void unwind(tree_t tree, suffix_prob_t prob);

    enum {IN_TREE, ROLLOUT_EVAL, PRIOR_EVAL, TERMINAL, FINISHED} state_;

    player_t search_player_;
    prefix_prob_t prefix_prob_;
    history_t history_;
    std::vector<path_item_t> path_;

    prob_t delta_;
  };

 private:
//  tree_t tree_;
};

} // namespace oz

#endif // OZ_OSS_H
