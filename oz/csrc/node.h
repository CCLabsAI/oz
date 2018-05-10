#ifndef OZ_NODE_H
#define OZ_NODE_H

#include "sigma.h"

namespace oz {

class sigma_regret_t final : public sigma_t::concept_t {
 public:
  explicit sigma_regret_t(const regret_map_t &regrets):
      regrets_(regrets) { };

  prob_t pr(infoset_t infoset, action_t a) const override;
  action_prob_t sample_pr(infoset_t infoset, rng_t &rng) const override;

 private:
  const regret_map_t &regrets_;
};

class sigma_regret_prior_t final : public sigma_t::concept_t {
 public:
  explicit sigma_regret_prior_t(const action_value_map_t &regrets,
                                const action_value_map_t &prior,
                                prob_t prior_alpha):
      regrets_(regrets),
      prior_(prior),
      prior_alpha_(prior_alpha) { };

  prob_t pr(infoset_t infoset, action_t a) const override;

 private:
  const action_value_map_t &regrets_;
  const action_value_map_t &prior_;
  const prob_t prior_alpha_;
};

class node_t final {
 public:
  using node_sigma_t = sigma_regret_t;

  explicit node_t(infoset_t::actions_list_t actions);

  node_sigma_t sigma_regret_matching() const;

  const value_t &regret(action_t a) const { return regrets_.at(a); }
  value_t &regret(action_t a) { return regrets_.at(a); }

  const prob_t &average_strategy(action_t a) const { return average_strategy_.at(a); }
  prob_t &average_strategy(action_t a) { return average_strategy_.at(a); }

  const prob_t &prior(action_t a) const { return prior_.at(a); }
  prob_t &prior(action_t a) { return prior_.at(a); }

  int regret_n() const { return regret_n_; }
  int &regret_n() { return regret_n_; }

 private:
  friend class oos_t;

  regret_map_t regrets_;
  action_prob_map_t average_strategy_;
  action_prob_map_t prior_;

  int regret_n_ = 0;

 public:
  // used only in python interface
  regret_map_t &regret_map() { return regrets_; }
  action_prob_map_t &average_strategy_map() { return average_strategy_; }
  void accumulate_regret(action_t a, value_t r) { regrets_[a] += r; }
  void accumulate_average_strategy(action_t a, prob_t s) { average_strategy_[a] += s; }
};

} // namespace oz

#endif // OZ_NODE_H
