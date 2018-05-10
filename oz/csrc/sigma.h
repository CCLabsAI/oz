#ifndef OZ_SIGMA_H
#define OZ_SIGMA_H

#include "oos_base.h"

namespace oz {

class sigma_t final {
 public:
  using allocator_t = polymorphic_allocator<sigma_t>;

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

 private:
  using ptr_t = std::shared_ptr<const concept_t>;

  explicit sigma_t(ptr_t self) : self_(move(self)) {};

  template<class Sigma, typename... Args>
  friend sigma_t make_sigma(Args&& ... args);

  template<class Sigma, class Alloc, typename... Args>
  friend sigma_t allocate_sigma(Alloc alloc, Args&& ... args);

  ptr_t self_;
};

template<class Sigma, typename... Args>
sigma_t make_sigma(Args&& ... args) {
  return sigma_t(std::make_shared<Sigma>(std::forward<Args>(args)...));
}

template<class Sigma, class Alloc, typename... Args>
sigma_t allocate_sigma(Alloc alloc, Args&& ... args) {
  return sigma_t(std::allocate_shared<Sigma>(alloc, std::forward<Args>(args)...));
}

} // namespace oz
#endif // OZ_SIGMA_H
