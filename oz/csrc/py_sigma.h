#ifndef OZ_PY_SIGMA_H
#define OZ_PY_SIGMA_H

#include <pybind11/pybind11.h>

#include "oos.h"

namespace oz {

class py_sigma_t final : public sigma_t::concept_t {
 public:
  explicit py_sigma_t(pybind11::object callback_fn):
      callback_fn_(move(callback_fn)) { }

  prob_t pr(infoset_t infoset, action_t a) const override;

 private:
  pybind11::object callback_fn_;
};

} // namespace oz

#endif // OZ_PY_SIGMA_H
