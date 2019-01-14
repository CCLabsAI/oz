#include "py_sigma.h"

namespace oz {

namespace py = pybind11;

auto py_sigma_t::pr(infoset_t infoset, action_t a) const -> prob_t {
  const auto ob = callback_fn_(infoset, a);
  const auto pr = ob.cast<prob_t>();

  return pr;
}

} // namespace oz
