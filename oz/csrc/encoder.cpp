#include "encoder.h"

#include "games/leduk.h"

namespace oz {

using namespace std;
using namespace at;

template <class T, class U>
auto assert_cast(U&& x) -> T {
#ifndef NDEBUG
  return dynamic_cast<T>(std::forward<U>(x));
#else
  return static_cast<T>(std::forward<U>(x));
#endif
};

void leduk_encoder_t::encode(infoset_t infoset, Tensor x) {
  const auto &leduk_infoset =
      assert_cast<const leduk_poker_t::infoset_t&>(infoset.get());

  auto x_a = x.accessor<float,1>();
  for (int i = 0; i < x_a.size(0); i++) {
    x_a[i] = i;
  }
}

} // namespace oz
