#ifndef OZ_UTIL_H
#define OZ_UTIL_H

#include <functional>
#include <algorithm>
#include <numeric>
#include <vector>

#include <cassert>
#include <cmath>

#define Expects(cond) assert(cond)
#define Ensures(cond) assert(cond)

namespace oz {

template <class T, class U>
auto assert_cast(U&& x) -> T {
#ifndef NDEBUG
  return dynamic_cast<T>(std::forward<U>(x));
#else
  return static_cast<T>(std::forward<U>(x));
#endif
};

template<typename ForwardIt, typename Projection>
ForwardIt max_element_by(ForwardIt first, ForwardIt last, Projection f) {
  if (first == last) return last;

  ForwardIt largest = first;
  ++first;
  for (; first != last; ++first) {
    if (f(*largest) < f(*first)) {
      largest = first;
    }
  }

  return largest;
};

template <typename T>
bool all_greater_equal_zero(T iter) {
  using namespace std;
  return all_of(begin(iter), end(iter),
                [](const auto &x) { return x >= 0; });
}

// TODO make this prob_t or something instead of double
template <typename T>
double sum_probs(T col) {
  using namespace std;
  return accumulate(begin(col), end(col), (double) 0);
}

template <typename T>
bool sum_to_one(T col) {
  using namespace std;
  return abs(1.0 - sum_probs(col)) < 1e-9;
}

template <typename T>
auto keys(const T &m) -> std::vector<typename T::key_type> {
  using namespace std;
  auto keys = vector<typename T::key_type>();

  transform(begin(m), end(m), back_inserter(keys), [](const auto& p) {
    return p.first;
  });

  return keys;
}

template <typename T>
T rectify(T x) {
  return x > 0 ? x : 0;
}

} // namespace oz

#endif // OZ_UTIL_H
