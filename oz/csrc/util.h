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
                [](auto x){ return x >= 0; });
}

// TODO make this prob_t or something instead of double
template <typename T>
double sum_probs(T col) {
  using namespace std;
  return accumulate(begin(col), end(col), (double) 0);
}

template <typename T>
bool sums_to_one(T col) {
  using namespace std;
  return abs(1.0 - sum_probs(col)) < 1e-9;
}

static inline const bool is_normal(double x) {
    switch(std::fpclassify(x)) {
        case FP_INFINITE:  return false;
        case FP_NAN:       return false;
        case FP_NORMAL:    return true;
        case FP_SUBNORMAL: return false;
        case FP_ZERO:      return true;
        default:           return false;
    }
}

} // namespace oz

#endif // OZ_UTIL_H
