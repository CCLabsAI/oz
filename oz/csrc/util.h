#ifndef OZ_UTIL_H
#define OZ_UTIL_H

#include <functional>
#include <utility>

template<class T>
inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> h;
  seed ^= h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {

template<typename S, typename T>
struct hash<std::pair<S, T>> {
  inline size_t operator ()(const std::pair<S, T> &v) const {
    size_t seed = 0;
    ::hash_combine(seed, v.first);
    ::hash_combine(seed, v.second);
    return seed;
  }
};

} // namespace std

namespace oz {

  template <typename ForwardIt, typename Projection>
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

} // namespace oz

#endif // OZ_UTIL_H
