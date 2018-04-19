#ifndef OZ_STL_HASH_H
#define OZ_STL_HASH_H

#include <functional>
#include <utility>

namespace oz {

template<class T>
inline void hash_combine(std::size_t &seed, const T &v) {
  std::hash<T> h;
  seed ^= h(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

};

namespace std {

template<typename S, typename T>
struct hash<pair<S, T>> {
  inline size_t operator ()(const pair<S, T>& v) const {
    size_t seed = 0;
    oz::hash_combine(seed, v.first);
    oz::hash_combine(seed, v.second);
    return seed;
  }
};

} // namespace std

#endif // OZ_STL_HASH_H
