#ifndef EKOREPP_HARMONICS_CACHE_HPP_
#define EKOREPP_HARMONICS_CACHE_HPP_

#include <unordered_map>
#include "../types.hpp"
#include "./w1.hpp"

namespace ekorepp {
namespace harmonics {

/** @brief available elements */
enum K {
    S1
};

/** @brief Cache type */
typedef std::unordered_map<K, cmplx> mp;

/** @brief A simple caching system */
class Cache {
    /** @brief Mellin base value */
    cmplx n;

    /** @brief cache list */
    mp m;

public:

    /**
     * @brief Init cache
     * @param N Mellin base value
     */
    Cache(const cmplx N) : n(N), m() {}

    /**
     * Get Mellin base value
     * @return Mellin base value
     */
    cmplx N() const { return this->n; }

    /**
     * @brief Obtain an element
     * @param key key
     * @return associated element
     */
    cmplx get(const K key) {
        mp::const_iterator it = this->m.find(key);
        if (it != this->m.cend())
            return it->second;
        cmplx res;
        switch (key)
        {
        case S1: res = w1::S1(this->n); break;
        }
        this->m[key] = res;
        return res;
    }
};

} // namespace harmonics
} // namespace ekorepp

#endif // EKOREPP_HARMONICS_CACHE_HPP_
