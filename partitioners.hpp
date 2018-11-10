#ifndef __PARTITIONERS_H
#define __PARTITIONERS_H

#include <utility>
#include <numeric>
#include <vector>
#include <cmath>

#include "glog/logging.h"

#include "utils.h"

using IndexRange = std::pair<int, int>;

template <typename T>
using CoefficientIterator = typename std::vector<T>::const_iterator;

//
// Interface.
//
template <typename T>
class Partitioner {
 public:
  virtual std::vector<IndexRange> Partition(
      CoefficientIterator<T> start, CoefficientIterator<T> end) const = 0;
};

//
// Implementations.
//
template <typename T>
class WidthPartitioner : public Partitioner<T> {
 public:
  /// fraction is in (0, 1]
  WidthPartitioner(const float fraction) : fraction_(fraction) {
    if (fraction > 1 || fraction < 0) abort();
  }

  std::vector<IndexRange> Partition(CoefficientIterator<T> start,
                                    CoefficientIterator<T> end) const override;

  float fraction() const { return fraction_; }

 private:
  const float fraction_;
};

template <typename T>
class SizePartitioner : public Partitioner<T> {
 public:
  SizePartitioner(const int max_size) : max_size_(max_size) {}
  std::vector<IndexRange> Partition(CoefficientIterator<T> start,
                                    CoefficientIterator<T> end) const override;

  int max_size() const { return max_size_; }

 private:
  const int max_size_;
};

//
// Implementation
//
template <typename T>
std::vector<IndexRange> WidthPartitioner<T>::Partition(
    CoefficientIterator<T> start, CoefficientIterator<T> end) const {
  /** Returns a vector of pairs. Each pair defines the initial and final index
   * of the coefficient's vector to be considered by each of the m sections.
   *
   * Each pair encompasses a evenly distributed amount of indices.
   *
   * */
  const int n = end - start;
  std::vector<IndexRange> res;
  const int elems_per_partition = std::ceil(n * fraction_);
  for (int i = 0; i < n; i += elems_per_partition) {
    res.push_back(std::make_pair(i, std::min(n, i + elems_per_partition) - 1));
  }
  return res;
}

template <typename T>
std::vector<IndexRange> SizePartitioner<T>::Partition(
    CoefficientIterator<T> start, CoefficientIterator<T> end) const {
  std::vector<int> sizes(end - start);
  std::transform(start, end, sizes.begin(),
                 [](const T& t) { return GetMemoryFootprint(t); });

  if (sizes.empty()) {
    return {};
  }
  auto cost_fn = [](const int size, const int range_size) {
    return size * range_size;
  };

  // Accumulate into a range until done or sum of range exceeds the cache
  // size.
  std::vector<IndexRange> ranges;

  const int n = sizes.size();
  int size = sizes[0];
  int cost = cost_fn(size, 1);
  int sum = cost;
  int start_idx = 0;
  for (int end_idx = 1; end_idx < n; ++end_idx) {
    size = sizes[end_idx];
    cost = cost_fn(size, end_idx - start_idx);
    if (sum + cost > max_size_) {
      // Define a new range.
      ranges.push_back(std::make_pair(start_idx, end_idx - 1));
      start_idx = end_idx;
      sum = cost;
    } else {
      sum += cost;
    }
  }
  if (ranges.empty() || ranges.back().second != n - 1) {
    ranges.push_back(std::make_pair(start_idx, sizes.size() - 1));
  }

  for (const auto& range : ranges) {
    const int sum =
        std::accumulate(sizes.begin() + range.first,
                        sizes.begin() + range.second, 0, std::plus<int>());
    DLOG(INFO) << "Σ(" << range.first << ", " << range.second << ") = " << sum;
  }

  return ranges;
}

#endif  // __PARTITIONERS_H
