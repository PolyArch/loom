//===-- SimTagOps.h - Simulated tag operations ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Combinational tag operation models:
//   fabric.add_tag - appends configured tag value
//   fabric.map_tag - table-lookup tag translation
//   fabric.del_tag - strips tag
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SIMULATOR_SIMTAGOPS_H
#define LOOM_SIMULATOR_SIMTAGOPS_H

#include "loom/Simulator/SimModule.h"

namespace loom {
namespace sim {

//===----------------------------------------------------------------------===//
// SimAddTag
//===----------------------------------------------------------------------===//

class SimAddTag : public SimModule {
public:
  SimAddTag(unsigned tagWidth);

  bool isCombinational() const override { return true; }
  void evaluateCombinational() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  unsigned tagWidth_;
  uint16_t configuredTag_ = 0;
};

//===----------------------------------------------------------------------===//
// SimMapTag
//===----------------------------------------------------------------------===//

class SimMapTag : public SimModule {
public:
  SimMapTag(unsigned inTagWidth, unsigned outTagWidth, unsigned tableSize);

  bool isCombinational() const override { return true; }
  void evaluateCombinational() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }

private:
  unsigned inTagWidth_;
  unsigned outTagWidth_;
  unsigned tableSize_;

  /// Translation table: each entry is (valid, srcTag, dstTag).
  struct MapEntry {
    bool valid = false;
    uint16_t srcTag = 0;
    uint16_t dstTag = 0;
  };
  std::vector<MapEntry> table_;
};

//===----------------------------------------------------------------------===//
// SimDelTag
//===----------------------------------------------------------------------===//

class SimDelTag : public SimModule {
public:
  SimDelTag();

  bool isCombinational() const override { return true; }
  void evaluateCombinational() override;
  void reset() override;
  void configure(const std::vector<uint32_t> &configWords) override;
  void collectTraceEvents(std::vector<TraceEvent> &events,
                          uint64_t cycle) override;
  PerfSnapshot getPerfSnapshot() const override { return perf_; }
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTAGOPS_H
