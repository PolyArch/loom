#ifndef LOOM_MULTICORESIM_FLITMODEL_H
#define LOOM_MULTICORESIM_FLITMODEL_H

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// SimFlit -- A single flit in the NoC simulation
//===----------------------------------------------------------------------===//

/// Flit type within a wormhole-switched message.
enum class SimFlitType : uint8_t {
  HEAD = 0,   // Header flit: reserves the path
  BODY = 1,   // Body flit: follows the path reserved by HEAD
  TAIL = 2,   // Tail flit: releases the path
  SINGLE = 3, // Single-flit message (combined HEAD + TAIL)
};

/// A single flit traversing the NoC.
struct SimFlit {
  SimFlitType type = SimFlitType::SINGLE;

  /// Source and destination core identifiers.
  unsigned srcCoreId = 0;
  unsigned dstCoreId = 0;

  /// Source and destination router coordinates in the mesh.
  int srcRow = 0;
  int srcCol = 0;
  int dstRow = 0;
  int dstCol = 0;

  /// Virtual channel assignment.
  unsigned vcId = 0;

  /// Cycle at which this flit was first injected into the network.
  uint64_t injectionCycle = 0;

  /// Unique message ID (all flits of one message share the same ID).
  uint64_t messageId = 0;

  /// Payload data.
  std::vector<uint8_t> payload;

  /// Contract edge name (for tracing provenance).
  std::string contractEdge;
};

//===----------------------------------------------------------------------===//
// SimFlitBuffer -- FIFO buffer with bounded capacity
//===----------------------------------------------------------------------===//

/// A bounded FIFO queue for storing flits at a router port.
class SimFlitBuffer {
public:
  explicit SimFlitBuffer(unsigned capacity = 4);

  /// Check if the buffer has room for another flit.
  bool canEnqueue() const;

  /// Check if the buffer has any flits.
  bool hasFlits() const;

  /// Get the number of flits currently in the buffer.
  unsigned size() const;

  /// Get the buffer capacity.
  unsigned capacity() const;

  /// Enqueue a flit. Returns false if the buffer is full.
  bool enqueue(const SimFlit &flit);

  /// Peek at the front flit without removing it.
  /// Precondition: hasFlits() is true.
  const SimFlit &front() const;

  /// Dequeue and return the front flit.
  /// Precondition: hasFlits() is true.
  SimFlit dequeue();

  /// Remove all flits from the buffer.
  void clear();

  /// Set buffer capacity (also clears existing contents).
  void setCapacity(unsigned cap);

private:
  std::deque<SimFlit> buffer_;
  unsigned capacity_ = 4;
};

} // namespace mcsim
} // namespace loom

#endif // LOOM_MULTICORESIM_FLITMODEL_H
