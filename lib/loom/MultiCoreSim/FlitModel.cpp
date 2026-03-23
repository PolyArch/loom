#include "loom/MultiCoreSim/FlitModel.h"

#include <cassert>

namespace loom {
namespace mcsim {

//===----------------------------------------------------------------------===//
// SimFlitBuffer
//===----------------------------------------------------------------------===//

SimFlitBuffer::SimFlitBuffer(unsigned capacity) : capacity_(capacity) {
  assert(capacity > 0 && "buffer capacity must be positive");
}

bool SimFlitBuffer::canEnqueue() const {
  return static_cast<unsigned>(buffer_.size()) < capacity_;
}

bool SimFlitBuffer::hasFlits() const { return !buffer_.empty(); }

unsigned SimFlitBuffer::size() const {
  return static_cast<unsigned>(buffer_.size());
}

unsigned SimFlitBuffer::capacity() const { return capacity_; }

bool SimFlitBuffer::enqueue(const SimFlit &flit) {
  if (!canEnqueue())
    return false;
  buffer_.push_back(flit);
  return true;
}

const SimFlit &SimFlitBuffer::front() const {
  assert(!buffer_.empty() && "front() called on empty buffer");
  return buffer_.front();
}

SimFlit SimFlitBuffer::dequeue() {
  assert(!buffer_.empty() && "dequeue() called on empty buffer");
  SimFlit flit = std::move(buffer_.front());
  buffer_.pop_front();
  return flit;
}

void SimFlitBuffer::clear() { buffer_.clear(); }

void SimFlitBuffer::setCapacity(unsigned cap) {
  assert(cap > 0 && "buffer capacity must be positive");
  capacity_ = cap;
  buffer_.clear();
}

} // namespace mcsim
} // namespace loom
