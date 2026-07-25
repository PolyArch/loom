#ifndef LOOM_EVALUATION_OWNERVALUE_H
#define LOOM_EVALUATION_OWNERVALUE_H

#include <memory>
#include <type_traits>
#include <utility>

namespace loom::evaluation {

/// Lifetime-safe type erasure for values adopted by an exact external owner.
/// The owner codec remains responsible for interpreting the value; Evaluation
/// retains it without introducing a generic property or payload API.
class OwnerValue {
public:
  OwnerValue() = default;

  template <typename T> static OwnerValue get(T value) {
    using Value = std::decay_t<T>;
    std::shared_ptr<const Value> storage =
        std::make_shared<const Value>(std::move(value));
    return OwnerValue(std::move(storage), typeToken<Value>());
  }

  template <typename T> const std::decay_t<T> *getIf() const {
    using Value = std::decay_t<T>;
    if (typeToken_ != typeToken<Value>())
      return nullptr;
    return static_cast<const Value *>(storage_.get());
  }

  explicit operator bool() const { return static_cast<bool>(storage_); }

private:
  template <typename T> static const void *typeToken() {
    static const char token = 0;
    return &token;
  }

  OwnerValue(std::shared_ptr<const void> storage, const void *typeToken)
      : storage_(std::move(storage)), typeToken_(typeToken) {}

  std::shared_ptr<const void> storage_;
  const void *typeToken_ = nullptr;
};

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_OWNERVALUE_H
