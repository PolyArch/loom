//===-- ADGBuilderTypes.cpp - Type and MemrefType implementation --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/adg.h"

#include <cassert>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

Type Type::tagged(Type value, Type tag) {
  Type t(Tagged);
  t.taggedData_ = std::make_shared<TaggedData>(TaggedData{value, tag});
  return t;
}

Type Type::getValueType() const {
  assert(kind_ == Tagged && taggedData_ && "Not a tagged type");
  return taggedData_->valueType;
}

Type Type::getTagType() const {
  assert(kind_ == Tagged && taggedData_ && "Not a tagged type");
  return taggedData_->tagType;
}

std::string Type::toMLIR() const {
  switch (kind_) {
  case I1:    return "i1";
  case I8:    return "i8";
  case I16:   return "i16";
  case I32:   return "i32";
  case I64:   return "i64";
  case IN:    return "i" + std::to_string(width_);
  case BF16:  return "bf16";
  case F16:   return "f16";
  case F32:   return "f32";
  case F64:   return "f64";
  case Index: return "index";
  case None:  return "none";
  case Tagged:
    assert(taggedData_ && "Tagged type missing data");
    return "!dataflow.tagged<" + taggedData_->valueType.toMLIR() + ", " +
           taggedData_->tagType.toMLIR() + ">";
  }
  return "i32"; // fallback
}

bool Type::operator==(const Type &other) const {
  if (kind_ != other.kind_)
    return false;
  if (kind_ == IN)
    return width_ == other.width_;
  if (kind_ == Tagged) {
    if (!taggedData_ || !other.taggedData_)
      return taggedData_ == other.taggedData_;
    return taggedData_->valueType == other.taggedData_->valueType &&
           taggedData_->tagType == other.taggedData_->tagType;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// MemrefType
//===----------------------------------------------------------------------===//

MemrefType MemrefType::static1D(unsigned size, Type elemType) {
  return MemrefType(false, size, elemType);
}

MemrefType MemrefType::dynamic1D(Type elemType) {
  return MemrefType(true, 0, elemType);
}

std::string MemrefType::toMLIR() const {
  if (isDynamic_)
    return "memref<?" + std::string("x") + elemType_.toMLIR() + ">";
  return "memref<" + std::to_string(size_) + "x" + elemType_.toMLIR() + ">";
}

} // namespace adg
} // namespace loom
