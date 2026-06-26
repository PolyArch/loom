// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// vec_size must be a power of two.
func.func @parallelize_bad_vec_size(%data: i8, %cont: i1) -> (i8, i8, i8, i4) {
  // expected-error @+1 {{'vec_size' must be a power of two in the range [1, 64]}}
  %a, %b, %c, %mask = dataflow.parallelize %data, %cont {vec_size = 3 : i64}
    : (i8, i1) -> (i8, i8, i8, i4)
  return %a, %b, %c, %mask : i8, i8, i8, i4
}

// -----
// Lane count must match vec_size.
func.func @pack_wrong_lane_count(%a: i8, %b: i8, %mask: i2) -> i16 {
  // expected-error @+1 {{lane count 2 must match 'vec_size' 4}}
  %packed = dataflow.pack %a, %b mask %mask {vec_size = 4 : i64}
    : (i8, i8, i2) -> i16
  return %packed : i16
}

// -----
// Mask width must match vec_size.
func.func @unpack_bad_mask(%packed: i32, %mask: i8) -> (i8, i8, i8, i8) {
  // expected-error @+1 {{mask type width 8 must match 'vec_size' 4}}
  %a, %b, %c, %d = dataflow.unpack %packed, %mask {vec_size = 4 : i64}
    : (i32, i8) -> (i8, i8, i8, i8)
  return %a, %b, %c, %d : i8, i8, i8, i8
}

// -----
// Packed width must equal lane width times vec_size.
func.func @serialize_bad_lane_type(%a: i8, %b: i16, %mask: i2) -> (i8, i1) {
  // expected-error @+1 {{lane #1 type 'i16' must match lane #0 type 'i8'}}
  %data, %cont = dataflow.serialize %a, %b mask %mask {vec_size = 2 : i64}
    : (i8, i16, i2) -> (i8, i1)
  return %data, %cont : i8, i1
}
