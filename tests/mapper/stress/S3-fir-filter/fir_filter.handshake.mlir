#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/fir_filter/fir_filter.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/fir_filter/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/fir_filter/fir_filter.cpp":8:0)
#loc3 = loc("tests/app/fir_filter/fir_filter.cpp":11:0)
#loc4 = loc("tests/app/fir_filter/fir_filter.cpp":13:0)
#loc11 = loc("tests/app/fir_filter/fir_filter.cpp":26:0)
#loc27 = loc("tests/app/fir_filter/main.cpp":28:0)
#loc31 = loc("tests/app/fir_filter/main.cpp":34:0)
#loc34 = loc("tests/app/fir_filter/main.cpp":40:0)
#loc38 = loc("tests/app/fir_filter/main.cpp":46:0)
#loc42 = loc("tests/app/fir_filter/main.cpp":53:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 256, elements = #llvm.di_subrange<count = 8 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 96, elements = #llvm.di_subrange<count = 3 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__int32_t", baseType = #di_basic_type2>
#di_label = #llvm.di_label<scope = #di_subprogram1, name = "outer_loop", file = #di_file, line = 29>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 11>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 31>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 28>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 34>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 40>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 46>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 53>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "passed", file = #di_file1, line = 52, type = #di_basic_type3>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "int32_t", baseType = #di_derived_type3>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 11>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 31>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "coeffs", file = #di_file1, line = 15, type = #di_composite_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "cpu_output", file = #di_file1, line = 18, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "dsa_output", file = #di_file1, line = 19, type = #di_composite_type>
#di_derived_type10 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_derived_type11 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type7>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 11>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 31>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 9, arg = 3, type = #di_derived_type6>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_size", file = #di_file, line = 9, arg = 4, type = #di_derived_type7>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "num_taps", file = #di_file, line = 10, arg = 5, type = #di_derived_type7>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block, name = "n", file = #di_file, line = 11, type = #di_derived_type7>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 27, arg = 3, type = #di_derived_type6>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_size", file = #di_file, line = 28, arg = 4, type = #di_derived_type7>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "num_taps", file = #di_file, line = 28, arg = 5, type = #di_derived_type7>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "n", file = #di_file, line = 31, type = #di_derived_type7>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 28, type = #di_derived_type7>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 34, type = #di_derived_type7>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 40, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "i", file = #di_file1, line = 46, type = #di_derived_type7>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "i", file = #di_file1, line = 53, type = #di_derived_type7>
#di_label1 = #llvm.di_label<scope = #di_lexical_block10, name = "inner_loop", file = #di_file, line = 35>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 13>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 36>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 8, arg = 1, type = #di_derived_type10>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "coeffs", file = #di_file, line = 8, arg = 2, type = #di_derived_type10>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "sum", file = #di_file, line = 12, type = #di_basic_type>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 26, arg = 1, type = #di_derived_type10>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "coeffs", file = #di_file, line = 27, arg = 2, type = #di_derived_type10>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "sum", file = #di_file, line = 33, type = #di_basic_type>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_size", file = #di_file1, line = 8, type = #di_derived_type11>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "num_taps", file = #di_file1, line = 9, type = #di_derived_type11>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type10, #di_derived_type10, #di_derived_type6, #di_derived_type7, #di_derived_type7>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 13>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 36>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "k", file = #di_file, line = 13, type = #di_derived_type7>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block12, name = "k", file = #di_file, line = 36, type = #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 7, scopeLine = 7, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable, #di_local_variable17>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 13>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 36>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 28>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 34>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 40>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 46>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 53>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block15, name = "idx", file = #di_file, line = 14, type = #di_derived_type8>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_lexical_block16, name = "idx", file = #di_file, line = 37, type = #di_derived_type8>
#loc62 = loc(fused<#di_lexical_block17>[#loc27])
#loc63 = loc(fused<#di_lexical_block18>[#loc31])
#loc64 = loc(fused<#di_lexical_block19>[#loc34])
#loc65 = loc(fused<#di_lexical_block20>[#loc38])
#loc66 = loc(fused<#di_lexical_block21>[#loc42])
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "fir_filter_cpu", linkageName = "_Z14fir_filter_cpuPKfS0_Pfjj", file = #di_file, line = 8, scopeLine = 10, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable18, #di_local_variable19, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable20, #di_local_variable26, #di_local_variable28>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "fir_filter_dsa", linkageName = "_Z14fir_filter_dsaPKfS0_Pfjj", file = #di_file, line = 26, scopeLine = 28, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_label, #di_local_variable12, #di_local_variable23, #di_label1, #di_local_variable27, #di_local_variable29>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 11>
#loc73 = loc(fused<#di_subprogram5>[#loc1])
#loc75 = loc(fused<#di_subprogram6>[#loc11])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 11>
#loc86 = loc(fused<#di_lexical_block33>[#loc3])
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 11>
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file, line = 13>
#loc94 = loc(fused<#di_lexical_block40>[#loc4])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<14xi8> = dense<[108, 111, 111, 109, 46, 114, 101, 100, 117, 99, 101, 61, 43, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<36xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 102, 105, 114, 95, 102, 105, 108, 116, 101, 114, 47, 102, 105, 114, 95, 102, 105, 108, 116, 101, 114, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<22xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 102, 105, 114, 95, 102, 105, 108, 116, 101, 114, 0]> loc(#loc)
  memref.global constant @__const.main.input : memref<8xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.coeffs : memref<3xf32> = dense<[2.500000e-01, 5.000000e-01, 2.500000e-01]> {alignment = 4 : i64} loc(#loc)
  memref.global constant @".str.1.4" : memref<14xi8> = dense<[73, 110, 112, 117, 116, 58, 32, 32, 32, 32, 32, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.2.6" : memref<7xi8> = dense<[37, 46, 49, 102, 37, 115, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.3.5" : memref<3xi8> = dense<[44, 32, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4" : memref<1xi8> = dense<0> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.6" : memref<14xi8> = dense<[67, 111, 101, 102, 102, 115, 58, 32, 32, 32, 32, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.7" : memref<7xi8> = dense<[37, 46, 50, 102, 37, 115, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.8" : memref<14xi8> = dense<[67, 80, 85, 32, 79, 117, 116, 112, 117, 116, 58, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.9" : memref<14xi8> = dense<[68, 83, 65, 32, 79, 117, 116, 112, 117, 116, 58, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str : memref<45xi8> = dense<[70, 73, 82, 32, 70, 105, 108, 116, 101, 114, 32, 82, 101, 115, 117, 108, 116, 115, 32, 40, 51, 45, 116, 97, 112, 32, 97, 118, 101, 114, 97, 103, 105, 110, 103, 32, 102, 105, 108, 116, 101, 114, 41, 58, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.15 : memref<2xi8> = dense<[93, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.16 : memref<26xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 82, 101, 115, 117, 108, 116, 115, 32, 109, 105, 115, 109, 97, 116, 99, 104, 33, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.17 : memref<29xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 65, 108, 108, 32, 114, 101, 115, 117, 108, 116, 115, 32, 99, 111, 114, 114, 101, 99, 116, 33, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z14fir_filter_cpuPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: i32 loc(fused<#di_subprogram5>[#loc1]), %arg4: i32 loc(fused<#di_subprogram5>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc89)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg3 : i32 to i64 loc(#loc89)
      %3 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %4 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %5 = scf.if %1 -> (f32) {
          scf.yield %cst : f32 loc(#loc94)
        } else {
          %9 = arith.trunci %arg5 : i64 to i32 loc(#loc2)
          %10:2 = scf.while (%arg6 = %c0_i64, %arg7 = %cst) : (i64, f32) -> (i64, f32) {
            %11 = arith.trunci %arg6 : i64 to i32 loc(#loc97)
            %12 = arith.subi %9, %11 : i32 loc(#loc97)
            %13 = arith.cmpi sgt, %12, %c-1_i32 : i32 loc(#loc100)
            %14 = scf.if %13 -> (f32) {
              %17 = arith.index_cast %arg6 : i64 to index loc(#loc102)
              %18 = memref.load %arg1[%17] : memref<?xf32, strided<[1], offset: ?>> loc(#loc102)
              %19 = arith.extui %12 : i32 to i64 loc(#loc102)
              %20 = arith.index_cast %19 : i64 to index loc(#loc102)
              %21 = memref.load %arg0[%20] : memref<?xf32, strided<[1], offset: ?>> loc(#loc102)
              %22 = math.fma %18, %21, %arg7 : f32 loc(#loc102)
              scf.yield %22 : f32 loc(#loc103)
            } else {
              scf.yield %arg7 : f32 loc(#loc100)
            } loc(#loc100)
            %15 = arith.addi %arg6, %c1_i64 : i64 loc(#loc96)
            %16 = arith.cmpi ne, %15, %3 : i64 loc(#loc98)
            scf.condition(%16) %15, %14 : i64, f32 loc(#loc94)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block40>[#loc4]), %arg7: f32 loc(fused<#di_lexical_block40>[#loc4])):
            scf.yield %arg6, %arg7 : i64, f32 loc(#loc94)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc94)
          scf.yield %10#1 : f32 loc(#loc94)
        } loc(#loc94)
        %6 = arith.index_cast %arg5 : i64 to index loc(#loc91)
        memref.store %5, %arg2[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc91)
        %7 = arith.addi %arg5, %c1_i64 : i64 loc(#loc89)
        %8 = arith.cmpi ne, %7, %2 : i64 loc(#loc92)
        scf.condition(%8) %7 : i64 loc(#loc86)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block33>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc86)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc86)
    } loc(#loc86)
    return loc(#loc74)
  } loc(#loc73)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  handshake.func @_Z14fir_filter_dsaPKfS0_Pfjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg3: i32 loc(fused<#di_subprogram6>[#loc11]), %arg4: i32 loc(fused<#di_subprogram6>[#loc11]), %arg5: i1 loc(fused<#di_subprogram6>[#loc11]), ...) -> i1 attributes {argNames = ["input", "coeffs", "output", "input_size", "num_taps", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=fir_filter"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc75)
    %1 = handshake.join %0 : none loc(#loc75)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = -1 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc90)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc87)
    %7 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc87)
    %9 = arith.index_cast %5 : i64 to index loc(#loc87)
    %10 = arith.index_cast %arg3 : i32 to index loc(#loc87)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc87)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc87)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc87)
    %12 = arith.index_cast %afterValue : index to i64 loc(#loc87)
    %13 = dataflow.invariant %afterCond, %7 : i1, i1 -> i1 loc(#loc95)
    %trueResult_0, %falseResult_1 = handshake.cond_br %13, %11 : none loc(#loc95)
    %14 = arith.trunci %12 : i64 to i32 loc(#loc2)
    %15 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc95)
    %16 = arith.index_cast %arg4 : i32 to index loc(#loc95)
    %index_2, %willContinue_3 = dataflow.stream %9, %15, %16 {step_op = "+=", stop_cond = "!="} loc(#loc95)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc95)
    %17 = dataflow.carry %willContinue_3, %4, %31 : i1, f32, f32 -> f32 loc(#loc95)
    %afterValue_6, %afterCond_7 = dataflow.gate %17, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc95)
    handshake.sink %afterCond_7 : i1 loc(#loc95)
    %18 = dataflow.carry %willContinue_3, %4, %32 : i1, f32, f32 -> f32 loc(#loc95)
    %afterValue_8, %afterCond_9 = dataflow.gate %18, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc95)
    handshake.sink %afterCond_9 : i1 loc(#loc95)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_3, %18 : f32 loc(#loc95)
    %19 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc95)
    %20 = arith.index_cast %afterValue_4 : index to i64 loc(#loc95)
    %21 = arith.trunci %20 : i64 to i32 loc(#loc99)
    %22 = dataflow.invariant %afterCond_5, %14 : i1, i32 -> i32 loc(#loc99)
    %23 = arith.subi %22, %21 : i32 loc(#loc99)
    %24 = arith.cmpi sgt, %23, %3 : i32 loc(#loc101)
    %dataResult, %addressResults = handshake.load [%afterValue_4] %37#0, %trueResult_17 : index, f32 loc(#loc104)
    %25 = arith.extui %23 : i32 to i64 loc(#loc104)
    %26 = arith.index_cast %25 : i64 to index loc(#loc104)
    %dataResult_12, %addressResults_13 = handshake.load [%26] %38#0, %trueResult_27 : index, f32 loc(#loc104)
    %27 = math.fma %dataResult, %dataResult_12, %afterValue_6 : f32 loc(#loc104)
    %28 = handshake.constant %19 {value = 0 : index} : index loc(#loc101)
    %29 = handshake.constant %19 {value = 1 : index} : index loc(#loc101)
    %30 = arith.select %24, %29, %28 : index loc(#loc101)
    %31 = handshake.mux %30 [%afterValue_6, %27] : index, f32 loc(#loc101)
    %32 = handshake.mux %30 [%afterValue_8, %27] : index, f32 loc(#loc101)
    %33 = handshake.constant %11 {value = 0 : index} : index loc(#loc95)
    %34 = handshake.constant %11 {value = 1 : index} : index loc(#loc95)
    %35 = arith.select %13, %34, %33 : index loc(#loc95)
    %36 = handshake.mux %35 [%falseResult_11, %4] : index, f32 loc(#loc95)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %36, %54 : index, f32 loc(#loc93)
    %37:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc75)
    %38:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (f32, none) loc(#loc75)
    %39 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc75)
    %40 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc87)
    %trueResult_15, %falseResult_16 = handshake.cond_br %13, %40 : none loc(#loc95)
    %41 = dataflow.carry %willContinue_3, %falseResult_16, %trueResult_19 : i1, none, none -> none loc(#loc95)
    %trueResult_17, %falseResult_18 = handshake.cond_br %24, %41 : none loc(#loc101)
    %42 = handshake.constant %41 {value = 0 : index} : index loc(#loc101)
    %43 = handshake.constant %41 {value = 1 : index} : index loc(#loc101)
    %44 = arith.select %24, %43, %42 : index loc(#loc101)
    %45 = handshake.mux %44 [%falseResult_18, %37#1] : index, none loc(#loc101)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue_3, %45 : none loc(#loc95)
    %46 = handshake.constant %40 {value = 0 : index} : index loc(#loc95)
    %47 = handshake.constant %40 {value = 1 : index} : index loc(#loc95)
    %48 = arith.select %13, %47, %46 : index loc(#loc95)
    %49 = handshake.mux %48 [%falseResult_20, %trueResult_15] : index, none loc(#loc95)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %49 : none loc(#loc87)
    %50 = handshake.constant %1 {value = 0 : index} : index loc(#loc87)
    %51 = handshake.constant %1 {value = 1 : index} : index loc(#loc87)
    %52 = arith.select %6, %51, %50 : index loc(#loc87)
    %53 = handshake.mux %52 [%falseResult_22, %trueResult] : index, none loc(#loc87)
    %54 = dataflow.carry %willContinue, %falseResult, %trueResult_23 : i1, none, none -> none loc(#loc87)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue, %39 : none loc(#loc87)
    %55 = handshake.mux %52 [%falseResult_24, %trueResult] : index, none loc(#loc87)
    %56 = dataflow.carry %willContinue, %falseResult, %trueResult_31 : i1, none, none -> none loc(#loc87)
    %trueResult_25, %falseResult_26 = handshake.cond_br %13, %56 : none loc(#loc95)
    %57 = dataflow.carry %willContinue_3, %falseResult_26, %trueResult_29 : i1, none, none -> none loc(#loc95)
    %trueResult_27, %falseResult_28 = handshake.cond_br %24, %57 : none loc(#loc101)
    %58 = handshake.constant %57 {value = 0 : index} : index loc(#loc101)
    %59 = handshake.constant %57 {value = 1 : index} : index loc(#loc101)
    %60 = arith.select %24, %59, %58 : index loc(#loc101)
    %61 = handshake.mux %60 [%falseResult_28, %38#1] : index, none loc(#loc101)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue_3, %61 : none loc(#loc95)
    %62 = handshake.constant %56 {value = 0 : index} : index loc(#loc95)
    %63 = handshake.constant %56 {value = 1 : index} : index loc(#loc95)
    %64 = arith.select %13, %63, %62 : index loc(#loc95)
    %65 = handshake.mux %64 [%falseResult_30, %trueResult_25] : index, none loc(#loc95)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue, %65 : none loc(#loc87)
    %66 = handshake.mux %52 [%falseResult_32, %trueResult] : index, none loc(#loc87)
    %67 = handshake.join %53, %55, %66 : none, none, none loc(#loc75)
    %68 = handshake.constant %67 {value = true} : i1 loc(#loc75)
    handshake.return %68 : i1 loc(#loc75)
  } loc(#loc75)
  handshake.func @_Z14fir_filter_dsaPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc11]), %arg3: i32 loc(fused<#di_subprogram6>[#loc11]), %arg4: i32 loc(fused<#di_subprogram6>[#loc11]), %arg5: none loc(fused<#di_subprogram6>[#loc11]), ...) -> none attributes {argNames = ["input", "coeffs", "output", "input_size", "num_taps", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=fir_filter"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc75)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = -1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc90)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc87)
    %6 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc87)
    %8 = arith.index_cast %4 : i64 to index loc(#loc87)
    %9 = arith.index_cast %arg3 : i32 to index loc(#loc87)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc87)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc87)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc87)
    %11 = arith.index_cast %afterValue : index to i64 loc(#loc87)
    %12 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc95)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %10 : none loc(#loc95)
    %13 = arith.trunci %11 : i64 to i32 loc(#loc2)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc95)
    %15 = arith.index_cast %arg4 : i32 to index loc(#loc95)
    %index_2, %willContinue_3 = dataflow.stream %8, %14, %15 {step_op = "+=", stop_cond = "!="} loc(#loc95)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc95)
    %16 = dataflow.carry %willContinue_3, %3, %30 : i1, f32, f32 -> f32 loc(#loc95)
    %afterValue_6, %afterCond_7 = dataflow.gate %16, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc95)
    handshake.sink %afterCond_7 : i1 loc(#loc95)
    %17 = dataflow.carry %willContinue_3, %3, %31 : i1, f32, f32 -> f32 loc(#loc95)
    %afterValue_8, %afterCond_9 = dataflow.gate %17, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc95)
    handshake.sink %afterCond_9 : i1 loc(#loc95)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_3, %17 : f32 loc(#loc95)
    %18 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc95)
    %19 = arith.index_cast %afterValue_4 : index to i64 loc(#loc95)
    %20 = arith.trunci %19 : i64 to i32 loc(#loc99)
    %21 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc99)
    %22 = arith.subi %21, %20 : i32 loc(#loc99)
    %23 = arith.cmpi sgt, %22, %2 : i32 loc(#loc101)
    %dataResult, %addressResults = handshake.load [%afterValue_4] %36#0, %trueResult_17 : index, f32 loc(#loc104)
    %24 = arith.extui %22 : i32 to i64 loc(#loc104)
    %25 = arith.index_cast %24 : i64 to index loc(#loc104)
    %dataResult_12, %addressResults_13 = handshake.load [%25] %37#0, %trueResult_27 : index, f32 loc(#loc104)
    %26 = math.fma %dataResult, %dataResult_12, %afterValue_6 : f32 loc(#loc104)
    %27 = handshake.constant %18 {value = 0 : index} : index loc(#loc101)
    %28 = handshake.constant %18 {value = 1 : index} : index loc(#loc101)
    %29 = arith.select %23, %28, %27 : index loc(#loc101)
    %30 = handshake.mux %29 [%afterValue_6, %26] : index, f32 loc(#loc101)
    %31 = handshake.mux %29 [%afterValue_8, %26] : index, f32 loc(#loc101)
    %32 = handshake.constant %10 {value = 0 : index} : index loc(#loc95)
    %33 = handshake.constant %10 {value = 1 : index} : index loc(#loc95)
    %34 = arith.select %12, %33, %32 : index loc(#loc95)
    %35 = handshake.mux %34 [%falseResult_11, %3] : index, f32 loc(#loc95)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %35, %53 : index, f32 loc(#loc93)
    %36:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc75)
    %37:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (f32, none) loc(#loc75)
    %38 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc75)
    %39 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc87)
    %trueResult_15, %falseResult_16 = handshake.cond_br %12, %39 : none loc(#loc95)
    %40 = dataflow.carry %willContinue_3, %falseResult_16, %trueResult_19 : i1, none, none -> none loc(#loc95)
    %trueResult_17, %falseResult_18 = handshake.cond_br %23, %40 : none loc(#loc101)
    %41 = handshake.constant %40 {value = 0 : index} : index loc(#loc101)
    %42 = handshake.constant %40 {value = 1 : index} : index loc(#loc101)
    %43 = arith.select %23, %42, %41 : index loc(#loc101)
    %44 = handshake.mux %43 [%falseResult_18, %36#1] : index, none loc(#loc101)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue_3, %44 : none loc(#loc95)
    %45 = handshake.constant %39 {value = 0 : index} : index loc(#loc95)
    %46 = handshake.constant %39 {value = 1 : index} : index loc(#loc95)
    %47 = arith.select %12, %46, %45 : index loc(#loc95)
    %48 = handshake.mux %47 [%falseResult_20, %trueResult_15] : index, none loc(#loc95)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %48 : none loc(#loc87)
    %49 = handshake.constant %0 {value = 0 : index} : index loc(#loc87)
    %50 = handshake.constant %0 {value = 1 : index} : index loc(#loc87)
    %51 = arith.select %5, %50, %49 : index loc(#loc87)
    %52 = handshake.mux %51 [%falseResult_22, %trueResult] : index, none loc(#loc87)
    %53 = dataflow.carry %willContinue, %falseResult, %trueResult_23 : i1, none, none -> none loc(#loc87)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue, %38 : none loc(#loc87)
    %54 = handshake.mux %51 [%falseResult_24, %trueResult] : index, none loc(#loc87)
    %55 = dataflow.carry %willContinue, %falseResult, %trueResult_31 : i1, none, none -> none loc(#loc87)
    %trueResult_25, %falseResult_26 = handshake.cond_br %12, %55 : none loc(#loc95)
    %56 = dataflow.carry %willContinue_3, %falseResult_26, %trueResult_29 : i1, none, none -> none loc(#loc95)
    %trueResult_27, %falseResult_28 = handshake.cond_br %23, %56 : none loc(#loc101)
    %57 = handshake.constant %56 {value = 0 : index} : index loc(#loc101)
    %58 = handshake.constant %56 {value = 1 : index} : index loc(#loc101)
    %59 = arith.select %23, %58, %57 : index loc(#loc101)
    %60 = handshake.mux %59 [%falseResult_28, %37#1] : index, none loc(#loc101)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue_3, %60 : none loc(#loc95)
    %61 = handshake.constant %55 {value = 0 : index} : index loc(#loc95)
    %62 = handshake.constant %55 {value = 1 : index} : index loc(#loc95)
    %63 = arith.select %12, %62, %61 : index loc(#loc95)
    %64 = handshake.mux %63 [%falseResult_30, %trueResult_25] : index, none loc(#loc95)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue, %64 : none loc(#loc87)
    %65 = handshake.mux %51 [%falseResult_32, %trueResult] : index, none loc(#loc87)
    %66 = handshake.join %52, %54, %65 : none, none, none loc(#loc75)
    handshake.return %66 : none loc(#loc76)
  } loc(#loc75)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc46)
    %false = arith.constant false loc(#loc46)
    %0 = seq.const_clock  low loc(#loc46)
    %1 = ub.poison : i64 loc(#loc46)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c3 = arith.constant 3 : index loc(#loc47)
    %c8 = arith.constant 8 : index loc(#loc48)
    %c1 = arith.constant 1 : index loc(#loc2)
    %cst = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %c3_i64 = arith.constant 3 : i64 loc(#loc2)
    %c2_i64 = arith.constant 2 : i64 loc(#loc2)
    %c8_i64 = arith.constant 8 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.input : memref<8xf32> loc(#loc2)
    %3 = memref.get_global @__const.main.coeffs : memref<3xf32> loc(#loc2)
    %4 = memref.get_global @str : memref<45xi8> loc(#loc2)
    %5 = memref.get_global @".str.1.4" : memref<14xi8> loc(#loc2)
    %6 = memref.get_global @".str.4" : memref<1xi8> loc(#loc2)
    %7 = memref.get_global @".str.3.5" : memref<3xi8> loc(#loc2)
    %8 = memref.get_global @".str.2.6" : memref<7xi8> loc(#loc2)
    %9 = memref.get_global @str.15 : memref<2xi8> loc(#loc2)
    %10 = memref.get_global @".str.6" : memref<14xi8> loc(#loc2)
    %11 = memref.get_global @".str.7" : memref<7xi8> loc(#loc2)
    %12 = memref.get_global @".str.8" : memref<14xi8> loc(#loc2)
    %13 = memref.get_global @".str.9" : memref<14xi8> loc(#loc2)
    %14 = memref.get_global @str.16 : memref<26xi8> loc(#loc2)
    %15 = memref.get_global @str.17 : memref<29xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<3xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<8xf32> loc(#loc2)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %50 = memref.load %2[%arg0] : memref<8xf32> loc(#loc48)
      memref.store %50, %alloca[%arg0] : memref<8xf32> loc(#loc48)
    } loc(#loc48)
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %50 = memref.load %3[%arg0] : memref<3xf32> loc(#loc47)
      memref.store %50, %alloca_0[%arg0] : memref<3xf32> loc(#loc47)
    } loc(#loc47)
    %cast = memref.cast %alloca : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    %cast_3 = memref.cast %alloca_0 : memref<3xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    %cast_4 = memref.cast %alloca_1 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    call @_Z14fir_filter_cpuPKfS0_Pfjj(%cast, %cast_3, %cast_4, %c8_i32, %c3_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc49)
    %cast_5 = memref.cast %alloca_2 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc50)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc50)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc50)
    %16 = handshake.esi_instance @_Z14fir_filter_dsaPKfS0_Pfjj_esi "_Z14fir_filter_dsaPKfS0_Pfjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc50)
    %rawOutput, %valid = esi.unwrap.vr %16, %true : i1 loc(#loc50)
    %intptr = memref.extract_aligned_pointer_as_index %4 : memref<45xi8> -> index loc(#loc51)
    %17 = arith.index_cast %intptr : index to i64 loc(#loc51)
    %18 = llvm.inttoptr %17 : i64 to !llvm.ptr loc(#loc51)
    %19 = llvm.call @puts(%18) : (!llvm.ptr) -> i32 loc(#loc51)
    %intptr_16 = memref.extract_aligned_pointer_as_index %5 : memref<14xi8> -> index loc(#loc52)
    %20 = arith.index_cast %intptr_16 : index to i64 loc(#loc52)
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr loc(#loc52)
    %22 = llvm.call @printf(%21) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc52)
    %cast_17 = memref.cast %6 : memref<1xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc77)
    %cast_18 = memref.cast %7 : memref<3xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc77)
    %intptr_19 = memref.extract_aligned_pointer_as_index %8 : memref<7xi8> -> index loc(#loc77)
    %23 = arith.index_cast %intptr_19 : index to i64 loc(#loc77)
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr loc(#loc77)
    %25 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %50 = arith.index_cast %arg0 : i64 to index loc(#loc77)
      %51 = memref.load %alloca[%50] : memref<8xf32> loc(#loc77)
      %52 = arith.extf %51 : f32 to f64 loc(#loc77)
      %53 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc77)
      %54 = arith.select %53, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc77)
      %intptr_27 = memref.extract_aligned_pointer_as_index %54 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc77)
      %55 = arith.index_cast %intptr_27 : index to i64 loc(#loc77)
      %56 = llvm.inttoptr %55 : i64 to !llvm.ptr loc(#loc77)
      %57 = llvm.call @printf(%24, %52, %56) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc77)
      %58 = arith.addi %arg0, %c1_i64 : i64 loc(#loc68)
      %59 = arith.cmpi ne, %58, %c8_i64 : i64 loc(#loc78)
      scf.condition(%59) %58 : i64 loc(#loc62)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block17>[#loc27])):
      scf.yield %arg0 : i64 loc(#loc62)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc62)
    %intptr_20 = memref.extract_aligned_pointer_as_index %9 : memref<2xi8> -> index loc(#loc53)
    %26 = arith.index_cast %intptr_20 : index to i64 loc(#loc53)
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr loc(#loc53)
    %28 = llvm.call @puts(%27) : (!llvm.ptr) -> i32 loc(#loc53)
    %intptr_21 = memref.extract_aligned_pointer_as_index %10 : memref<14xi8> -> index loc(#loc54)
    %29 = arith.index_cast %intptr_21 : index to i64 loc(#loc54)
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr loc(#loc54)
    %31 = llvm.call @printf(%30) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc54)
    %intptr_22 = memref.extract_aligned_pointer_as_index %11 : memref<7xi8> -> index loc(#loc79)
    %32 = arith.index_cast %intptr_22 : index to i64 loc(#loc79)
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr loc(#loc79)
    %34 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %50 = arith.index_cast %arg0 : i64 to index loc(#loc79)
      %51 = memref.load %alloca_0[%50] : memref<3xf32> loc(#loc79)
      %52 = arith.extf %51 : f32 to f64 loc(#loc79)
      %53 = arith.cmpi eq, %arg0, %c2_i64 : i64 loc(#loc79)
      %54 = arith.select %53, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc79)
      %intptr_27 = memref.extract_aligned_pointer_as_index %54 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc79)
      %55 = arith.index_cast %intptr_27 : index to i64 loc(#loc79)
      %56 = llvm.inttoptr %55 : i64 to !llvm.ptr loc(#loc79)
      %57 = llvm.call @printf(%33, %52, %56) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc79)
      %58 = arith.addi %arg0, %c1_i64 : i64 loc(#loc69)
      %59 = arith.cmpi ne, %58, %c3_i64 : i64 loc(#loc80)
      scf.condition(%59) %58 : i64 loc(#loc63)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block18>[#loc31])):
      scf.yield %arg0 : i64 loc(#loc63)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc63)
    %35 = llvm.call @puts(%27) : (!llvm.ptr) -> i32 loc(#loc55)
    %intptr_23 = memref.extract_aligned_pointer_as_index %12 : memref<14xi8> -> index loc(#loc56)
    %36 = arith.index_cast %intptr_23 : index to i64 loc(#loc56)
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr loc(#loc56)
    %38 = llvm.call @printf(%37) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc56)
    %39 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %50 = arith.index_cast %arg0 : i64 to index loc(#loc81)
      %51 = memref.load %alloca_1[%50] : memref<8xf32> loc(#loc81)
      %52 = arith.extf %51 : f32 to f64 loc(#loc81)
      %53 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc81)
      %54 = arith.select %53, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc81)
      %intptr_27 = memref.extract_aligned_pointer_as_index %54 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc81)
      %55 = arith.index_cast %intptr_27 : index to i64 loc(#loc81)
      %56 = llvm.inttoptr %55 : i64 to !llvm.ptr loc(#loc81)
      %57 = llvm.call @printf(%33, %52, %56) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc81)
      %58 = arith.addi %arg0, %c1_i64 : i64 loc(#loc70)
      %59 = arith.cmpi ne, %58, %c8_i64 : i64 loc(#loc82)
      scf.condition(%59) %58 : i64 loc(#loc64)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block19>[#loc34])):
      scf.yield %arg0 : i64 loc(#loc64)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc64)
    %40 = llvm.call @puts(%27) : (!llvm.ptr) -> i32 loc(#loc57)
    %intptr_24 = memref.extract_aligned_pointer_as_index %13 : memref<14xi8> -> index loc(#loc58)
    %41 = arith.index_cast %intptr_24 : index to i64 loc(#loc58)
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr loc(#loc58)
    %43 = llvm.call @printf(%42) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc58)
    %44 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %50 = arith.index_cast %arg0 : i64 to index loc(#loc83)
      %51 = memref.load %alloca_2[%50] : memref<8xf32> loc(#loc83)
      %52 = arith.extf %51 : f32 to f64 loc(#loc83)
      %53 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc83)
      %54 = arith.select %53, %cast_17, %cast_18 : memref<?xi8, strided<[1], offset: ?>> loc(#loc83)
      %intptr_27 = memref.extract_aligned_pointer_as_index %54 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc83)
      %55 = arith.index_cast %intptr_27 : index to i64 loc(#loc83)
      %56 = llvm.inttoptr %55 : i64 to !llvm.ptr loc(#loc83)
      %57 = llvm.call @printf(%33, %52, %56) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc83)
      %58 = arith.addi %arg0, %c1_i64 : i64 loc(#loc71)
      %59 = arith.cmpi ne, %58, %c8_i64 : i64 loc(#loc84)
      scf.condition(%59) %58 : i64 loc(#loc65)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block20>[#loc38])):
      scf.yield %arg0 : i64 loc(#loc65)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc65)
    %45 = llvm.call @puts(%27) : (!llvm.ptr) -> i32 loc(#loc59)
    %cast_25 = memref.cast %14 : memref<26xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc88)
    %46:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, memref<?xi8, strided<[1], offset: ?>>, i32) {
      %50 = arith.index_cast %arg0 : i64 to index loc(#loc88)
      %51 = memref.load %alloca_1[%50] : memref<8xf32> loc(#loc88)
      %52 = memref.load %alloca_2[%50] : memref<8xf32> loc(#loc88)
      %53 = arith.subf %51, %52 : f32 loc(#loc88)
      %54 = math.absf %53 : f32 loc(#loc88)
      %55 = arith.cmpf ogt, %54, %cst : f32 loc(#loc88)
      %56 = arith.extui %55 : i1 to i32 loc(#loc88)
      %57:3 = scf.if %55 -> (i64, memref<?xi8, strided<[1], offset: ?>>, i32) {
        scf.yield %1, %cast_25, %c0_i32 : i64, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc88)
      } else {
        %59 = arith.addi %arg0, %c1_i64 : i64 loc(#loc72)
        %cast_27 = memref.cast %15 : memref<29xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc66)
        %60 = arith.cmpi ne, %59, %c8_i64 : i64 loc(#loc85)
        %61 = arith.extui %60 : i1 to i32 loc(#loc66)
        scf.yield %59, %cast_27, %61 : i64, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc88)
      } loc(#loc88)
      %58 = arith.trunci %57#2 : i32 to i1 loc(#loc66)
      scf.condition(%58) %57#0, %57#1, %56 : i64, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc66)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block21>[#loc42]), %arg1: memref<?xi8, strided<[1], offset: ?>> loc(fused<#di_lexical_block21>[#loc42]), %arg2: i32 loc(fused<#di_lexical_block21>[#loc42])):
      scf.yield %arg0 : i64 loc(#loc66)
    } loc(#loc66)
    %intptr_26 = memref.extract_aligned_pointer_as_index %46#1 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc67)
    %47 = arith.index_cast %intptr_26 : index to i64 loc(#loc67)
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr loc(#loc67)
    %49 = llvm.call @puts(%48) : (!llvm.ptr) -> i32 loc(#loc67)
    return %46#2 : i32 loc(#loc60)
  } loc(#loc46)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc61)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
} loc(#loc)
#di_basic_type4 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file2 = #llvm.di_file<"/usr/include/stdio.h" in "">
#loc = loc("tests/app/fir_filter/fir_filter.cpp":0:0)
#loc2 = loc(unknown)
#loc5 = loc("tests/app/fir_filter/fir_filter.cpp":14:0)
#loc6 = loc("tests/app/fir_filter/fir_filter.cpp":15:0)
#loc7 = loc("tests/app/fir_filter/fir_filter.cpp":16:0)
#loc8 = loc("tests/app/fir_filter/fir_filter.cpp":17:0)
#loc9 = loc("tests/app/fir_filter/fir_filter.cpp":19:0)
#loc10 = loc("tests/app/fir_filter/fir_filter.cpp":21:0)
#loc12 = loc("tests/app/fir_filter/fir_filter.cpp":31:0)
#loc13 = loc("tests/app/fir_filter/fir_filter.cpp":36:0)
#loc14 = loc("tests/app/fir_filter/fir_filter.cpp":37:0)
#loc15 = loc("tests/app/fir_filter/fir_filter.cpp":38:0)
#loc16 = loc("tests/app/fir_filter/fir_filter.cpp":39:0)
#loc17 = loc("tests/app/fir_filter/fir_filter.cpp":42:0)
#loc18 = loc("tests/app/fir_filter/fir_filter.cpp":44:0)
#loc19 = loc("tests/app/fir_filter/main.cpp":7:0)
#loc20 = loc("tests/app/fir_filter/main.cpp":15:0)
#loc21 = loc("tests/app/fir_filter/main.cpp":12:0)
#loc22 = loc("tests/app/fir_filter/main.cpp":22:0)
#loc23 = loc("tests/app/fir_filter/main.cpp":23:0)
#loc24 = loc("tests/app/fir_filter/main.cpp":26:0)
#loc25 = loc("tests/app/fir_filter/main.cpp":27:0)
#loc26 = loc("tests/app/fir_filter/main.cpp":29:0)
#loc28 = loc("tests/app/fir_filter/main.cpp":31:0)
#loc29 = loc("tests/app/fir_filter/main.cpp":33:0)
#loc30 = loc("tests/app/fir_filter/main.cpp":35:0)
#loc32 = loc("tests/app/fir_filter/main.cpp":37:0)
#loc33 = loc("tests/app/fir_filter/main.cpp":39:0)
#loc35 = loc("tests/app/fir_filter/main.cpp":41:0)
#loc36 = loc("tests/app/fir_filter/main.cpp":43:0)
#loc37 = loc("tests/app/fir_filter/main.cpp":45:0)
#loc39 = loc("tests/app/fir_filter/main.cpp":47:0)
#loc40 = loc("tests/app/fir_filter/main.cpp":49:0)
#loc41 = loc("tests/app/fir_filter/main.cpp":54:0)
#loc43 = loc("tests/app/fir_filter/main.cpp":0:0)
#loc44 = loc("tests/app/fir_filter/main.cpp":67:0)
#loc45 = loc("/usr/include/stdio.h":363:0)
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type4>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type12 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type9>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_basic_type2, #di_derived_type12, #di_null_type>
#di_subprogram4 = #llvm.di_subprogram<scope = #di_file2, name = "printf", file = #di_file2, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type2>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 60>
#loc46 = loc(fused<#di_subprogram3>[#loc19])
#loc47 = loc(fused<#di_subprogram3>[#loc20])
#loc48 = loc(fused<#di_subprogram3>[#loc21])
#loc49 = loc(fused<#di_subprogram3>[#loc22])
#loc50 = loc(fused<#di_subprogram3>[#loc23])
#loc51 = loc(fused<#di_subprogram3>[#loc24])
#loc52 = loc(fused<#di_subprogram3>[#loc25])
#loc53 = loc(fused<#di_subprogram3>[#loc28])
#loc54 = loc(fused<#di_subprogram3>[#loc29])
#loc55 = loc(fused<#di_subprogram3>[#loc32])
#loc56 = loc(fused<#di_subprogram3>[#loc33])
#loc57 = loc(fused<#di_subprogram3>[#loc36])
#loc58 = loc(fused<#di_subprogram3>[#loc37])
#loc59 = loc(fused<#di_subprogram3>[#loc40])
#loc60 = loc(fused<#di_subprogram3>[#loc44])
#loc61 = loc(fused<#di_subprogram4>[#loc45])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 28>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 34>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 40>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 46>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 53>
#loc67 = loc(fused<#di_lexical_block22>[#loc43])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 28>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 34>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 40>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 46>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 53>
#loc68 = loc(fused<#di_lexical_block23>[#loc27])
#loc69 = loc(fused<#di_lexical_block24>[#loc31])
#loc70 = loc(fused<#di_lexical_block25>[#loc34])
#loc71 = loc(fused<#di_lexical_block26>[#loc38])
#loc72 = loc(fused<#di_lexical_block27>[#loc42])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 31>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file1, line = 54>
#loc74 = loc(fused<#di_subprogram5>[#loc10])
#loc76 = loc(fused<#di_subprogram6>[#loc18])
#loc77 = loc(fused<#di_lexical_block28>[#loc26])
#loc78 = loc(fused[#loc62, #loc68])
#loc79 = loc(fused<#di_lexical_block29>[#loc30])
#loc80 = loc(fused[#loc63, #loc69])
#loc81 = loc(fused<#di_lexical_block30>[#loc35])
#loc82 = loc(fused[#loc64, #loc70])
#loc83 = loc(fused<#di_lexical_block31>[#loc39])
#loc84 = loc(fused[#loc65, #loc71])
#loc85 = loc(fused[#loc66, #loc72])
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 31>
#loc87 = loc(fused<#di_lexical_block34>[#loc12])
#loc88 = loc(fused<#di_lexical_block35>[#loc41])
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file, line = 31>
#loc89 = loc(fused<#di_lexical_block36>[#loc3])
#loc90 = loc(fused<#di_lexical_block37>[#loc12])
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file, line = 36>
#loc91 = loc(fused<#di_lexical_block38>[#loc9])
#loc92 = loc(fused[#loc86, #loc89])
#loc93 = loc(fused<#di_lexical_block39>[#loc17])
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file, line = 13>
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file, line = 36>
#loc95 = loc(fused<#di_lexical_block41>[#loc13])
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_lexical_block42, file = #di_file, line = 13>
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_lexical_block43, file = #di_file, line = 36>
#loc96 = loc(fused<#di_lexical_block42>[#loc4])
#di_lexical_block46 = #llvm.di_lexical_block<scope = #di_lexical_block44, file = #di_file, line = 15>
#di_lexical_block47 = #llvm.di_lexical_block<scope = #di_lexical_block45, file = #di_file, line = 38>
#loc97 = loc(fused<#di_lexical_block44>[#loc5])
#loc98 = loc(fused[#loc94, #loc96])
#loc99 = loc(fused<#di_lexical_block45>[#loc14])
#di_lexical_block48 = #llvm.di_lexical_block<scope = #di_lexical_block46, file = #di_file, line = 15>
#di_lexical_block49 = #llvm.di_lexical_block<scope = #di_lexical_block47, file = #di_file, line = 38>
#loc100 = loc(fused<#di_lexical_block46>[#loc6])
#loc101 = loc(fused<#di_lexical_block47>[#loc15])
#loc102 = loc(fused<#di_lexical_block48>[#loc7])
#loc103 = loc(fused<#di_lexical_block48>[#loc8])
#loc104 = loc(fused<#di_lexical_block49>[#loc16])
