#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/dotprod/dotprod.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/dotprod/kernels.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file2 = #llvm.di_file<"tests/app/dotprod/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[3]<>, isRecSelf = true>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[4]<>, isRecSelf = true>
#loc1 = loc("tests/app/dotprod/dotprod.cpp":20:0)
#loc3 = loc("tests/app/dotprod/dotprod.cpp":22:0)
#loc6 = loc("tests/app/dotprod/dotprod.cpp":42:0)
#loc13 = loc("tests/app/dotprod/kernels.cpp":19:0)
#loc15 = loc("tests/app/dotprod/kernels.cpp":34:0)
#loc25 = loc("tests/app/dotprod/main.cpp":29:0)
#loc28 = loc("tests/app/dotprod/main.cpp":35:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[5]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[6]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit2 = #llvm.di_compile_unit<id = distinct[7]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file2, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 256, elements = #llvm.di_subrange<count = 8 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 22>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 39>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 22>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file2, line = 29>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file2, line = 35>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "n", file = #di_file, line = 20, arg = 3, type = #di_basic_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "sum", file = #di_file, line = 21, type = #di_basic_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "n", file = #di_file, line = 43, arg = 3, type = #di_basic_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram1, name = "result", file = #di_file, line = 47, type = #di_basic_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "n", file = #di_file1, line = 35, arg = 3, type = #di_basic_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "sum", file = #di_file1, line = 37, type = #di_basic_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram3, name = "n", file = #di_file1, line = 20, arg = 4, type = #di_basic_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram4, name = "expected", file = #di_file2, line = 19, type = #di_basic_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram4, name = "ref_result", file = #di_file2, line = 22, type = #di_basic_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram4, name = "accel_result", file = #di_file2, line = 25, type = #di_basic_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram4, name = "ref_ok", file = #di_file2, line = 44, type = #di_basic_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram4, name = "accel_ok", file = #di_file2, line = 45, type = #di_basic_type2>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 22, type = #di_basic_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "products", file = #di_file, line = 45, type = #di_composite_type>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file1, line = 39, type = #di_basic_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 22, type = #di_basic_type1>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram4, name = "a", file = #di_file2, line = 14, type = #di_composite_type1>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram4, name = "b", file = #di_file2, line = 15, type = #di_composite_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file2, line = 29, type = #di_basic_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file2, line = 35, type = #di_basic_type1>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram4, name = "epsilon", file = #di_file2, line = 43, type = #di_derived_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, line = 20, arg = 1, type = #di_derived_type3>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram, name = "b", file = #di_file, line = 20, arg = 2, type = #di_derived_type3>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "result", file = #di_file1, line = 34, arg = 2, type = #di_derived_type4>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram3, name = "products", file = #di_file1, line = 20, arg = 3, type = #di_derived_type4>
#di_subprogram7 = #llvm.di_subprogram<recId = distinct[4]<>, id = distinct[8]<>, compileUnit = #di_compile_unit2, scope = #di_file2, name = "main", file = #di_file2, line = 13, scopeLine = 13, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable10, #di_local_variable11>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_basic_type, #di_derived_type3, #di_derived_type3, #di_basic_type1>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram7, file = #di_file2, line = 29>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram7, file = #di_file2, line = 35>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram1, name = "a", file = #di_file, line = 42, arg = 1, type = #di_derived_type6>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "b", file = #di_file, line = 42, arg = 2, type = #di_derived_type6>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram2, name = "data", file = #di_file1, line = 34, arg = 1, type = #di_derived_type6>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram3, name = "a", file = #di_file1, line = 19, arg = 1, type = #di_derived_type6>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram3, name = "b", file = #di_file1, line = 19, arg = 2, type = #di_derived_type6>
#di_subprogram8 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[9]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "dotprod_cpu", linkageName = "_Z11dotprod_cpuPKfS0_i", file = #di_file, line = 20, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable22, #di_local_variable23, #di_local_variable, #di_local_variable1, #di_local_variable13>
#di_subroutine_type3 = #llvm.di_subroutine_type<types = #di_basic_type, #di_derived_type6, #di_derived_type6, #di_basic_type1>
#di_subroutine_type4 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type4, #di_basic_type1>
#di_subroutine_type5 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_basic_type1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram8, file = #di_file, line = 22>
#di_subprogram9 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[10]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "dotprod_dsa", linkageName = "_Z11dotprod_dsaPKfS0_i", file = #di_file, line = 42, scopeLine = 43, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type3, retainedNodes = #di_local_variable26, #di_local_variable27, #di_local_variable2, #di_local_variable14, #di_local_variable3>
#di_subprogram10 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[11]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecsum_kernel", linkageName = "_Z13vecsum_kernelPKfPfi", file = #di_file1, line = 34, scopeLine = 35, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type4, retainedNodes = #di_local_variable28, #di_local_variable24, #di_local_variable4, #di_local_variable5, #di_local_variable15>
#di_subprogram11 = #llvm.di_subprogram<recId = distinct[3]<>, id = distinct[12]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecmul_kernel", linkageName = "_Z13vecmul_kernelPKfS0_Pfi", file = #di_file1, line = 19, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type5, retainedNodes = #di_local_variable29, #di_local_variable30, #di_local_variable25, #di_local_variable6, #di_local_variable16>
#loc58 = loc(fused<#di_subprogram8>[#loc1])
#loc60 = loc(fused<#di_lexical_block5>[#loc25])
#loc61 = loc(fused<#di_lexical_block6>[#loc28])
#loc65 = loc(fused<#di_lexical_block8>[#loc3])
#loc66 = loc(fused<#di_subprogram9>[#loc6])
#loc69 = loc(fused<#di_subprogram11>[#loc13])
#loc71 = loc(fused<#di_subprogram10>[#loc15])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<30xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 100, 111, 116, 112, 114, 111, 100, 47, 100, 111, 116, 112, 114, 111, 100, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 100, 111, 116, 112, 114, 111, 100, 0]> loc(#loc)
  memref.global constant @".str.10" : memref<14xi8> = dense<[108, 111, 111, 109, 46, 114, 101, 100, 117, 99, 101, 61, 43, 0]> loc(#loc)
  memref.global constant @".str.1.7" : memref<30xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 100, 111, 116, 112, 114, 111, 100, 47, 107, 101, 114, 110, 101, 108, 115, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2.6" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<18xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 118, 101, 99, 109, 117, 108, 0]> loc(#loc)
  memref.global constant @".str.4" : memref<18xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 118, 101, 99, 115, 117, 109, 0]> loc(#loc)
  memref.global constant @__const.main.a : memref<8xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.b : memref<8xf32> = dense<[5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @".str.1.11" : memref<6xi8> = dense<[97, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.2.14" : memref<7xi8> = dense<[37, 46, 49, 102, 37, 115, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.3.13" : memref<3xi8> = dense<[44, 32, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4.12" : memref<1xi8> = dense<0> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.6" : memref<6xi8> = dense<[98, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.7" : memref<30xi8> = dense<[97, 32, 46, 32, 98, 32, 61, 32, 37, 46, 49, 102, 32, 40, 101, 120, 112, 101, 99, 116, 101, 100, 32, 37, 46, 49, 102, 41, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.9" : memref<48xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 82, 101, 102, 101, 114, 101, 110, 99, 101, 32, 114, 101, 115, 117, 108, 116, 32, 37, 46, 49, 102, 32, 33, 61, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 37, 46, 49, 102, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.10.15" : memref<44xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 65, 99, 99, 101, 108, 32, 114, 101, 115, 117, 108, 116, 32, 37, 46, 49, 102, 32, 33, 61, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 37, 46, 49, 102, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str : memref<21xi8> = dense<[68, 111, 116, 32, 80, 114, 111, 100, 117, 99, 116, 32, 82, 101, 115, 117, 108, 116, 115, 58, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.12 : memref<2xi8> = dense<[93, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.13 : memref<29xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 65, 108, 108, 32, 114, 101, 115, 117, 108, 116, 115, 32, 99, 111, 114, 114, 101, 99, 116, 33, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z11dotprod_cpuPKfS0_i(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc1]), %arg2: i32 loc(fused<#di_subprogram8>[#loc1])) -> f32 {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi sgt, %arg2, %c0_i32 : i32 loc(#loc77)
    %1 = scf.if %0 -> (f32) {
      %2 = arith.extui %arg2 : i32 to i64 loc(#loc77)
      %3:2 = scf.while (%arg3 = %c0_i64, %arg4 = %cst) : (i64, f32) -> (i64, f32) {
        %4 = arith.index_cast %arg3 : i64 to index loc(#loc86)
        %5 = memref.load %arg0[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc86)
        %6 = memref.load %arg1[%4] : memref<?xf32, strided<[1], offset: ?>> loc(#loc86)
        %7 = math.fma %5, %6, %arg4 : f32 loc(#loc86)
        %8 = arith.addi %arg3, %c1_i64 : i64 loc(#loc77)
        %9 = arith.cmpi ne, %8, %2 : i64 loc(#loc87)
        scf.condition(%9) %8, %7 : i64, f32 loc(#loc65)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block8>[#loc3]), %arg4: f32 loc(fused<#di_lexical_block8>[#loc3])):
        scf.yield %arg3, %arg4 : i64, f32 loc(#loc65)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc65)
      scf.yield %3#1 : f32 loc(#loc65)
    } else {
      scf.yield %cst : f32 loc(#loc65)
    } loc(#loc65)
    return %1 : f32 loc(#loc59)
  } loc(#loc58)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  handshake.func @_Z11dotprod_dsaPKfS0_i_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram9>[#loc6]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram9>[#loc6]), %arg2: i32 loc(fused<#di_subprogram9>[#loc6]), %arg3: i1 loc(fused<#di_subprogram9>[#loc6]), ...) -> (f32, i1) attributes {argNames = ["a", "b", "n", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=dotprod"], resNames = ["result", "done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc66)
    %1 = handshake.join %0 : none loc(#loc66)
    %2 = handshake.constant %1 {value = 0 : index} : index loc(#loc78)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi sgt, %arg2, %3 : i32 loc(#loc88)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc79)
    handshake.sink %falseResult : none loc(#loc79)
    %6 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc79)
    %7 = arith.index_cast %4 : i64 to index loc(#loc79)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc79)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc79)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc79)
    %dataResult, %addressResults = handshake.load [%afterValue] %18#0, %20 : index, f32 loc(#loc92)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %17#0, %29 : index, f32 loc(#loc92)
    %9 = arith.mulf %dataResult, %dataResult_0 : f32 loc(#loc92)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %9, %22 : index, f32 loc(#loc92)
    %10 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    handshake.sink %falseResult : none loc(#loc78)
    %index_3, %willContinue_4 = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc78)
    %afterValue_5, %afterCond_6 = dataflow.gate %index_3, %willContinue_4 : index, i1 -> index, i1 loc(#loc78)
    %11 = dataflow.carry %willContinue_4, %10, %12 : i1, f32, f32 -> f32 loc(#loc78)
    %afterValue_7, %afterCond_8 = dataflow.gate %11, %willContinue_4 : f32, i1 -> f32, i1 loc(#loc78)
    handshake.sink %afterCond_8 : i1 loc(#loc78)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue_4, %11 : f32 loc(#loc78)
    %dataResult_11, %addressResults_12 = handshake.load [%afterValue_5] %19#0, %24 : index, f32 loc(#loc93)
    %12 = arith.addf %dataResult_11, %afterValue_7 : f32 loc(#loc93)
    %13 = handshake.constant %1 {value = 1 : index} : index loc(#loc78)
    %14 = arith.select %5, %13, %2 : index loc(#loc78)
    %15 = handshake.mux %14 [%10, %falseResult_10] : index, f32 loc(#loc78)
    %dataResult_13, %addressResult_14 = handshake.store [%2] %15, %1 : index, f32 loc(#loc67)
    %dataResult_15, %addressResults_16 = handshake.load [%2] %16#0, %16#1 : index, f32 loc(#loc68)
    %16:3 = handshake.memory[ld = 1, st = 1] (%dataResult_13, %addressResult_14, %addressResults_16) {id = 0 : i32, lsq = false} : memref<1xf32>, (f32, index, index) -> (f32, none, none) loc(#loc2)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc66)
    %18:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 2 : i32} : (index) -> (f32, none) loc(#loc66)
    %19:3 = handshake.memory[ld = 1, st = 1] (%dataResult_2, %addressResult, %addressResults_12) {id = 3 : i32, lsq = false} : memref<1024xf32>, (f32, index, index) -> (f32, none, none) loc(#loc2)
    %20 = dataflow.carry %willContinue, %trueResult, %trueResult_17 : i1, none, none -> none loc(#loc79)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %18#1 : none loc(#loc79)
    %21 = handshake.mux %14 [%falseResult, %falseResult_18] : index, none loc(#loc79)
    %22 = dataflow.carry %willContinue, %trueResult, %trueResult_19 : i1, none, none -> none loc(#loc79)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %19#1 : none loc(#loc79)
    %23 = handshake.mux %14 [%falseResult, %falseResult_20] : index, none loc(#loc79)
    %trueResult_21, %falseResult_22 = handshake.cond_br %5, %23 : none loc(#loc78)
    %24 = dataflow.carry %willContinue_4, %trueResult_21, %trueResult_23 : i1, none, none -> none loc(#loc78)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_4, %19#2 : none loc(#loc78)
    %25 = handshake.constant %23 {value = 0 : index} : index loc(#loc78)
    %26 = handshake.constant %23 {value = 1 : index} : index loc(#loc78)
    %27 = arith.select %5, %26, %25 : index loc(#loc78)
    %28 = handshake.mux %27 [%falseResult_22, %falseResult_24] : index, none loc(#loc78)
    %29 = dataflow.carry %willContinue, %trueResult, %trueResult_25 : i1, none, none -> none loc(#loc79)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %17#1 : none loc(#loc79)
    %30 = handshake.mux %14 [%falseResult, %falseResult_26] : index, none loc(#loc79)
    %31 = handshake.join %21, %28, %30, %16#2 : none, none, none, none loc(#loc66)
    %32 = handshake.constant %31 {value = true} : i1 loc(#loc66)
    handshake.return %dataResult_15, %32 : f32, i1 loc(#loc66)
  } loc(#loc66)
  handshake.func @_Z11dotprod_dsaPKfS0_i(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram9>[#loc6]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram9>[#loc6]), %arg2: i32 loc(fused<#di_subprogram9>[#loc6]), %arg3: none loc(fused<#di_subprogram9>[#loc6]), ...) -> (f32, none) attributes {argNames = ["a", "b", "n", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=dotprod"], resNames = ["result", "done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc66)
    %1 = handshake.constant %0 {value = 0 : index} : index loc(#loc78)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi sgt, %arg2, %2 : i32 loc(#loc88)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc79)
    handshake.sink %falseResult : none loc(#loc79)
    %5 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc79)
    %6 = arith.index_cast %3 : i64 to index loc(#loc79)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc79)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc79)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc79)
    %dataResult, %addressResults = handshake.load [%afterValue] %17#0, %19 : index, f32 loc(#loc92)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %16#0, %28 : index, f32 loc(#loc92)
    %8 = arith.mulf %dataResult, %dataResult_0 : f32 loc(#loc92)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %8, %21 : index, f32 loc(#loc92)
    %9 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    handshake.sink %falseResult : none loc(#loc78)
    %index_3, %willContinue_4 = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc78)
    %afterValue_5, %afterCond_6 = dataflow.gate %index_3, %willContinue_4 : index, i1 -> index, i1 loc(#loc78)
    %10 = dataflow.carry %willContinue_4, %9, %11 : i1, f32, f32 -> f32 loc(#loc78)
    %afterValue_7, %afterCond_8 = dataflow.gate %10, %willContinue_4 : f32, i1 -> f32, i1 loc(#loc78)
    handshake.sink %afterCond_8 : i1 loc(#loc78)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue_4, %10 : f32 loc(#loc78)
    %dataResult_11, %addressResults_12 = handshake.load [%afterValue_5] %18#0, %23 : index, f32 loc(#loc93)
    %11 = arith.addf %dataResult_11, %afterValue_7 : f32 loc(#loc93)
    %12 = handshake.constant %0 {value = 1 : index} : index loc(#loc78)
    %13 = arith.select %4, %12, %1 : index loc(#loc78)
    %14 = handshake.mux %13 [%9, %falseResult_10] : index, f32 loc(#loc78)
    %dataResult_13, %addressResult_14 = handshake.store [%1] %14, %0 : index, f32 loc(#loc67)
    %dataResult_15, %addressResults_16 = handshake.load [%1] %15#0, %15#1 : index, f32 loc(#loc68)
    %15:3 = handshake.memory[ld = 1, st = 1] (%dataResult_13, %addressResult_14, %addressResults_16) {id = 0 : i32, lsq = false} : memref<1xf32>, (f32, index, index) -> (f32, none, none) loc(#loc2)
    %16:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc66)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 2 : i32} : (index) -> (f32, none) loc(#loc66)
    %18:3 = handshake.memory[ld = 1, st = 1] (%dataResult_2, %addressResult, %addressResults_12) {id = 3 : i32, lsq = false} : memref<1024xf32>, (f32, index, index) -> (f32, none, none) loc(#loc2)
    %19 = dataflow.carry %willContinue, %trueResult, %trueResult_17 : i1, none, none -> none loc(#loc79)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %17#1 : none loc(#loc79)
    %20 = handshake.mux %13 [%falseResult, %falseResult_18] : index, none loc(#loc79)
    %21 = dataflow.carry %willContinue, %trueResult, %trueResult_19 : i1, none, none -> none loc(#loc79)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %18#1 : none loc(#loc79)
    %22 = handshake.mux %13 [%falseResult, %falseResult_20] : index, none loc(#loc79)
    %trueResult_21, %falseResult_22 = handshake.cond_br %4, %22 : none loc(#loc78)
    %23 = dataflow.carry %willContinue_4, %trueResult_21, %trueResult_23 : i1, none, none -> none loc(#loc78)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_4, %18#2 : none loc(#loc78)
    %24 = handshake.constant %22 {value = 0 : index} : index loc(#loc78)
    %25 = handshake.constant %22 {value = 1 : index} : index loc(#loc78)
    %26 = arith.select %4, %25, %24 : index loc(#loc78)
    %27 = handshake.mux %26 [%falseResult_22, %falseResult_24] : index, none loc(#loc78)
    %28 = dataflow.carry %willContinue, %trueResult, %trueResult_25 : i1, none, none -> none loc(#loc79)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %16#1 : none loc(#loc79)
    %29 = handshake.mux %13 [%falseResult, %falseResult_26] : index, none loc(#loc79)
    %30 = handshake.join %20, %27, %29, %15#2 : none, none, none, none loc(#loc66)
    handshake.return %dataResult_15, %30 : f32, none loc(#loc68)
  } loc(#loc66)
  handshake.func @_Z13vecmul_kernelPKfS0_Pfi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram11>[#loc13]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram11>[#loc13]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram11>[#loc13]), %arg3: i32 loc(fused<#di_subprogram11>[#loc13]), %arg4: none loc(fused<#di_subprogram11>[#loc13]), ...) -> none attributes {argNames = ["a", "b", "products", "n", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=vecmul"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc69)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi sgt, %arg3, %1 : i32 loc(#loc88)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc79)
    handshake.sink %falseResult : none loc(#loc79)
    %4 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc79)
    %5 = arith.index_cast %2 : i64 to index loc(#loc79)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc79)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc79)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc79)
    %dataResult, %addressResults = handshake.load [%afterValue] %9#0, %11 : index, f32 loc(#loc92)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %8#0, %18 : index, f32 loc(#loc92)
    %7 = arith.mulf %dataResult, %dataResult_0 : f32 loc(#loc92)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %7, %16 : index, f32 loc(#loc92)
    %8:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (f32, none) loc(#loc69)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc69)
    %10 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc69)
    %11 = dataflow.carry %willContinue, %trueResult, %trueResult_3 : i1, none, none -> none loc(#loc79)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %9#1 : none loc(#loc79)
    %12 = handshake.constant %0 {value = 0 : index} : index loc(#loc79)
    %13 = handshake.constant %0 {value = 1 : index} : index loc(#loc79)
    %14 = arith.select %3, %13, %12 : index loc(#loc79)
    %15 = handshake.mux %14 [%falseResult, %falseResult_4] : index, none loc(#loc79)
    %16 = dataflow.carry %willContinue, %trueResult, %trueResult_5 : i1, none, none -> none loc(#loc79)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %10 : none loc(#loc79)
    %17 = handshake.mux %14 [%falseResult, %falseResult_6] : index, none loc(#loc79)
    %18 = dataflow.carry %willContinue, %trueResult, %trueResult_7 : i1, none, none -> none loc(#loc79)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %8#1 : none loc(#loc79)
    %19 = handshake.mux %14 [%falseResult, %falseResult_8] : index, none loc(#loc79)
    %20 = handshake.join %15, %17, %19 : none, none, none loc(#loc69)
    handshake.return %20 : none loc(#loc70)
  } loc(#loc69)
  handshake.func @_Z13vecsum_kernelPKfPfi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram10>[#loc15]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram10>[#loc15]), %arg2: i32 loc(fused<#di_subprogram10>[#loc15]), %arg3: none loc(fused<#di_subprogram10>[#loc15]), ...) -> none attributes {argNames = ["data", "result", "n", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel=vecsum"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc71)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : index} : index loc(#loc78)
    %5 = arith.cmpi sgt, %arg2, %1 : i32 loc(#loc89)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc78)
    handshake.sink %falseResult : none loc(#loc78)
    %6 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc78)
    %7 = arith.index_cast %3 : i64 to index loc(#loc78)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc78)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc78)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc78)
    %9 = dataflow.carry %willContinue, %2, %10 : i1, f32, f32 -> f32 loc(#loc78)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : f32, i1 -> f32, i1 loc(#loc78)
    handshake.sink %afterCond_1 : i1 loc(#loc78)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %9 : f32 loc(#loc78)
    %dataResult, %addressResults = handshake.load [%afterValue] %14#0, %16 : index, f32 loc(#loc93)
    %10 = arith.addf %dataResult, %afterValue_0 : f32 loc(#loc93)
    %11 = handshake.constant %0 {value = 1 : index} : index loc(#loc78)
    %12 = arith.select %5, %11, %4 : index loc(#loc78)
    %13 = handshake.mux %12 [%2, %falseResult_3] : index, f32 loc(#loc78)
    %dataResult_4, %addressResult = handshake.store [%4] %13, %0 : index, f32 loc(#loc67)
    %14:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc71)
    %15 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc71)
    %16 = dataflow.carry %willContinue, %trueResult, %trueResult_5 : i1, none, none -> none loc(#loc78)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %14#1 : none loc(#loc78)
    %17 = handshake.mux %12 [%falseResult, %falseResult_6] : index, none loc(#loc78)
    %18 = handshake.join %17, %15 : none, none loc(#loc71)
    handshake.return %18 : none loc(#loc72)
  } loc(#loc71)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc44)
    %0 = seq.const_clock  low loc(#loc44)
    %c8 = arith.constant 8 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %cst = arith.constant 1.000000e-03 : f32 loc(#loc2)
    %cst_0 = arith.constant -1.020000e+02 : f32 loc(#loc2)
    %cst_1 = arith.constant 1.020000e+02 : f64 loc(#loc2)
    %c8_i64 = arith.constant 8 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %1 = memref.get_global @__const.main.a : memref<8xf32> loc(#loc2)
    %2 = memref.get_global @__const.main.b : memref<8xf32> loc(#loc2)
    %3 = memref.get_global @str : memref<21xi8> loc(#loc2)
    %4 = memref.get_global @".str.1.11" : memref<6xi8> loc(#loc2)
    %5 = memref.get_global @".str.4.12" : memref<1xi8> loc(#loc2)
    %6 = memref.get_global @".str.3.13" : memref<3xi8> loc(#loc2)
    %7 = memref.get_global @".str.2.14" : memref<7xi8> loc(#loc2)
    %8 = memref.get_global @str.12 : memref<2xi8> loc(#loc2)
    %9 = memref.get_global @".str.6" : memref<6xi8> loc(#loc2)
    %10 = memref.get_global @".str.7" : memref<30xi8> loc(#loc2)
    %11 = memref.get_global @".str.9" : memref<48xi8> loc(#loc2)
    %12 = memref.get_global @".str.10.15" : memref<44xi8> loc(#loc2)
    %13 = memref.get_global @str.13 : memref<29xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<8xf32> loc(#loc2)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %46 = memref.load %1[%arg0] : memref<8xf32> loc(#loc45)
      memref.store %46, %alloca[%arg0] : memref<8xf32> loc(#loc45)
    } loc(#loc45)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %46 = memref.load %2[%arg0] : memref<8xf32> loc(#loc46)
      memref.store %46, %alloca_2[%arg0] : memref<8xf32> loc(#loc46)
    } loc(#loc46)
    %cast = memref.cast %alloca : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc47)
    %cast_3 = memref.cast %alloca_2 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc47)
    %14 = call @_Z11dotprod_cpuPKfS0_i(%cast, %cast_3, %c8_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> f32 loc(#loc47)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc48)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc48)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc48)
    %15:2 = handshake.esi_instance @_Z11dotprod_dsaPKfS0_i_esi "_Z11dotprod_dsaPKfS0_i_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<f32>, !esi.channel<i1>) loc(#loc48)
    %rawOutput, %valid = esi.unwrap.vr %15#0, %true : f32 loc(#loc48)
    %rawOutput_10, %valid_11 = esi.unwrap.vr %15#1, %true : i1 loc(#loc48)
    %intptr = memref.extract_aligned_pointer_as_index %3 : memref<21xi8> -> index loc(#loc49)
    %16 = arith.index_cast %intptr : index to i64 loc(#loc49)
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr loc(#loc49)
    %18 = llvm.call @puts(%17) : (!llvm.ptr) -> i32 loc(#loc49)
    %intptr_12 = memref.extract_aligned_pointer_as_index %4 : memref<6xi8> -> index loc(#loc50)
    %19 = arith.index_cast %intptr_12 : index to i64 loc(#loc50)
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr loc(#loc50)
    %21 = llvm.call @printf(%20) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc50)
    %cast_13 = memref.cast %5 : memref<1xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc80)
    %cast_14 = memref.cast %6 : memref<3xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc80)
    %intptr_15 = memref.extract_aligned_pointer_as_index %7 : memref<7xi8> -> index loc(#loc80)
    %22 = arith.index_cast %intptr_15 : index to i64 loc(#loc80)
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr loc(#loc80)
    %24 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %46 = arith.index_cast %arg0 : i64 to index loc(#loc80)
      %47 = memref.load %alloca[%46] : memref<8xf32> loc(#loc80)
      %48 = arith.extf %47 : f32 to f64 loc(#loc80)
      %49 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc80)
      %50 = arith.select %49, %cast_13, %cast_14 : memref<?xi8, strided<[1], offset: ?>> loc(#loc80)
      %intptr_19 = memref.extract_aligned_pointer_as_index %50 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc80)
      %51 = arith.index_cast %intptr_19 : index to i64 loc(#loc80)
      %52 = llvm.inttoptr %51 : i64 to !llvm.ptr loc(#loc80)
      %53 = llvm.call @printf(%23, %48, %52) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc80)
      %54 = arith.addi %arg0, %c1_i64 : i64 loc(#loc73)
      %55 = arith.cmpi ne, %54, %c8_i64 : i64 loc(#loc81)
      scf.condition(%55) %54 : i64 loc(#loc60)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc25])):
      scf.yield %arg0 : i64 loc(#loc60)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc60)
    %intptr_16 = memref.extract_aligned_pointer_as_index %8 : memref<2xi8> -> index loc(#loc51)
    %25 = arith.index_cast %intptr_16 : index to i64 loc(#loc51)
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr loc(#loc51)
    %27 = llvm.call @puts(%26) : (!llvm.ptr) -> i32 loc(#loc51)
    %intptr_17 = memref.extract_aligned_pointer_as_index %9 : memref<6xi8> -> index loc(#loc52)
    %28 = arith.index_cast %intptr_17 : index to i64 loc(#loc52)
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr loc(#loc52)
    %30 = llvm.call @printf(%29) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc52)
    %31 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %46 = arith.index_cast %arg0 : i64 to index loc(#loc82)
      %47 = memref.load %alloca_2[%46] : memref<8xf32> loc(#loc82)
      %48 = arith.extf %47 : f32 to f64 loc(#loc82)
      %49 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc82)
      %50 = arith.select %49, %cast_13, %cast_14 : memref<?xi8, strided<[1], offset: ?>> loc(#loc82)
      %intptr_19 = memref.extract_aligned_pointer_as_index %50 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc82)
      %51 = arith.index_cast %intptr_19 : index to i64 loc(#loc82)
      %52 = llvm.inttoptr %51 : i64 to !llvm.ptr loc(#loc82)
      %53 = llvm.call @printf(%23, %48, %52) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc82)
      %54 = arith.addi %arg0, %c1_i64 : i64 loc(#loc74)
      %55 = arith.cmpi ne, %54, %c8_i64 : i64 loc(#loc83)
      scf.condition(%55) %54 : i64 loc(#loc61)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block6>[#loc28])):
      scf.yield %arg0 : i64 loc(#loc61)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc61)
    %32 = llvm.call @puts(%26) : (!llvm.ptr) -> i32 loc(#loc53)
    %33 = arith.extf %rawOutput : f32 to f64 loc(#loc54)
    %intptr_18 = memref.extract_aligned_pointer_as_index %10 : memref<30xi8> -> index loc(#loc54)
    %34 = arith.index_cast %intptr_18 : index to i64 loc(#loc54)
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr loc(#loc54)
    %36 = llvm.call @printf(%35, %33, %cst_1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32 loc(#loc54)
    %37 = arith.addf %14, %cst_0 : f32 loc(#loc55)
    %38 = math.absf %37 : f32 loc(#loc62)
    %39 = arith.cmpf olt, %38, %cst : f32 loc(#loc55)
    %40 = arith.addf %rawOutput, %cst_0 : f32 loc(#loc56)
    %41 = math.absf %40 : f32 loc(#loc63)
    %42 = arith.cmpf olt, %41, %cst : f32 loc(#loc56)
    %43 = arith.andi %39, %42 : i1 loc(#loc64)
    %44 = arith.xori %43, %true : i1 loc(#loc64)
    %45 = arith.extui %44 : i1 to i32 loc(#loc64)
    scf.if %43 {
      %intptr_19 = memref.extract_aligned_pointer_as_index %13 : memref<29xi8> -> index loc(#loc75)
      %46 = arith.index_cast %intptr_19 : index to i64 loc(#loc75)
      %47 = llvm.inttoptr %46 : i64 to !llvm.ptr loc(#loc75)
      %48 = llvm.call @puts(%47) : (!llvm.ptr) -> i32 loc(#loc75)
    } else {
      scf.if %39 {
      } else {
        %46 = arith.extf %14 : f32 to f64 loc(#loc90)
        %intptr_19 = memref.extract_aligned_pointer_as_index %11 : memref<48xi8> -> index loc(#loc90)
        %47 = arith.index_cast %intptr_19 : index to i64 loc(#loc90)
        %48 = llvm.inttoptr %47 : i64 to !llvm.ptr loc(#loc90)
        %49 = llvm.call @printf(%48, %46, %cst_1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32 loc(#loc90)
      } loc(#loc84)
      scf.if %42 {
      } else {
        %intptr_19 = memref.extract_aligned_pointer_as_index %12 : memref<44xi8> -> index loc(#loc91)
        %46 = arith.index_cast %intptr_19 : index to i64 loc(#loc91)
        %47 = llvm.inttoptr %46 : i64 to !llvm.ptr loc(#loc91)
        %48 = llvm.call @printf(%47, %33, %cst_1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32 loc(#loc91)
      } loc(#loc85)
    } loc(#loc64)
    return %45 : i32 loc(#loc57)
  } loc(#loc44)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc76)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
} loc(#loc)
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file3 = #llvm.di_file<"/usr/lib/gcc/x86_64-redhat-linux/14/../../../../include/c++/14/cmath" in "">
#di_file4 = #llvm.di_file<"/usr/include/stdio.h" in "">
#di_namespace = #llvm.di_namespace<name = "std", exportSymbols = false>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[13]<>, isRecSelf = true>
#loc = loc("tests/app/dotprod/dotprod.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/dotprod/dotprod.cpp":23:0)
#loc5 = loc("tests/app/dotprod/dotprod.cpp":25:0)
#loc7 = loc("tests/app/dotprod/kernels.cpp":39:0)
#loc8 = loc("tests/app/dotprod/kernels.cpp":22:0)
#loc9 = loc("tests/app/dotprod/kernels.cpp":23:0)
#loc10 = loc("tests/app/dotprod/kernels.cpp":40:0)
#loc11 = loc("tests/app/dotprod/kernels.cpp":42:0)
#loc12 = loc("tests/app/dotprod/dotprod.cpp":55:0)
#loc14 = loc("tests/app/dotprod/kernels.cpp":25:0)
#loc16 = loc("tests/app/dotprod/kernels.cpp":43:0)
#loc17 = loc("tests/app/dotprod/main.cpp":13:0)
#loc18 = loc("tests/app/dotprod/main.cpp":14:0)
#loc19 = loc("tests/app/dotprod/main.cpp":15:0)
#loc20 = loc("tests/app/dotprod/main.cpp":22:0)
#loc21 = loc("tests/app/dotprod/main.cpp":25:0)
#loc22 = loc("tests/app/dotprod/main.cpp":27:0)
#loc23 = loc("tests/app/dotprod/main.cpp":28:0)
#loc24 = loc("tests/app/dotprod/main.cpp":30:0)
#loc26 = loc("tests/app/dotprod/main.cpp":32:0)
#loc27 = loc("tests/app/dotprod/main.cpp":34:0)
#loc29 = loc("tests/app/dotprod/main.cpp":36:0)
#loc30 = loc("tests/app/dotprod/main.cpp":38:0)
#loc31 = loc("tests/app/dotprod/main.cpp":40:0)
#loc32 = loc("tests/app/dotprod/main.cpp":44:0)
#loc33 = loc("/usr/lib/gcc/x86_64-redhat-linux/14/../../../../include/c++/14/cmath":239:0)
#loc34 = loc("tests/app/dotprod/main.cpp":45:0)
#loc35 = loc("tests/app/dotprod/main.cpp":47:0)
#loc36 = loc("tests/app/dotprod/main.cpp":48:0)
#loc37 = loc("tests/app/dotprod/main.cpp":51:0)
#loc38 = loc("tests/app/dotprod/main.cpp":52:0)
#loc39 = loc("tests/app/dotprod/main.cpp":55:0)
#loc40 = loc("tests/app/dotprod/main.cpp":56:0)
#loc41 = loc("tests/app/dotprod/main.cpp":61:0)
#loc42 = loc("/usr/include/stdio.h":363:0)
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type3>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram5, name = "__x", file = #di_file3, line = 238, arg = 1, type = #di_basic_type>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type, #di_basic_type>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[13]<>, id = distinct[14]<>, compileUnit = #di_compile_unit2, scope = #di_namespace, name = "fabs", linkageName = "_ZSt4fabsf", file = #di_file3, line = 238, scopeLine = 239, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable12>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#loc43 = loc(fused<#di_subprogram6>[#loc33])
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram7, file = #di_file2, line = 47>
#di_subroutine_type6 = #llvm.di_subroutine_type<types = #di_basic_type1, #di_derived_type7, #di_null_type>
#loc44 = loc(fused<#di_subprogram7>[#loc17])
#loc45 = loc(fused<#di_subprogram7>[#loc18])
#loc46 = loc(fused<#di_subprogram7>[#loc19])
#loc47 = loc(fused<#di_subprogram7>[#loc20])
#loc48 = loc(fused<#di_subprogram7>[#loc21])
#loc49 = loc(fused<#di_subprogram7>[#loc22])
#loc50 = loc(fused<#di_subprogram7>[#loc23])
#loc51 = loc(fused<#di_subprogram7>[#loc26])
#loc52 = loc(fused<#di_subprogram7>[#loc27])
#loc53 = loc(fused<#di_subprogram7>[#loc30])
#loc54 = loc(fused<#di_subprogram7>[#loc31])
#loc55 = loc(fused<#di_subprogram7>[#loc32])
#loc56 = loc(fused<#di_subprogram7>[#loc34])
#loc57 = loc(fused<#di_subprogram7>[#loc41])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file2, line = 29>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file2, line = 35>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file2, line = 47>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file2, line = 50>
#di_subprogram12 = #llvm.di_subprogram<scope = #di_file4, name = "printf", file = #di_file4, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type6>
#loc59 = loc(fused<#di_subprogram8>[#loc5])
#loc62 = loc(callsite(#loc43 at #loc55))
#loc63 = loc(callsite(#loc43 at #loc56))
#loc64 = loc(fused<#di_lexical_block7>[#loc35])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 22>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram10, file = #di_file1, line = 39>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram11, file = #di_file1, line = 22>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file2, line = 29>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file2, line = 35>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file2, line = 51>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file2, line = 55>
#loc67 = loc(fused<#di_subprogram10>[#loc11])
#loc68 = loc(fused<#di_subprogram9>[#loc12])
#loc70 = loc(fused<#di_subprogram11>[#loc14])
#loc72 = loc(fused<#di_subprogram10>[#loc16])
#loc73 = loc(fused<#di_lexical_block9>[#loc25])
#loc74 = loc(fused<#di_lexical_block10>[#loc28])
#loc75 = loc(fused<#di_lexical_block11>[#loc36])
#loc76 = loc(fused<#di_subprogram12>[#loc42])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 22>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 22>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 39>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file2, line = 51>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file2, line = 55>
#loc77 = loc(fused<#di_lexical_block13>[#loc3])
#loc78 = loc(fused<#di_lexical_block14>[#loc7])
#loc79 = loc(fused<#di_lexical_block15>[#loc8])
#loc80 = loc(fused<#di_lexical_block16>[#loc24])
#loc81 = loc(fused[#loc60, #loc73])
#loc82 = loc(fused<#di_lexical_block17>[#loc29])
#loc83 = loc(fused[#loc61, #loc74])
#loc84 = loc(fused<#di_lexical_block18>[#loc37])
#loc85 = loc(fused<#di_lexical_block19>[#loc39])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 22>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 39>
#loc86 = loc(fused<#di_lexical_block20>[#loc4])
#loc87 = loc(fused[#loc65, #loc77])
#loc88 = loc(fused<#di_lexical_block21>[#loc8])
#loc89 = loc(fused<#di_lexical_block22>[#loc7])
#loc90 = loc(fused<#di_lexical_block23>[#loc38])
#loc91 = loc(fused<#di_lexical_block24>[#loc40])
#loc92 = loc(fused<#di_lexical_block25>[#loc9])
#loc93 = loc(fused<#di_lexical_block26>[#loc10])
