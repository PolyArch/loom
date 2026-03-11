#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/compare_swap/compare_swap.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/compare_swap/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/compare_swap/compare_swap.cpp":12:0)
#loc3 = loc("tests/app/compare_swap/compare_swap.cpp":17:0)
#loc6 = loc("tests/app/compare_swap/compare_swap.cpp":31:0)
#loc16 = loc("tests/app/compare_swap/main.cpp":30:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 512, elements = #llvm.di_subrange<count = 16 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 38>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 30>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_a", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_b", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_min", file = #di_file1, line = 16, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_max", file = #di_file1, line = 17, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_min", file = #di_file1, line = 20, type = #di_composite_type>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_max", file = #di_file1, line = 21, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_min", file = #di_file, line = 14, arg = 3, type = #di_derived_type4>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_max", file = #di_file, line = 15, arg = 4, type = #di_derived_type4>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_min", file = #di_file, line = 33, arg = 3, type = #di_derived_type4>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_max", file = #di_file, line = 34, arg = 4, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 38, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 30, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_a", file = #di_file, line = 12, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_b", file = #di_file, line = 13, arg = 2, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 5, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_a", file = #di_file, line = 31, arg = 1, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_b", file = #di_file, line = 32, arg = 2, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 35, arg = 5, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 7, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_derived_type4, #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compare_swap_cpu", linkageName = "_Z16compare_swap_cpuPKfS0_PfS1_j", file = #di_file, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable6, #di_local_variable7, #di_local_variable15, #di_local_variable8>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compare_swap_dsa", linkageName = "_Z16compare_swap_dsaPKfS0_PfS1_j", file = #di_file, line = 31, scopeLine = 35, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable17, #di_local_variable9, #di_local_variable10, #di_local_variable18, #di_local_variable11>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable19, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5, #di_local_variable12>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 30>
#loc21 = loc(fused<#di_subprogram3>[#loc1])
#loc23 = loc(fused<#di_subprogram4>[#loc6])
#loc32 = loc(fused<#di_lexical_block3>[#loc3])
#loc34 = loc(fused<#di_lexical_block5>[#loc16])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<40xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 111, 109, 112, 97, 114, 101, 95, 115, 119, 97, 112, 47, 99, 111, 109, 112, 97, 114, 101, 95, 115, 119, 97, 112, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @__const.main.input_a : memref<16xf32> = dense<[5.000000e+00, 2.000000e+00, 8.000000e+00, 1.000000e+00, 9.000000e+00, 3.000000e+00, 7.000000e+00, 4.000000e+00, 6.000000e+00, 1.000000e+01, 1.500000e+01, 1.200000e+01, 1.100000e+01, 1.400000e+01, 1.300000e+01, 1.600000e+01]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.input_b : memref<16xf32> = dense<[3.000000e+00, 7.000000e+00, 1.000000e+00, 9.000000e+00, 2.000000e+00, 8.000000e+00, 4.000000e+00, 6.000000e+00, 1.000000e+01, 5.000000e+00, 1.200000e+01, 1.500000e+01, 1.400000e+01, 1.100000e+01, 1.600000e+01, 1.300000e+01]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @str.2 : memref<21xi8> = dense<[99, 111, 109, 112, 97, 114, 101, 95, 115, 119, 97, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.3 : memref<21xi8> = dense<[99, 111, 109, 112, 97, 114, 101, 95, 115, 119, 97, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z16compare_swap_cpuPKfS0_PfS1_j(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg3: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg4: i32 loc(fused<#di_subprogram3>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc35)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg4 : i32 to i64 loc(#loc35)
      %2 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg5 : i64 to index loc(#loc41)
        %4 = memref.load %arg0[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc41)
        %5 = memref.load %arg1[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc41)
        %6 = arith.cmpf ugt, %4, %5 : f32 loc(#loc41)
        %7 = arith.select %6, %5, %4 : f32 loc(#loc2)
        %8 = arith.select %6, %4, %5 : f32 loc(#loc2)
        memref.store %7, %arg2[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc42)
        memref.store %8, %arg3[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc42)
        %9 = arith.addi %arg5, %c1_i64 : i64 loc(#loc35)
        %10 = arith.cmpi ne, %9, %1 : i64 loc(#loc38)
        scf.condition(%10) %9 : i64 loc(#loc32)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block3>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc32)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc32)
    } loc(#loc32)
    return loc(#loc22)
  } loc(#loc21)
  handshake.func @_Z16compare_swap_dsaPKfS0_PfS1_j_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg4: i32 loc(fused<#di_subprogram4>[#loc6]), %arg5: i1 loc(fused<#di_subprogram4>[#loc6]), ...) -> i1 attributes {argNames = ["input_a", "input_b", "output_min", "output_max", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc23)
    %1 = handshake.join %0 : none loc(#loc23)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc36)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc33)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc33)
    %6 = arith.index_cast %3 : i64 to index loc(#loc33)
    %7 = arith.index_cast %arg4 : i32 to index loc(#loc33)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=2", "loom.loop.tripcount typical=0 avg=0 min=10 max=1000"], step_op = "+=", stop_cond = "!="} loc(#loc33)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc33)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %15 : index, f32 loc(#loc43)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %12#0, %22 : index, f32 loc(#loc43)
    %8 = arith.cmpf ugt, %dataResult, %dataResult_0 : f32 loc(#loc43)
    %9 = arith.select %8, %dataResult_0, %dataResult : f32 loc(#loc2)
    %10 = arith.select %8, %dataResult, %dataResult_0 : f32 loc(#loc2)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %9, %20 : index, f32 loc(#loc44)
    %dataResult_3, %addressResult_4 = handshake.store [%afterValue] %10, %24 : index, f32 loc(#loc44)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc23)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc23)
    %13 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc23)
    %14 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_3, %addressResult_4) {id = 3 : i32} : (f32, index) -> none loc(#loc23)
    %15 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc33)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11#1 : none loc(#loc33)
    %16 = handshake.constant %1 {value = 0 : index} : index loc(#loc33)
    %17 = handshake.constant %1 {value = 1 : index} : index loc(#loc33)
    %18 = arith.select %4, %17, %16 : index loc(#loc33)
    %19 = handshake.mux %18 [%falseResult_6, %trueResult] : index, none loc(#loc33)
    %20 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc33)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %13 : none loc(#loc33)
    %21 = handshake.mux %18 [%falseResult_8, %trueResult] : index, none loc(#loc33)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc33)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %12#1 : none loc(#loc33)
    %23 = handshake.mux %18 [%falseResult_10, %trueResult] : index, none loc(#loc33)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc33)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %14 : none loc(#loc33)
    %25 = handshake.mux %18 [%falseResult_12, %trueResult] : index, none loc(#loc33)
    %26 = handshake.join %19, %21, %23, %25 : none, none, none, none loc(#loc23)
    %27 = handshake.constant %26 {value = true} : i1 loc(#loc23)
    handshake.return %27 : i1 loc(#loc23)
  } loc(#loc23)
  handshake.func @_Z16compare_swap_dsaPKfS0_PfS1_j(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc6]), %arg4: i32 loc(fused<#di_subprogram4>[#loc6]), %arg5: none loc(fused<#di_subprogram4>[#loc6]), ...) -> none attributes {argNames = ["input_a", "input_b", "output_min", "output_max", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc23)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc36)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc33)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc33)
    %5 = arith.index_cast %2 : i64 to index loc(#loc33)
    %6 = arith.index_cast %arg4 : i32 to index loc(#loc33)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=2", "loom.loop.tripcount typical=0 avg=0 min=10 max=1000"], step_op = "+=", stop_cond = "!="} loc(#loc33)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc33)
    %dataResult, %addressResults = handshake.load [%afterValue] %10#0, %14 : index, f32 loc(#loc43)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %11#0, %21 : index, f32 loc(#loc43)
    %7 = arith.cmpf ugt, %dataResult, %dataResult_0 : f32 loc(#loc43)
    %8 = arith.select %7, %dataResult_0, %dataResult : f32 loc(#loc2)
    %9 = arith.select %7, %dataResult, %dataResult_0 : f32 loc(#loc2)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %8, %19 : index, f32 loc(#loc44)
    %dataResult_3, %addressResult_4 = handshake.store [%afterValue] %9, %23 : index, f32 loc(#loc44)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc23)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc23)
    %12 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc23)
    %13 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_3, %addressResult_4) {id = 3 : i32} : (f32, index) -> none loc(#loc23)
    %14 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc33)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %10#1 : none loc(#loc33)
    %15 = handshake.constant %0 {value = 0 : index} : index loc(#loc33)
    %16 = handshake.constant %0 {value = 1 : index} : index loc(#loc33)
    %17 = arith.select %3, %16, %15 : index loc(#loc33)
    %18 = handshake.mux %17 [%falseResult_6, %trueResult] : index, none loc(#loc33)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc33)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %12 : none loc(#loc33)
    %20 = handshake.mux %17 [%falseResult_8, %trueResult] : index, none loc(#loc33)
    %21 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc33)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %11#1 : none loc(#loc33)
    %22 = handshake.mux %17 [%falseResult_10, %trueResult] : index, none loc(#loc33)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc33)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %13 : none loc(#loc33)
    %24 = handshake.mux %17 [%falseResult_12, %trueResult] : index, none loc(#loc33)
    %25 = handshake.join %18, %20, %22, %24 : none, none, none, none loc(#loc23)
    handshake.return %25 : none loc(#loc24)
  } loc(#loc23)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc25)
    %c2_i32 = arith.constant 2 : i32 loc(#loc25)
    %1 = ub.poison : i64 loc(#loc25)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c16 = arith.constant 16 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c16_i64 = arith.constant 16 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %cst = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.input_a : memref<16xf32> loc(#loc2)
    %3 = memref.get_global @__const.main.input_b : memref<16xf32> loc(#loc2)
    %4 = memref.get_global @str.2 : memref<21xi8> loc(#loc2)
    %5 = memref.get_global @str.3 : memref<21xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<16xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<16xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<16xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<16xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<16xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<16xf32> loc(#loc2)
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %11 = memref.load %2[%arg0] : memref<16xf32> loc(#loc26)
      memref.store %11, %alloca[%arg0] : memref<16xf32> loc(#loc26)
    } loc(#loc26)
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %11 = memref.load %3[%arg0] : memref<16xf32> loc(#loc27)
      memref.store %11, %alloca_0[%arg0] : memref<16xf32> loc(#loc27)
    } loc(#loc27)
    %cast = memref.cast %alloca : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    %cast_5 = memref.cast %alloca_0 : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    %cast_6 = memref.cast %alloca_1 : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    %cast_7 = memref.cast %alloca_2 : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    call @_Z16compare_swap_cpuPKfS0_PfS1_j(%cast, %cast_5, %cast_6, %cast_7, %c16_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc28)
    %cast_8 = memref.cast %alloca_3 : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %cast_9 = memref.cast %alloca_4 : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_10, %ready_11 = esi.wrap.vr %cast_5, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_12, %ready_13 = esi.wrap.vr %cast_8, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_14, %ready_15 = esi.wrap.vr %cast_9, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_16, %ready_17 = esi.wrap.vr %c16_i32, %true : i32 loc(#loc29)
    %chanOutput_18, %ready_19 = esi.wrap.vr %true, %true : i1 loc(#loc29)
    %6 = handshake.esi_instance @_Z16compare_swap_dsaPKfS0_PfS1_j_esi "_Z16compare_swap_dsaPKfS0_PfS1_j_inst0" clk %0 rst %false(%chanOutput, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc29)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc29)
    %cast_20 = memref.cast %4 : memref<21xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc45)
    %7:2 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      %12 = memref.load %alloca_1[%11] : memref<16xf32> loc(#loc45)
      %13 = memref.load %alloca_3[%11] : memref<16xf32> loc(#loc45)
      %14 = arith.subf %12, %13 : f32 loc(#loc45)
      %15 = math.absf %14 : f32 loc(#loc45)
      %16 = arith.cmpf ogt, %15, %cst : f32 loc(#loc45)
      %17:3 = scf.if %16 -> (i64, i32, i32) {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc45)
      } else {
        %19 = memref.load %alloca_2[%11] : memref<16xf32> loc(#loc46)
        %20 = memref.load %alloca_4[%11] : memref<16xf32> loc(#loc46)
        %21 = arith.subf %19, %20 : f32 loc(#loc46)
        %22 = math.absf %21 : f32 loc(#loc46)
        %23 = arith.cmpf ogt, %22, %cst : f32 loc(#loc46)
        %24:3 = scf.if %23 -> (i64, i32, i32) {
          scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc46)
        } else {
          %25 = arith.addi %arg0, %c1_i64 : i64 loc(#loc37)
          %26 = arith.cmpi eq, %25, %c16_i64 : i64 loc(#loc37)
          %27 = arith.extui %26 : i1 to i32 loc(#loc34)
          %28 = arith.cmpi ne, %25, %c16_i64 : i64 loc(#loc39)
          %29 = arith.extui %28 : i1 to i32 loc(#loc34)
          scf.yield %25, %27, %29 : i64, i32, i32 loc(#loc46)
        } loc(#loc46)
        scf.yield %24#0, %24#1, %24#2 : i64, i32, i32 loc(#loc45)
      } loc(#loc45)
      %18 = arith.trunci %17#2 : i32 to i1 loc(#loc34)
      scf.condition(%18) %17#0, %17#1 : i64, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc16]), %arg1: i32 loc(fused<#di_lexical_block5>[#loc16])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %8 = arith.index_castui %7#1 : i32 to index loc(#loc34)
    %9:2 = scf.index_switch %8 -> i1, i32 
    case 1 {
      scf.yield %true, %c0_i32 : i1, i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %cast_20 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc40)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc40)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc40)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc40)
      scf.yield %false, %c1_i32 : i1, i32 loc(#loc2)
    } loc(#loc34)
    %10 = arith.select %9#0, %c0_i32, %9#1 : i32 loc(#loc2)
    scf.if %9#0 {
      %intptr = memref.extract_aligned_pointer_as_index %5 : memref<21xi8> -> index loc(#loc30)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc30)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc30)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc30)
    } loc(#loc2)
    return %10 : i32 loc(#loc31)
  } loc(#loc25)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/compare_swap/compare_swap.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/compare_swap/compare_swap.cpp":18:0)
#loc5 = loc("tests/app/compare_swap/compare_swap.cpp":26:0)
#loc7 = loc("tests/app/compare_swap/compare_swap.cpp":38:0)
#loc8 = loc("tests/app/compare_swap/compare_swap.cpp":39:0)
#loc9 = loc("tests/app/compare_swap/compare_swap.cpp":47:0)
#loc10 = loc("tests/app/compare_swap/main.cpp":6:0)
#loc11 = loc("tests/app/compare_swap/main.cpp":10:0)
#loc12 = loc("tests/app/compare_swap/main.cpp":12:0)
#loc13 = loc("tests/app/compare_swap/main.cpp":24:0)
#loc14 = loc("tests/app/compare_swap/main.cpp":27:0)
#loc15 = loc("tests/app/compare_swap/main.cpp":31:0)
#loc17 = loc("tests/app/compare_swap/main.cpp":35:0)
#loc18 = loc("tests/app/compare_swap/main.cpp":0:0)
#loc19 = loc("tests/app/compare_swap/main.cpp":41:0)
#loc20 = loc("tests/app/compare_swap/main.cpp":43:0)
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 38>
#loc22 = loc(fused<#di_subprogram3>[#loc5])
#loc24 = loc(fused<#di_subprogram4>[#loc9])
#loc25 = loc(fused<#di_subprogram5>[#loc10])
#loc26 = loc(fused<#di_subprogram5>[#loc11])
#loc27 = loc(fused<#di_subprogram5>[#loc12])
#loc28 = loc(fused<#di_subprogram5>[#loc13])
#loc29 = loc(fused<#di_subprogram5>[#loc14])
#loc30 = loc(fused<#di_subprogram5>[#loc19])
#loc31 = loc(fused<#di_subprogram5>[#loc20])
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 38>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 30>
#loc33 = loc(fused<#di_lexical_block4>[#loc7])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 17>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 38>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 30>
#loc35 = loc(fused<#di_lexical_block6>[#loc3])
#loc36 = loc(fused<#di_lexical_block7>[#loc7])
#loc37 = loc(fused<#di_lexical_block8>[#loc16])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 18>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 39>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 31>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 35>
#loc38 = loc(fused[#loc32, #loc35])
#loc39 = loc(fused[#loc34, #loc37])
#loc40 = loc(fused<#di_lexical_block11>[#loc18])
#loc41 = loc(fused<#di_lexical_block12>[#loc4])
#loc42 = loc(fused<#di_lexical_block12>[#loc])
#loc43 = loc(fused<#di_lexical_block13>[#loc8])
#loc44 = loc(fused<#di_lexical_block13>[#loc])
#loc45 = loc(fused<#di_lexical_block14>[#loc15])
#loc46 = loc(fused<#di_lexical_block15>[#loc17])
