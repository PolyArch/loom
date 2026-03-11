#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/cumsum/cumsum.cpp" in "/home/sihao/github.com/PolyArch/gem5-loom/ext/loom">
#di_file1 = #llvm.di_file<"tests/app/cumsum/main.cpp" in "/home/sihao/github.com/PolyArch/gem5-loom/ext/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/cumsum/cumsum.cpp":13:0)
#loc3 = loc("tests/app/cumsum/cumsum.cpp":17:0)
#loc7 = loc("tests/app/cumsum/cumsum.cpp":27:0)
#loc13 = loc("tests/app/cumsum/main.cpp":11:0)
#loc17 = loc("tests/app/cumsum/main.cpp":26:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 33>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 11>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 26>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "sum", file = #di_file, line = 16, type = #di_basic_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram1, name = "sum", file = #di_file, line = 31, type = #di_basic_type>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 16, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 17, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 14, arg = 2, type = #di_derived_type4>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 28, arg = 2, type = #di_derived_type4>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 33, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 11, type = #di_derived_type5>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 26, type = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 13, arg = 1, type = #di_derived_type6>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 15, arg = 3, type = #di_derived_type7>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 27, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 29, arg = 3, type = #di_derived_type7>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 7, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type4, #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "cumsum_cpu", linkageName = "_Z10cumsum_cpuPKfPfj", file = #di_file, line = 13, scopeLine = 15, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable11, #di_local_variable5, #di_local_variable12, #di_local_variable, #di_local_variable6>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "cumsum_dsa", linkageName = "_Z10cumsum_dsaPKfPfj", file = #di_file, line = 27, scopeLine = 29, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable7, #di_local_variable14, #di_local_variable1, #di_local_variable8>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable15, #di_local_variable2, #di_local_variable9, #di_local_variable3, #di_local_variable4, #di_local_variable10>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 11>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 26>
#loc23 = loc(fused<#di_subprogram3>[#loc1])
#loc25 = loc(fused<#di_subprogram4>[#loc7])
#loc32 = loc(fused<#di_lexical_block4>[#loc3])
#loc34 = loc(fused<#di_lexical_block6>[#loc13])
#loc35 = loc(fused<#di_lexical_block7>[#loc17])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<14xi8> = dense<[108, 111, 111, 109, 46, 114, 101, 100, 117, 99, 101, 61, 43, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 117, 109, 115, 117, 109, 47, 99, 117, 109, 115, 117, 109, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<15xi8> = dense<[99, 117, 109, 115, 117, 109, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[99, 117, 109, 115, 117, 109, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z10cumsum_cpuPKfPfj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg2: i32 loc(fused<#di_subprogram3>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc36)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc36)
      %2:2 = scf.while (%arg3 = %c0_i64, %arg4 = %cst) : (i64, f32) -> (i64, f32) {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc40)
        %4 = memref.load %arg0[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
        %5 = arith.addf %arg4, %4 : f32 loc(#loc40)
        memref.store %5, %arg1[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc41)
        %6 = arith.addi %arg3, %c1_i64 : i64 loc(#loc36)
        %7 = arith.cmpi ne, %6, %1 : i64 loc(#loc42)
        scf.condition(%7) %6, %5 : i64, f32 loc(#loc32)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block4>[#loc3]), %arg4: f32 loc(fused<#di_lexical_block4>[#loc3])):
        scf.yield %arg3, %arg4 : i64, f32 loc(#loc32)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc32)
    } loc(#loc32)
    return loc(#loc24)
  } loc(#loc23)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  handshake.func @_Z10cumsum_dsaPKfPfj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc7]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc7]), %arg2: i32 loc(fused<#di_subprogram4>[#loc7]), %arg3: i1 loc(fused<#di_subprogram4>[#loc7]), ...) -> i1 attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc25)
    %1 = handshake.join %0 : none loc(#loc25)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc37)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc33)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc33)
    %7 = arith.index_cast %4 : i64 to index loc(#loc33)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc33)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc33)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc33)
    %9 = dataflow.carry %willContinue, %3, %10 : i1, f32, f32 -> f32 loc(#loc33)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : f32, i1 -> f32, i1 loc(#loc33)
    handshake.sink %afterCond_1 : i1 loc(#loc33)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %13 : index, f32 loc(#loc43)
    %10 = arith.addf %dataResult, %afterValue_0 : f32 loc(#loc43)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %10, %18 : index, f32 loc(#loc44)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc25)
    %12 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc25)
    %13 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc33)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %11#1 : none loc(#loc33)
    %14 = handshake.constant %1 {value = 0 : index} : index loc(#loc33)
    %15 = handshake.constant %1 {value = 1 : index} : index loc(#loc33)
    %16 = arith.select %5, %15, %14 : index loc(#loc33)
    %17 = handshake.mux %16 [%falseResult_4, %trueResult] : index, none loc(#loc33)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc33)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %12 : none loc(#loc33)
    %19 = handshake.mux %16 [%falseResult_6, %trueResult] : index, none loc(#loc33)
    %20 = handshake.join %17, %19 : none, none loc(#loc25)
    %21 = handshake.constant %20 {value = true} : i1 loc(#loc25)
    handshake.return %21 : i1 loc(#loc25)
  } loc(#loc25)
  handshake.func @_Z10cumsum_dsaPKfPfj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc7]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc7]), %arg2: i32 loc(fused<#di_subprogram4>[#loc7]), %arg3: none loc(fused<#di_subprogram4>[#loc7]), ...) -> none attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc25)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %1 : i32 loc(#loc37)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc33)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc33)
    %6 = arith.index_cast %3 : i64 to index loc(#loc33)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc33)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc33)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc33)
    %8 = dataflow.carry %willContinue, %2, %9 : i1, f32, f32 -> f32 loc(#loc33)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : f32, i1 -> f32, i1 loc(#loc33)
    handshake.sink %afterCond_1 : i1 loc(#loc33)
    %dataResult, %addressResults = handshake.load [%afterValue] %10#0, %12 : index, f32 loc(#loc43)
    %9 = arith.addf %dataResult, %afterValue_0 : f32 loc(#loc43)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %9, %17 : index, f32 loc(#loc44)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc25)
    %11 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc25)
    %12 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc33)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %10#1 : none loc(#loc33)
    %13 = handshake.constant %0 {value = 0 : index} : index loc(#loc33)
    %14 = handshake.constant %0 {value = 1 : index} : index loc(#loc33)
    %15 = arith.select %4, %14, %13 : index loc(#loc33)
    %16 = handshake.mux %15 [%falseResult_4, %trueResult] : index, none loc(#loc33)
    %17 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc33)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11 : none loc(#loc33)
    %18 = handshake.mux %15 [%falseResult_6, %trueResult] : index, none loc(#loc33)
    %19 = handshake.join %16, %18 : none, none loc(#loc25)
    handshake.return %19 : none loc(#loc26)
  } loc(#loc25)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc27)
    %false = arith.constant false loc(#loc27)
    %0 = seq.const_clock  low loc(#loc27)
    %c2_i32 = arith.constant 2 : i32 loc(#loc27)
    %1 = ub.poison : i64 loc(#loc27)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %cst = arith.constant 1.000000e-03 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc45)
      %11 = arith.remui %10, %c10_i32 : i32 loc(#loc45)
      %12 = arith.addi %11, %c1_i32 : i32 loc(#loc45)
      %13 = arith.uitofp %12 : i32 to f32 loc(#loc45)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      memref.store %13, %alloca[%14] : memref<1024xf32> loc(#loc45)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc38)
      %16 = arith.cmpi ne, %15, %c1024_i64 : i64 loc(#loc46)
      scf.condition(%16) %15 : i64 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block6>[#loc13])):
      scf.yield %arg0 : i64 loc(#loc34)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc34)
    %cast = memref.cast %alloca : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    %cast_2 = memref.cast %alloca_0 : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc28)
    call @_Z10cumsum_cpuPKfPfj(%cast, %cast_2, %c1024_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc28)
    %cast_3 = memref.cast %alloca_1 : memref<1024xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc29)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc29)
    %5 = handshake.esi_instance @_Z10cumsum_dsaPKfPfj_esi "_Z10cumsum_dsaPKfPfj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc29)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc29)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc48)
      %11 = memref.load %alloca_0[%10] : memref<1024xf32> loc(#loc48)
      %12 = memref.load %alloca_1[%10] : memref<1024xf32> loc(#loc48)
      %13 = arith.subf %11, %12 : f32 loc(#loc48)
      %14 = math.absf %13 : f32 loc(#loc48)
      %15 = arith.cmpf ule, %14, %cst : f32 loc(#loc48)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
        %19 = arith.cmpi eq, %18, %c1024_i64 : i64 loc(#loc39)
        %20 = arith.extui %19 : i1 to i32 loc(#loc35)
        %21 = arith.cmpi ne, %18, %c1024_i64 : i64 loc(#loc47)
        %22 = arith.extui %21 : i1 to i32 loc(#loc35)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc48)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc48)
      } loc(#loc48)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc35)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc35)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block7>[#loc17]), %arg1: i1 loc(fused<#di_lexical_block7>[#loc17]), %arg2: i32 loc(fused<#di_lexical_block7>[#loc17])):
      scf.yield %arg0 : i64 loc(#loc35)
    } loc(#loc35)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc35)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc35)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc49)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc49)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc49)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc49)
      scf.yield %c1_i32 : i32 loc(#loc50)
    } loc(#loc35)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc30)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc30)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc30)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc30)
    } loc(#loc2)
    return %9 : i32 loc(#loc31)
  } loc(#loc27)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/cumsum/cumsum.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/cumsum/cumsum.cpp":18:0)
#loc5 = loc("tests/app/cumsum/cumsum.cpp":19:0)
#loc6 = loc("tests/app/cumsum/cumsum.cpp":21:0)
#loc8 = loc("tests/app/cumsum/cumsum.cpp":33:0)
#loc9 = loc("tests/app/cumsum/cumsum.cpp":34:0)
#loc10 = loc("tests/app/cumsum/cumsum.cpp":35:0)
#loc11 = loc("tests/app/cumsum/cumsum.cpp":37:0)
#loc12 = loc("tests/app/cumsum/main.cpp":6:0)
#loc14 = loc("tests/app/cumsum/main.cpp":12:0)
#loc15 = loc("tests/app/cumsum/main.cpp":20:0)
#loc16 = loc("tests/app/cumsum/main.cpp":23:0)
#loc18 = loc("tests/app/cumsum/main.cpp":27:0)
#loc19 = loc("tests/app/cumsum/main.cpp":28:0)
#loc20 = loc("tests/app/cumsum/main.cpp":29:0)
#loc21 = loc("tests/app/cumsum/main.cpp":33:0)
#loc22 = loc("tests/app/cumsum/main.cpp":35:0)
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 33>
#loc24 = loc(fused<#di_subprogram3>[#loc6])
#loc26 = loc(fused<#di_subprogram4>[#loc11])
#loc27 = loc(fused<#di_subprogram5>[#loc12])
#loc28 = loc(fused<#di_subprogram5>[#loc15])
#loc29 = loc(fused<#di_subprogram5>[#loc16])
#loc30 = loc(fused<#di_subprogram5>[#loc21])
#loc31 = loc(fused<#di_subprogram5>[#loc22])
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 33>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 11>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 26>
#loc33 = loc(fused<#di_lexical_block5>[#loc8])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 17>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 33>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 11>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 26>
#loc36 = loc(fused<#di_lexical_block8>[#loc3])
#loc37 = loc(fused<#di_lexical_block9>[#loc8])
#loc38 = loc(fused<#di_lexical_block10>[#loc13])
#loc39 = loc(fused<#di_lexical_block11>[#loc17])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 27>
#loc40 = loc(fused<#di_lexical_block12>[#loc4])
#loc41 = loc(fused<#di_lexical_block12>[#loc5])
#loc42 = loc(fused[#loc32, #loc36])
#loc43 = loc(fused<#di_lexical_block13>[#loc9])
#loc44 = loc(fused<#di_lexical_block13>[#loc10])
#loc45 = loc(fused<#di_lexical_block14>[#loc14])
#loc46 = loc(fused[#loc34, #loc38])
#loc47 = loc(fused[#loc35, #loc39])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 27>
#loc48 = loc(fused<#di_lexical_block16>[#loc18])
#loc49 = loc(fused<#di_lexical_block17>[#loc19])
#loc50 = loc(fused<#di_lexical_block17>[#loc20])
