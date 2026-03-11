#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/compact/compact.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/compact/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/compact/compact.cpp":14:0)
#loc3 = loc("tests/app/compact/compact.cpp":18:0)
#loc9 = loc("tests/app/compact/compact.cpp":29:0)
#loc16 = loc("tests/app/compact/main.cpp":16:0)
#loc21 = loc("tests/app/compact/main.cpp":29:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 33>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 29>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "count", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "count", file = #di_file, line = 32, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 33, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_count", file = #di_file1, line = 21, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_count", file = #di_file1, line = 22, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 29, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 3, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 31, arg = 3, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 13, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 15, arg = 2, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 30, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 16>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 29>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 14, arg = 1, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 29, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_derived_type1, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compact_cpu", linkageName = "_Z11compact_cpuPKjPjj", file = #di_file, line = 14, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable14, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compact_dsa", linkageName = "_Z11compact_dsaPKjPjj", file = #di_file, line = 29, scopeLine = 31, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable15, #di_local_variable9, #di_local_variable2, #di_local_variable3>
#loc32 = loc(fused<#di_lexical_block4>[#loc16])
#loc34 = loc(fused<#di_lexical_block6>[#loc21])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc35 = loc(fused<#di_subprogram4>[#loc1])
#loc37 = loc(fused<#di_subprogram5>[#loc9])
#loc41 = loc(fused<#di_lexical_block9>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<30xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 111, 109, 112, 97, 99, 116, 47, 99, 111, 109, 112, 97, 99, 116, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str.2 : memref<16xi8> = dense<[99, 111, 109, 112, 97, 99, 116, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.3 : memref<16xi8> = dense<[99, 111, 109, 112, 97, 99, 116, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z11compact_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) -> i32 {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc46)
    %1 = scf.if %0 -> (i32) {
      scf.yield %c0_i32 : i32 loc(#loc41)
    } else {
      %2 = arith.extui %arg2 : i32 to i64 loc(#loc46)
      %3:2 = scf.while (%arg3 = %c0_i64, %arg4 = %c0_i32) : (i64, i32) -> (i64, i32) {
        %4 = arith.index_cast %arg3 : i64 to index loc(#loc52)
        %5 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc52)
        %6 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc52)
        %7 = scf.if %6 -> (i32) {
          scf.yield %arg4 : i32 loc(#loc52)
        } else {
          %10 = arith.extui %arg4 : i32 to i64 loc(#loc54)
          %11 = arith.index_cast %10 : i64 to index loc(#loc54)
          memref.store %5, %arg1[%11] : memref<?xi32, strided<[1], offset: ?>> loc(#loc54)
          %12 = arith.addi %arg4, %c1_i32 : i32 loc(#loc55)
          scf.yield %12 : i32 loc(#loc56)
        } loc(#loc52)
        %8 = arith.addi %arg3, %c1_i64 : i64 loc(#loc46)
        %9 = arith.cmpi ne, %8, %2 : i64 loc(#loc49)
        scf.condition(%9) %8, %7 : i64, i32 loc(#loc41)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block9>[#loc3]), %arg4: i32 loc(fused<#di_lexical_block9>[#loc3])):
        scf.yield %arg3, %arg4 : i64, i32 loc(#loc41)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc41)
      scf.yield %3#1 : i32 loc(#loc41)
    } loc(#loc41)
    return %1 : i32 loc(#loc36)
  } loc(#loc35)
  handshake.func @_Z11compact_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: i32 loc(fused<#di_subprogram5>[#loc9]), %arg3: i1 loc(fused<#di_subprogram5>[#loc9]), ...) -> (i32, i1) attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["count", "done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc37)
    %1 = handshake.join %0 : none loc(#loc37)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc42)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc42)
    %7 = arith.index_cast %4 : i64 to index loc(#loc42)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc42)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc42)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc42)
    %9 = dataflow.carry %willContinue, %3, %18 : i1, i32, i32 -> i32 loc(#loc42)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : i32, i1 -> i32, i1 loc(#loc42)
    handshake.sink %afterCond_1 : i1 loc(#loc42)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %9 : i32 loc(#loc42)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc42)
    %dataResult, %addressResults = handshake.load [%afterValue] %23#0, %25 : index, i32 loc(#loc53)
    %11 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc53)
    %12 = arith.extui %afterValue_0 : i32 to i64 loc(#loc57)
    %13 = arith.index_cast %12 : i64 to index loc(#loc57)
    %dataResult_4, %addressResult = handshake.store [%13] %dataResult, %falseResult_8 : index, i32 loc(#loc57)
    %14 = arith.addi %afterValue_0, %2 : i32 loc(#loc58)
    %15 = handshake.constant %10 {value = 0 : index} : index loc(#loc53)
    %16 = handshake.constant %10 {value = 1 : index} : index loc(#loc53)
    %17 = arith.select %11, %16, %15 : index loc(#loc53)
    %18 = handshake.mux %17 [%14, %afterValue_0] : index, i32 loc(#loc53)
    %19 = handshake.constant %1 {value = 0 : index} : index loc(#loc42)
    %20 = handshake.constant %1 {value = 1 : index} : index loc(#loc42)
    %21 = arith.select %5, %20, %19 : index loc(#loc42)
    %22 = handshake.mux %21 [%falseResult_3, %3] : index, i32 loc(#loc42)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %24 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc37)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc42)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %23#1 : none loc(#loc42)
    %26 = handshake.mux %21 [%falseResult_6, %trueResult] : index, none loc(#loc42)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc42)
    %trueResult_7, %falseResult_8 = handshake.cond_br %11, %27 : none loc(#loc53)
    %28 = handshake.constant %27 {value = 0 : index} : index loc(#loc53)
    %29 = handshake.constant %27 {value = 1 : index} : index loc(#loc53)
    %30 = arith.select %11, %29, %28 : index loc(#loc53)
    %31 = handshake.mux %30 [%24, %trueResult_7] : index, none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %31 : none loc(#loc42)
    %32 = handshake.mux %21 [%falseResult_10, %trueResult] : index, none loc(#loc42)
    %33 = handshake.join %26, %32 : none, none loc(#loc37)
    %34 = handshake.constant %33 {value = true} : i1 loc(#loc37)
    handshake.return %22, %34 : i32, i1 loc(#loc37)
  } loc(#loc37)
  handshake.func @_Z11compact_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: i32 loc(fused<#di_subprogram5>[#loc9]), %arg3: none loc(fused<#di_subprogram5>[#loc9]), ...) -> (i32, none) attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["count", "done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc37)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc42)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc42)
    %6 = arith.index_cast %3 : i64 to index loc(#loc42)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc42)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc42)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc42)
    %8 = dataflow.carry %willContinue, %2, %17 : i1, i32, i32 -> i32 loc(#loc42)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc42)
    handshake.sink %afterCond_1 : i1 loc(#loc42)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %8 : i32 loc(#loc42)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc42)
    %dataResult, %addressResults = handshake.load [%afterValue] %22#0, %24 : index, i32 loc(#loc53)
    %10 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc53)
    %11 = arith.extui %afterValue_0 : i32 to i64 loc(#loc57)
    %12 = arith.index_cast %11 : i64 to index loc(#loc57)
    %dataResult_4, %addressResult = handshake.store [%12] %dataResult, %falseResult_8 : index, i32 loc(#loc57)
    %13 = arith.addi %afterValue_0, %1 : i32 loc(#loc58)
    %14 = handshake.constant %9 {value = 0 : index} : index loc(#loc53)
    %15 = handshake.constant %9 {value = 1 : index} : index loc(#loc53)
    %16 = arith.select %10, %15, %14 : index loc(#loc53)
    %17 = handshake.mux %16 [%13, %afterValue_0] : index, i32 loc(#loc53)
    %18 = handshake.constant %0 {value = 0 : index} : index loc(#loc42)
    %19 = handshake.constant %0 {value = 1 : index} : index loc(#loc42)
    %20 = arith.select %4, %19, %18 : index loc(#loc42)
    %21 = handshake.mux %20 [%falseResult_3, %2] : index, i32 loc(#loc42)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %23 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc37)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc42)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %22#1 : none loc(#loc42)
    %25 = handshake.mux %20 [%falseResult_6, %trueResult] : index, none loc(#loc42)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc42)
    %trueResult_7, %falseResult_8 = handshake.cond_br %10, %26 : none loc(#loc53)
    %27 = handshake.constant %26 {value = 0 : index} : index loc(#loc53)
    %28 = handshake.constant %26 {value = 1 : index} : index loc(#loc53)
    %29 = arith.select %10, %28, %27 : index loc(#loc53)
    %30 = handshake.mux %29 [%23, %trueResult_7] : index, none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %30 : none loc(#loc42)
    %31 = handshake.mux %20 [%falseResult_10, %trueResult] : index, none loc(#loc42)
    %32 = handshake.join %25, %31 : none, none loc(#loc37)
    handshake.return %21, %32 : i32, none loc(#loc38)
  } loc(#loc37)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc27)
    %c2_i32 = arith.constant 2 : i32 loc(#loc27)
    %1 = ub.poison : i32 loc(#loc27)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str.3 : memref<16xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<16xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %13 = arith.trunci %arg0 : i64 to i32 loc(#loc43)
      %14 = arith.remui %13, %c5_i32 : i32 loc(#loc43)
      %15 = arith.cmpi eq, %14, %c0_i32 : i32 loc(#loc43)
      %16 = arith.remui %13, %c100_i32 : i32 loc(#loc43)
      %17 = arith.select %15, %c0_i32, %16 : i32 loc(#loc43)
      %18 = arith.index_cast %arg0 : i64 to index loc(#loc43)
      memref.store %17, %alloca[%18] : memref<1024xi32> loc(#loc43)
      %19 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
      %20 = arith.cmpi ne, %19, %c1024_i64 : i64 loc(#loc44)
      scf.condition(%20) %19 : i64 loc(#loc32)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc16])):
      scf.yield %arg0 : i64 loc(#loc32)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc32)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %cast_2 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %5 = call @_Z11compact_cpuPKjPjj(%cast, %cast_2, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> i32 loc(#loc28)
    %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc29)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc29)
    %6:2 = handshake.esi_instance @_Z11compact_dsaPKjPjj_esi "_Z11compact_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>) loc(#loc29)
    %rawOutput, %valid = esi.unwrap.vr %6#0, %true : i32 loc(#loc29)
    %rawOutput_10, %valid_11 = esi.unwrap.vr %6#1, %true : i1 loc(#loc29)
    %7 = arith.cmpi eq, %5, %rawOutput : i32 loc(#loc33)
    %cast_12 = memref.cast %2 : memref<16xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc33)
    %8 = arith.cmpi ne, %5, %rawOutput : i32 loc(#loc33)
    %9 = arith.extui %8 : i1 to i32 loc(#loc33)
    %10:3 = scf.if %7 -> (i32, memref<?xi8, strided<[1], offset: ?>>, i32) {
      %13 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc40)
      %14:2 = scf.if %13 -> (i1, i32) {
        scf.yield %false, %c0_i32 : i1, i32 loc(#loc34)
      } else {
        %17:2 = scf.while (%arg0 = %c0_i32) : (i32) -> (i32, i32) {
          %20 = arith.extui %arg0 : i32 to i64 loc(#loc48)
          %21 = arith.index_cast %20 : i64 to index loc(#loc48)
          %22 = memref.load %alloca_0[%21] : memref<1024xi32> loc(#loc48)
          %23 = memref.load %alloca_1[%21] : memref<1024xi32> loc(#loc48)
          %24 = arith.cmpi eq, %22, %23 : i32 loc(#loc48)
          %25:3 = scf.if %24 -> (i32, i32, i32) {
            %27 = arith.addi %arg0, %c1_i32 : i32 loc(#loc40)
            %28 = arith.cmpi eq, %27, %5 : i32 loc(#loc40)
            %29 = arith.extui %28 : i1 to i32 loc(#loc34)
            %30 = arith.cmpi ne, %27, %5 : i32 loc(#loc45)
            %31 = arith.extui %30 : i1 to i32 loc(#loc34)
            scf.yield %27, %29, %31 : i32, i32, i32 loc(#loc48)
          } else {
            scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc48)
          } loc(#loc48)
          %26 = arith.trunci %25#2 : i32 to i1 loc(#loc34)
          scf.condition(%26) %25#0, %25#1 : i32, i32 loc(#loc34)
        } do {
        ^bb0(%arg0: i32 loc(fused<#di_lexical_block6>[#loc21]), %arg1: i32 loc(fused<#di_lexical_block6>[#loc21])):
          scf.yield %arg0 : i32 loc(#loc34)
        } loc(#loc34)
        %18 = arith.index_castui %17#1 : i32 to index loc(#loc34)
        %19:2 = scf.index_switch %18 -> i1, i32 
        case 1 {
          scf.yield %false, %c0_i32 : i1, i32 loc(#loc34)
        }
        default {
          %intptr = memref.extract_aligned_pointer_as_index %2 : memref<16xi8> -> index loc(#loc50)
          %20 = arith.index_cast %intptr : index to i64 loc(#loc50)
          %21 = llvm.inttoptr %20 : i64 to !llvm.ptr loc(#loc50)
          %22 = llvm.call @puts(%21) : (!llvm.ptr) -> i32 loc(#loc50)
          scf.yield %true, %c1_i32 : i1, i32 loc(#loc51)
        } loc(#loc34)
        scf.yield %19#0, %19#1 : i1, i32 loc(#loc34)
      } loc(#loc34)
      %cast_13 = memref.cast %3 : memref<16xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc2)
      %15 = arith.xori %14#0, %true : i1 loc(#loc2)
      %16 = arith.extui %15 : i1 to i32 loc(#loc2)
      scf.yield %14#1, %cast_13, %16 : i32, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc33)
    } else {
      scf.yield %1, %cast_12, %c1_i32 : i32, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc33)
    } loc(#loc33)
    %11 = arith.index_castui %10#2 : i32 to index loc(#loc2)
    %12 = scf.index_switch %11 -> i32 
    case 0 {
      scf.yield %10#0 : i32 loc(#loc2)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %10#1 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc30)
      %13 = arith.index_cast %intptr : index to i64 loc(#loc30)
      %14 = llvm.inttoptr %13 : i64 to !llvm.ptr loc(#loc30)
      %15 = llvm.call @puts(%14) : (!llvm.ptr) -> i32 loc(#loc30)
      scf.yield %9 : i32 loc(#loc31)
    } loc(#loc2)
    return %12 : i32 loc(#loc31)
  } loc(#loc27)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/compact/compact.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/compact/compact.cpp":19:0)
#loc5 = loc("tests/app/compact/compact.cpp":20:0)
#loc6 = loc("tests/app/compact/compact.cpp":21:0)
#loc7 = loc("tests/app/compact/compact.cpp":22:0)
#loc8 = loc("tests/app/compact/compact.cpp":24:0)
#loc10 = loc("tests/app/compact/compact.cpp":33:0)
#loc11 = loc("tests/app/compact/compact.cpp":34:0)
#loc12 = loc("tests/app/compact/compact.cpp":35:0)
#loc13 = loc("tests/app/compact/compact.cpp":36:0)
#loc14 = loc("tests/app/compact/compact.cpp":39:0)
#loc15 = loc("tests/app/compact/main.cpp":5:0)
#loc17 = loc("tests/app/compact/main.cpp":17:0)
#loc18 = loc("tests/app/compact/main.cpp":21:0)
#loc19 = loc("tests/app/compact/main.cpp":22:0)
#loc20 = loc("tests/app/compact/main.cpp":24:0)
#loc22 = loc("tests/app/compact/main.cpp":30:0)
#loc23 = loc("tests/app/compact/main.cpp":31:0)
#loc24 = loc("tests/app/compact/main.cpp":32:0)
#loc25 = loc("tests/app/compact/main.cpp":0:0)
#loc26 = loc("tests/app/compact/main.cpp":38:0)
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 24>
#loc27 = loc(fused<#di_subprogram3>[#loc15])
#loc28 = loc(fused<#di_subprogram3>[#loc18])
#loc29 = loc(fused<#di_subprogram3>[#loc19])
#loc30 = loc(fused<#di_subprogram3>[#loc25])
#loc31 = loc(fused<#di_subprogram3>[#loc26])
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 16>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 29>
#loc33 = loc(fused<#di_lexical_block5>[#loc20])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 33>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 16>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 29>
#loc36 = loc(fused<#di_subprogram4>[#loc8])
#loc38 = loc(fused<#di_subprogram5>[#loc14])
#loc39 = loc(fused<#di_lexical_block7>[#loc16])
#loc40 = loc(fused<#di_lexical_block8>[#loc21])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 18>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 33>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 30>
#loc42 = loc(fused<#di_lexical_block10>[#loc10])
#loc43 = loc(fused<#di_lexical_block11>[#loc17])
#loc44 = loc(fused[#loc32, #loc39])
#loc45 = loc(fused[#loc34, #loc40])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 18>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 33>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 30>
#loc46 = loc(fused<#di_lexical_block13>[#loc3])
#loc47 = loc(fused<#di_lexical_block14>[#loc10])
#loc48 = loc(fused<#di_lexical_block15>[#loc22])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 19>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 34>
#loc49 = loc(fused[#loc41, #loc46])
#loc50 = loc(fused<#di_lexical_block18>[#loc23])
#loc51 = loc(fused<#di_lexical_block18>[#loc24])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 19>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 34>
#loc52 = loc(fused<#di_lexical_block19>[#loc4])
#loc53 = loc(fused<#di_lexical_block20>[#loc11])
#loc54 = loc(fused<#di_lexical_block21>[#loc5])
#loc55 = loc(fused<#di_lexical_block21>[#loc6])
#loc56 = loc(fused<#di_lexical_block21>[#loc7])
#loc57 = loc(fused<#di_lexical_block22>[#loc12])
#loc58 = loc(fused<#di_lexical_block22>[#loc13])
