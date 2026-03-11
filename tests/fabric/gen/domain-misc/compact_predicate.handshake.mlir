#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/compact_predicate/compact_predicate.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/compact_predicate/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/compact_predicate/compact_predicate.cpp":14:0)
#loc3 = loc("tests/app/compact_predicate/compact_predicate.cpp":19:0)
#loc9 = loc("tests/app/compact_predicate/compact_predicate.cpp":30:0)
#loc16 = loc("tests/app/compact_predicate/main.cpp":19:0)
#loc22 = loc("tests/app/compact_predicate/main.cpp":33:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 19>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 35>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 19>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 33>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "count", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "count", file = #di_file, line = 34, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 35, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 19, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_count", file = #di_file1, line = 25, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_count", file = #di_file1, line = 26, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 33, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 17, arg = 4, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 33, arg = 4, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "predicate", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 15, type = #di_composite_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 16, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 16, arg = 3, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 32, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 19>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 33>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 14, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "predicate", file = #di_file, line = 15, arg = 2, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 30, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "predicate", file = #di_file, line = 31, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_derived_type1, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compact_predicate_cpu", linkageName = "_Z21compact_predicate_cpuPKjS0_Pjj", file = #di_file, line = 14, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable15, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "compact_predicate_dsa", linkageName = "_Z21compact_predicate_dsaPKjS0_Pjj", file = #di_file, line = 30, scopeLine = 33, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable16, #di_local_variable9, #di_local_variable2, #di_local_variable3>
#loc33 = loc(fused<#di_lexical_block4>[#loc16])
#loc35 = loc(fused<#di_lexical_block6>[#loc22])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 19>
#loc36 = loc(fused<#di_subprogram4>[#loc1])
#loc38 = loc(fused<#di_subprogram5>[#loc9])
#loc42 = loc(fused<#di_lexical_block9>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<50xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 111, 109, 112, 97, 99, 116, 95, 112, 114, 101, 100, 105, 99, 97, 116, 101, 47, 99, 111, 109, 112, 97, 99, 116, 95, 112, 114, 101, 100, 105, 99, 97, 116, 101, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str.2 : memref<26xi8> = dense<[99, 111, 109, 112, 97, 99, 116, 95, 112, 114, 101, 100, 105, 99, 97, 116, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.3 : memref<26xi8> = dense<[99, 111, 109, 112, 97, 99, 116, 95, 112, 114, 101, 100, 105, 99, 97, 116, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z21compact_predicate_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1])) -> i32 {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc48)
    %1 = scf.if %0 -> (i32) {
      scf.yield %c0_i32 : i32 loc(#loc42)
    } else {
      %2 = arith.extui %arg3 : i32 to i64 loc(#loc48)
      %3:2 = scf.while (%arg4 = %c0_i64, %arg5 = %c0_i32) : (i64, i32) -> (i64, i32) {
        %4 = arith.index_cast %arg4 : i64 to index loc(#loc54)
        %5 = memref.load %arg1[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc54)
        %6 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc54)
        %7 = scf.if %6 -> (i32) {
          scf.yield %arg5 : i32 loc(#loc54)
        } else {
          %10 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc56)
          %11 = arith.extui %arg5 : i32 to i64 loc(#loc56)
          %12 = arith.index_cast %11 : i64 to index loc(#loc56)
          memref.store %10, %arg2[%12] : memref<?xi32, strided<[1], offset: ?>> loc(#loc56)
          %13 = arith.addi %arg5, %c1_i32 : i32 loc(#loc57)
          scf.yield %13 : i32 loc(#loc58)
        } loc(#loc54)
        %8 = arith.addi %arg4, %c1_i64 : i64 loc(#loc48)
        %9 = arith.cmpi ne, %8, %2 : i64 loc(#loc51)
        scf.condition(%9) %8, %7 : i64, i32 loc(#loc42)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block9>[#loc3]), %arg5: i32 loc(fused<#di_lexical_block9>[#loc3])):
        scf.yield %arg4, %arg5 : i64, i32 loc(#loc42)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
      scf.yield %3#1 : i32 loc(#loc42)
    } loc(#loc42)
    return %1 : i32 loc(#loc37)
  } loc(#loc36)
  handshake.func @_Z21compact_predicate_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg3: i32 loc(fused<#di_subprogram5>[#loc9]), %arg4: i1 loc(fused<#di_subprogram5>[#loc9]), ...) -> (i32, i1) attributes {argNames = ["input", "predicate", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["count", "done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc38)
    %1 = handshake.join %0 : none loc(#loc38)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc49)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc43)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc43)
    %7 = arith.index_cast %4 : i64 to index loc(#loc43)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc43)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %9 = dataflow.carry %willContinue, %3, %18 : i1, i32, i32 -> i32 loc(#loc43)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : i32, i1 -> i32, i1 loc(#loc43)
    handshake.sink %afterCond_1 : i1 loc(#loc43)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %9 : i32 loc(#loc43)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %24#0, %26 : index, i32 loc(#loc55)
    %11 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc55)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %23#0, %falseResult_14 : index, i32 loc(#loc59)
    %12 = arith.extui %afterValue_0 : i32 to i64 loc(#loc59)
    %13 = arith.index_cast %12 : i64 to index loc(#loc59)
    %dataResult_6, %addressResult = handshake.store [%13] %dataResult_4, %falseResult_10 : index, i32 loc(#loc59)
    %14 = arith.addi %afterValue_0, %2 : i32 loc(#loc60)
    %15 = handshake.constant %10 {value = 0 : index} : index loc(#loc55)
    %16 = handshake.constant %10 {value = 1 : index} : index loc(#loc55)
    %17 = arith.select %11, %16, %15 : index loc(#loc55)
    %18 = handshake.mux %17 [%14, %afterValue_0] : index, i32 loc(#loc55)
    %19 = handshake.constant %1 {value = 0 : index} : index loc(#loc43)
    %20 = handshake.constant %1 {value = 1 : index} : index loc(#loc43)
    %21 = arith.select %5, %20, %19 : index loc(#loc43)
    %22 = handshake.mux %21 [%falseResult_3, %3] : index, i32 loc(#loc43)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_5) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc43)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %24#1 : none loc(#loc43)
    %27 = handshake.mux %21 [%falseResult_8, %trueResult] : index, none loc(#loc43)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc43)
    %trueResult_9, %falseResult_10 = handshake.cond_br %11, %28 : none loc(#loc55)
    %29 = handshake.constant %28 {value = 0 : index} : index loc(#loc55)
    %30 = handshake.constant %28 {value = 1 : index} : index loc(#loc55)
    %31 = arith.select %11, %30, %29 : index loc(#loc55)
    %32 = handshake.mux %31 [%25, %trueResult_9] : index, none loc(#loc55)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %32 : none loc(#loc43)
    %33 = handshake.mux %21 [%falseResult_12, %trueResult] : index, none loc(#loc43)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc43)
    %trueResult_13, %falseResult_14 = handshake.cond_br %11, %34 : none loc(#loc55)
    %35 = handshake.constant %34 {value = 0 : index} : index loc(#loc55)
    %36 = handshake.constant %34 {value = 1 : index} : index loc(#loc55)
    %37 = arith.select %11, %36, %35 : index loc(#loc55)
    %38 = handshake.mux %37 [%23#1, %trueResult_13] : index, none loc(#loc55)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %38 : none loc(#loc43)
    %39 = handshake.mux %21 [%falseResult_16, %trueResult] : index, none loc(#loc43)
    %40 = handshake.join %27, %33, %39 : none, none, none loc(#loc38)
    %41 = handshake.constant %40 {value = true} : i1 loc(#loc38)
    handshake.return %22, %41 : i32, i1 loc(#loc38)
  } loc(#loc38)
  handshake.func @_Z21compact_predicate_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc9]), %arg3: i32 loc(fused<#di_subprogram5>[#loc9]), %arg4: none loc(fused<#di_subprogram5>[#loc9]), ...) -> (i32, none) attributes {argNames = ["input", "predicate", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["count", "done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc38)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc49)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc43)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc43)
    %6 = arith.index_cast %3 : i64 to index loc(#loc43)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc43)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %8 = dataflow.carry %willContinue, %2, %17 : i1, i32, i32 -> i32 loc(#loc43)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc43)
    handshake.sink %afterCond_1 : i1 loc(#loc43)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %8 : i32 loc(#loc43)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %23#0, %25 : index, i32 loc(#loc55)
    %10 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc55)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %22#0, %falseResult_14 : index, i32 loc(#loc59)
    %11 = arith.extui %afterValue_0 : i32 to i64 loc(#loc59)
    %12 = arith.index_cast %11 : i64 to index loc(#loc59)
    %dataResult_6, %addressResult = handshake.store [%12] %dataResult_4, %falseResult_10 : index, i32 loc(#loc59)
    %13 = arith.addi %afterValue_0, %1 : i32 loc(#loc60)
    %14 = handshake.constant %9 {value = 0 : index} : index loc(#loc55)
    %15 = handshake.constant %9 {value = 1 : index} : index loc(#loc55)
    %16 = arith.select %10, %15, %14 : index loc(#loc55)
    %17 = handshake.mux %16 [%13, %afterValue_0] : index, i32 loc(#loc55)
    %18 = handshake.constant %0 {value = 0 : index} : index loc(#loc43)
    %19 = handshake.constant %0 {value = 1 : index} : index loc(#loc43)
    %20 = arith.select %4, %19, %18 : index loc(#loc43)
    %21 = handshake.mux %20 [%falseResult_3, %2] : index, i32 loc(#loc43)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_5) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %24 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc43)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %23#1 : none loc(#loc43)
    %26 = handshake.mux %20 [%falseResult_8, %trueResult] : index, none loc(#loc43)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc43)
    %trueResult_9, %falseResult_10 = handshake.cond_br %10, %27 : none loc(#loc55)
    %28 = handshake.constant %27 {value = 0 : index} : index loc(#loc55)
    %29 = handshake.constant %27 {value = 1 : index} : index loc(#loc55)
    %30 = arith.select %10, %29, %28 : index loc(#loc55)
    %31 = handshake.mux %30 [%24, %trueResult_9] : index, none loc(#loc55)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %31 : none loc(#loc43)
    %32 = handshake.mux %20 [%falseResult_12, %trueResult] : index, none loc(#loc43)
    %33 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc43)
    %trueResult_13, %falseResult_14 = handshake.cond_br %10, %33 : none loc(#loc55)
    %34 = handshake.constant %33 {value = 0 : index} : index loc(#loc55)
    %35 = handshake.constant %33 {value = 1 : index} : index loc(#loc55)
    %36 = arith.select %10, %35, %34 : index loc(#loc55)
    %37 = handshake.mux %36 [%22#1, %trueResult_13] : index, none loc(#loc55)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %37 : none loc(#loc43)
    %38 = handshake.mux %20 [%falseResult_16, %trueResult] : index, none loc(#loc43)
    %39 = handshake.join %26, %32, %38 : none, none, none loc(#loc38)
    handshake.return %21, %39 : i32, none loc(#loc39)
  } loc(#loc38)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc28)
    %c2_i32 = arith.constant 2 : i32 loc(#loc28)
    %1 = ub.poison : i32 loc(#loc28)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str.3 : memref<26xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<26xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc44)
      %14 = arith.trunci %arg0 : i64 to i32 loc(#loc44)
      memref.store %14, %alloca[%13] : memref<1024xi32> loc(#loc44)
      %15 = arith.remui %14, %c3_i32 : i32 loc(#loc45)
      %16 = arith.cmpi ne, %15, %c0_i32 : i32 loc(#loc45)
      %17 = arith.extui %16 : i1 to i32 loc(#loc45)
      memref.store %17, %alloca_0[%13] : memref<1024xi32> loc(#loc45)
      %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc40)
      %19 = arith.cmpi ne, %18, %c1024_i64 : i64 loc(#loc46)
      scf.condition(%19) %18 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc16])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %5 = call @_Z21compact_predicate_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> i32 loc(#loc29)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc30)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %6:2 = handshake.esi_instance @_Z21compact_predicate_dsaPKjS0_Pjj_esi "_Z21compact_predicate_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>) loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %6#0, %true : i32 loc(#loc30)
    %rawOutput_14, %valid_15 = esi.unwrap.vr %6#1, %true : i1 loc(#loc30)
    %7 = arith.cmpi eq, %5, %rawOutput : i32 loc(#loc34)
    %cast_16 = memref.cast %2 : memref<26xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc34)
    %8 = arith.cmpi ne, %5, %rawOutput : i32 loc(#loc34)
    %9 = arith.extui %8 : i1 to i32 loc(#loc34)
    %10:3 = scf.if %7 -> (i32, memref<?xi8, strided<[1], offset: ?>>, i32) {
      %13 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc41)
      %14:2 = scf.if %13 -> (i1, i32) {
        scf.yield %false, %c0_i32 : i1, i32 loc(#loc35)
      } else {
        %17:2 = scf.while (%arg0 = %c0_i32) : (i32) -> (i32, i32) {
          %20 = arith.extui %arg0 : i32 to i64 loc(#loc50)
          %21 = arith.index_cast %20 : i64 to index loc(#loc50)
          %22 = memref.load %alloca_1[%21] : memref<1024xi32> loc(#loc50)
          %23 = memref.load %alloca_2[%21] : memref<1024xi32> loc(#loc50)
          %24 = arith.cmpi eq, %22, %23 : i32 loc(#loc50)
          %25:3 = scf.if %24 -> (i32, i32, i32) {
            %27 = arith.addi %arg0, %c1_i32 : i32 loc(#loc41)
            %28 = arith.cmpi eq, %27, %5 : i32 loc(#loc41)
            %29 = arith.extui %28 : i1 to i32 loc(#loc35)
            %30 = arith.cmpi ne, %27, %5 : i32 loc(#loc47)
            %31 = arith.extui %30 : i1 to i32 loc(#loc35)
            scf.yield %27, %29, %31 : i32, i32, i32 loc(#loc50)
          } else {
            scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc50)
          } loc(#loc50)
          %26 = arith.trunci %25#2 : i32 to i1 loc(#loc35)
          scf.condition(%26) %25#0, %25#1 : i32, i32 loc(#loc35)
        } do {
        ^bb0(%arg0: i32 loc(fused<#di_lexical_block6>[#loc22]), %arg1: i32 loc(fused<#di_lexical_block6>[#loc22])):
          scf.yield %arg0 : i32 loc(#loc35)
        } loc(#loc35)
        %18 = arith.index_castui %17#1 : i32 to index loc(#loc35)
        %19:2 = scf.index_switch %18 -> i1, i32 
        case 1 {
          scf.yield %false, %c0_i32 : i1, i32 loc(#loc35)
        }
        default {
          %intptr = memref.extract_aligned_pointer_as_index %2 : memref<26xi8> -> index loc(#loc52)
          %20 = arith.index_cast %intptr : index to i64 loc(#loc52)
          %21 = llvm.inttoptr %20 : i64 to !llvm.ptr loc(#loc52)
          %22 = llvm.call @puts(%21) : (!llvm.ptr) -> i32 loc(#loc52)
          scf.yield %true, %c1_i32 : i1, i32 loc(#loc53)
        } loc(#loc35)
        scf.yield %19#0, %19#1 : i1, i32 loc(#loc35)
      } loc(#loc35)
      %cast_17 = memref.cast %3 : memref<26xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc2)
      %15 = arith.xori %14#0, %true : i1 loc(#loc2)
      %16 = arith.extui %15 : i1 to i32 loc(#loc2)
      scf.yield %14#1, %cast_17, %16 : i32, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc34)
    } else {
      scf.yield %1, %cast_16, %c1_i32 : i32, memref<?xi8, strided<[1], offset: ?>>, i32 loc(#loc34)
    } loc(#loc34)
    %11 = arith.index_castui %10#2 : i32 to index loc(#loc2)
    %12 = scf.index_switch %11 -> i32 
    case 0 {
      scf.yield %10#0 : i32 loc(#loc2)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %10#1 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc31)
      %13 = arith.index_cast %intptr : index to i64 loc(#loc31)
      %14 = llvm.inttoptr %13 : i64 to !llvm.ptr loc(#loc31)
      %15 = llvm.call @puts(%14) : (!llvm.ptr) -> i32 loc(#loc31)
      scf.yield %9 : i32 loc(#loc32)
    } loc(#loc2)
    return %12 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/compact_predicate/compact_predicate.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/compact_predicate/compact_predicate.cpp":20:0)
#loc5 = loc("tests/app/compact_predicate/compact_predicate.cpp":21:0)
#loc6 = loc("tests/app/compact_predicate/compact_predicate.cpp":22:0)
#loc7 = loc("tests/app/compact_predicate/compact_predicate.cpp":23:0)
#loc8 = loc("tests/app/compact_predicate/compact_predicate.cpp":25:0)
#loc10 = loc("tests/app/compact_predicate/compact_predicate.cpp":35:0)
#loc11 = loc("tests/app/compact_predicate/compact_predicate.cpp":36:0)
#loc12 = loc("tests/app/compact_predicate/compact_predicate.cpp":37:0)
#loc13 = loc("tests/app/compact_predicate/compact_predicate.cpp":38:0)
#loc14 = loc("tests/app/compact_predicate/compact_predicate.cpp":41:0)
#loc15 = loc("tests/app/compact_predicate/main.cpp":5:0)
#loc17 = loc("tests/app/compact_predicate/main.cpp":20:0)
#loc18 = loc("tests/app/compact_predicate/main.cpp":21:0)
#loc19 = loc("tests/app/compact_predicate/main.cpp":25:0)
#loc20 = loc("tests/app/compact_predicate/main.cpp":26:0)
#loc21 = loc("tests/app/compact_predicate/main.cpp":28:0)
#loc23 = loc("tests/app/compact_predicate/main.cpp":34:0)
#loc24 = loc("tests/app/compact_predicate/main.cpp":35:0)
#loc25 = loc("tests/app/compact_predicate/main.cpp":36:0)
#loc26 = loc("tests/app/compact_predicate/main.cpp":0:0)
#loc27 = loc("tests/app/compact_predicate/main.cpp":42:0)
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 28>
#loc28 = loc(fused<#di_subprogram3>[#loc15])
#loc29 = loc(fused<#di_subprogram3>[#loc19])
#loc30 = loc(fused<#di_subprogram3>[#loc20])
#loc31 = loc(fused<#di_subprogram3>[#loc26])
#loc32 = loc(fused<#di_subprogram3>[#loc27])
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 19>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 33>
#loc34 = loc(fused<#di_lexical_block5>[#loc21])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 35>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 19>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 33>
#loc37 = loc(fused<#di_subprogram4>[#loc8])
#loc39 = loc(fused<#di_subprogram5>[#loc14])
#loc40 = loc(fused<#di_lexical_block7>[#loc16])
#loc41 = loc(fused<#di_lexical_block8>[#loc22])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 19>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 35>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 34>
#loc43 = loc(fused<#di_lexical_block10>[#loc10])
#loc44 = loc(fused<#di_lexical_block11>[#loc17])
#loc45 = loc(fused<#di_lexical_block11>[#loc18])
#loc46 = loc(fused[#loc33, #loc40])
#loc47 = loc(fused[#loc35, #loc41])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 19>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 35>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 34>
#loc48 = loc(fused<#di_lexical_block13>[#loc3])
#loc49 = loc(fused<#di_lexical_block14>[#loc10])
#loc50 = loc(fused<#di_lexical_block15>[#loc23])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 20>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 36>
#loc51 = loc(fused[#loc42, #loc48])
#loc52 = loc(fused<#di_lexical_block18>[#loc24])
#loc53 = loc(fused<#di_lexical_block18>[#loc25])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 20>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 36>
#loc54 = loc(fused<#di_lexical_block19>[#loc4])
#loc55 = loc(fused<#di_lexical_block20>[#loc11])
#loc56 = loc(fused<#di_lexical_block21>[#loc5])
#loc57 = loc(fused<#di_lexical_block21>[#loc6])
#loc58 = loc(fused<#di_lexical_block21>[#loc7])
#loc59 = loc(fused<#di_lexical_block22>[#loc12])
#loc60 = loc(fused<#di_lexical_block22>[#loc13])
