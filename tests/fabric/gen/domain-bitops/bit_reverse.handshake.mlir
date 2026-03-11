#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/bit_reverse/bit_reverse.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/bit_reverse/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/bit_reverse/bit_reverse.cpp":14:0)
#loc3 = loc("tests/app/bit_reverse/bit_reverse.cpp":17:0)
#loc5 = loc("tests/app/bit_reverse/bit_reverse.cpp":21:0)
#loc10 = loc("tests/app/bit_reverse/bit_reverse.cpp":32:0)
#loc25 = loc("tests/app/bit_reverse/main.cpp":17:0)
#loc29 = loc("tests/app/bit_reverse/main.cpp":32:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 37>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 32>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 37>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 37>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 37, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 32, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 21>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 41>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 3, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "result", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 34, arg = 3, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file, line = 38, type = #di_derived_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "result", file = #di_file, line = 39, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_reversed", file = #di_file1, line = 22, type = #di_composite_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_reversed", file = #di_file1, line = 23, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_reversed", file = #di_file, line = 15, arg = 2, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "bit", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_reversed", file = #di_file, line = 33, arg = 2, type = #di_derived_type5>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "bit", file = #di_file, line = 41, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable11, #di_local_variable2, #di_local_variable12, #di_local_variable13, #di_local_variable3>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 17>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 32>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 14, arg = 1, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 32, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bit_reverse_cpu", linkageName = "_Z15bit_reverse_cpuPKjPjj", file = #di_file, line = 14, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable18, #di_local_variable14, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable6, #di_local_variable15>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bit_reverse_dsa", linkageName = "_Z15bit_reverse_dsaPKjPjj", file = #di_file, line = 32, scopeLine = 34, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable16, #di_local_variable7, #di_local_variable1, #di_local_variable8, #di_local_variable9, #di_local_variable17>
#loc46 = loc(fused<#di_lexical_block10>[#loc25])
#loc47 = loc(fused<#di_lexical_block11>[#loc29])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 17>
#loc48 = loc(fused<#di_subprogram4>[#loc1])
#loc50 = loc(fused<#di_subprogram5>[#loc10])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 17>
#loc54 = loc(fused<#di_lexical_block14>[#loc3])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 17>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 21>
#loc69 = loc(fused<#di_lexical_block24>[#loc5])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 105, 116, 95, 114, 101, 118, 101, 114, 115, 101, 47, 98, 105, 116, 95, 114, 101, 118, 101, 114, 115, 101, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<20xi8> = dense<[98, 105, 116, 95, 114, 101, 118, 101, 114, 115, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[98, 105, 116, 95, 114, 101, 118, 101, 114, 115, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z15bit_reverse_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc59)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc59)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc62)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
        %5:3 = scf.while (%arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %4) : (i32, i32, i32) -> (i32, i32, i32) {
          %8 = arith.shli %arg5, %c1_i32 : i32 loc(#loc72)
          %9 = arith.andi %arg6, %c1_i32 : i32 loc(#loc72)
          %10 = arith.ori %8, %9 : i32 loc(#loc72)
          %11 = arith.shrui %arg6, %c1_i32 : i32 loc(#loc73)
          %12 = arith.addi %arg4, %c1_i32 : i32 loc(#loc71)
          %13 = arith.cmpi ne, %12, %c32_i32 : i32 loc(#loc74)
          scf.condition(%13) %12, %10, %11 : i32, i32, i32 loc(#loc69)
        } do {
        ^bb0(%arg4: i32 loc(fused<#di_lexical_block24>[#loc5]), %arg5: i32 loc(fused<#di_lexical_block24>[#loc5]), %arg6: i32 loc(fused<#di_lexical_block24>[#loc5])):
          scf.yield %arg4, %arg5, %arg6 : i32, i32, i32 loc(#loc69)
        } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc69)
        memref.store %5#1, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc63)
        %6 = arith.addi %arg3, %c1_i64 : i64 loc(#loc59)
        %7 = arith.cmpi ne, %6, %1 : i64 loc(#loc64)
        scf.condition(%7) %6 : i64 loc(#loc54)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block14>[#loc3])):
        scf.yield %arg3 : i64 loc(#loc54)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc54)
    } loc(#loc54)
    return loc(#loc49)
  } loc(#loc48)
  handshake.func @_Z15bit_reverse_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: i1 loc(fused<#di_subprogram5>[#loc10]), ...) -> i1 attributes {argNames = ["input_data", "output_reversed", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc50)
    %1 = handshake.join %0 : none loc(#loc50)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 32 : i32} : i32 loc(#loc2)
    %6 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc60)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc55)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc55)
    %8 = arith.index_cast %4 : i64 to index loc(#loc55)
    %9 = arith.index_cast %arg2 : i32 to index loc(#loc55)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc55)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc55)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc55)
    %dataResult, %addressResults = handshake.load [%afterValue] %20#0, %22 : index, i32 loc(#loc65)
    %11 = handshake.constant %10 {value = 1 : index} : index loc(#loc70)
    %12 = arith.index_cast %3 : i32 to index loc(#loc70)
    %13 = arith.index_cast %5 : i32 to index loc(#loc70)
    %index_0, %willContinue_1 = dataflow.stream %12, %11, %13 {step_op = "+=", stop_cond = "!="} loc(#loc70)
    %14 = dataflow.carry %willContinue_1, %3, %18 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_2, %afterCond_3 = dataflow.gate %14, %willContinue_1 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_3 : i1 loc(#loc70)
    %trueResult_4, %falseResult_5 = handshake.cond_br %willContinue_1, %14 : i32 loc(#loc70)
    %15 = dataflow.carry %willContinue_1, %dataResult, %19 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_6, %afterCond_7 = dataflow.gate %15, %willContinue_1 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_7 : i1 loc(#loc70)
    %16 = arith.shli %afterValue_2, %2 : i32 loc(#loc75)
    %17 = arith.andi %afterValue_6, %2 : i32 loc(#loc75)
    %18 = arith.ori %16, %17 : i32 loc(#loc75)
    %19 = arith.shrui %afterValue_6, %2 : i32 loc(#loc76)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %falseResult_5, %27 : index, i32 loc(#loc66)
    %20:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc50)
    %21 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc50)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc55)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %20#1 : none loc(#loc55)
    %23 = handshake.constant %1 {value = 0 : index} : index loc(#loc55)
    %24 = handshake.constant %1 {value = 1 : index} : index loc(#loc55)
    %25 = arith.select %6, %24, %23 : index loc(#loc55)
    %26 = handshake.mux %25 [%falseResult_10, %trueResult] : index, none loc(#loc55)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc55)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %21 : none loc(#loc55)
    %28 = handshake.mux %25 [%falseResult_12, %trueResult] : index, none loc(#loc55)
    %29 = handshake.join %26, %28 : none, none loc(#loc50)
    %30 = handshake.constant %29 {value = true} : i1 loc(#loc50)
    handshake.return %30 : i1 loc(#loc50)
  } loc(#loc50)
  handshake.func @_Z15bit_reverse_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: none loc(fused<#di_subprogram5>[#loc10]), ...) -> none attributes {argNames = ["input_data", "output_reversed", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc50)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 32 : i32} : i32 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc60)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc55)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc55)
    %7 = arith.index_cast %3 : i64 to index loc(#loc55)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc55)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc55)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc55)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc55)
    %dataResult, %addressResults = handshake.load [%afterValue] %19#0, %21 : index, i32 loc(#loc65)
    %10 = handshake.constant %9 {value = 1 : index} : index loc(#loc70)
    %11 = arith.index_cast %2 : i32 to index loc(#loc70)
    %12 = arith.index_cast %4 : i32 to index loc(#loc70)
    %index_0, %willContinue_1 = dataflow.stream %11, %10, %12 {step_op = "+=", stop_cond = "!="} loc(#loc70)
    %13 = dataflow.carry %willContinue_1, %2, %17 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_2, %afterCond_3 = dataflow.gate %13, %willContinue_1 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_3 : i1 loc(#loc70)
    %trueResult_4, %falseResult_5 = handshake.cond_br %willContinue_1, %13 : i32 loc(#loc70)
    %14 = dataflow.carry %willContinue_1, %dataResult, %18 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_6, %afterCond_7 = dataflow.gate %14, %willContinue_1 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_7 : i1 loc(#loc70)
    %15 = arith.shli %afterValue_2, %1 : i32 loc(#loc75)
    %16 = arith.andi %afterValue_6, %1 : i32 loc(#loc75)
    %17 = arith.ori %15, %16 : i32 loc(#loc75)
    %18 = arith.shrui %afterValue_6, %1 : i32 loc(#loc76)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %falseResult_5, %26 : index, i32 loc(#loc66)
    %19:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc50)
    %20 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc50)
    %21 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc55)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %19#1 : none loc(#loc55)
    %22 = handshake.constant %0 {value = 0 : index} : index loc(#loc55)
    %23 = handshake.constant %0 {value = 1 : index} : index loc(#loc55)
    %24 = arith.select %5, %23, %22 : index loc(#loc55)
    %25 = handshake.mux %24 [%falseResult_10, %trueResult] : index, none loc(#loc55)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc55)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %20 : none loc(#loc55)
    %27 = handshake.mux %24 [%falseResult_12, %trueResult] : index, none loc(#loc55)
    %28 = handshake.join %25, %27 : none, none loc(#loc50)
    handshake.return %28 : none loc(#loc51)
  } loc(#loc50)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc35)
    %false = arith.constant false loc(#loc35)
    %0 = seq.const_clock  low loc(#loc35)
    %c2_i32 = arith.constant 2 : i32 loc(#loc35)
    %1 = ub.poison : i64 loc(#loc35)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c5 = arith.constant 5 : index loc(#loc36)
    %c4 = arith.constant 4 : index loc(#loc37)
    %c3 = arith.constant 3 : index loc(#loc38)
    %c2 = arith.constant 2 : index loc(#loc39)
    %c1 = arith.constant 1 : index loc(#loc40)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c-2147483648_i32 = arith.constant -2147483648 : i32 loc(#loc2)
    %c-252645136_i32 = arith.constant -252645136 : i32 loc(#loc2)
    %c305419896_i32 = arith.constant 305419896 : i32 loc(#loc2)
    %c6_i64 = arith.constant 6 : i64 loc(#loc2)
    %c-1412623820_i32 = arith.constant -1412623820 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    memref.store %c0_i32, %alloca[%c0] : memref<256xi32> loc(#loc41)
    memref.store %c-1_i32, %alloca[%c1] : memref<256xi32> loc(#loc40)
    memref.store %c-2147483648_i32, %alloca[%c2] : memref<256xi32> loc(#loc39)
    memref.store %c1_i32, %alloca[%c3] : memref<256xi32> loc(#loc38)
    memref.store %c-252645136_i32, %alloca[%c4] : memref<256xi32> loc(#loc37)
    memref.store %c305419896_i32, %alloca[%c5] : memref<256xi32> loc(#loc36)
    %4 = scf.while (%arg0 = %c6_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc56)
      %11 = arith.muli %10, %c-1412623820_i32 : i32 loc(#loc56)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc56)
      memref.store %11, %alloca[%12] : memref<256xi32> loc(#loc56)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc52)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc57)
      scf.condition(%14) %13 : i64 loc(#loc46)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc25])):
      scf.yield %arg0 : i64 loc(#loc46)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc42)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc42)
    call @_Z15bit_reverse_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc42)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc43)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc43)
    %5 = handshake.esi_instance @_Z15bit_reverse_dsaPKjPjj_esi "_Z15bit_reverse_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc43)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc43)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc61)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc61)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc61)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc61)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc53)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc53)
        %18 = arith.extui %17 : i1 to i32 loc(#loc47)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc58)
        %20 = arith.extui %19 : i1 to i32 loc(#loc47)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc61)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc61)
      } loc(#loc61)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc47)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc47)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc29]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc29]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc29])):
      scf.yield %arg0 : i64 loc(#loc47)
    } loc(#loc47)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc47)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc47)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc67)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc67)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc67)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc67)
      scf.yield %c1_i32 : i32 loc(#loc68)
    } loc(#loc47)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc44)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc44)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc44)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc44)
    } loc(#loc2)
    return %9 : i32 loc(#loc45)
  } loc(#loc35)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/bit_reverse/bit_reverse.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/bit_reverse/bit_reverse.cpp":18:0)
#loc6 = loc("tests/app/bit_reverse/bit_reverse.cpp":22:0)
#loc7 = loc("tests/app/bit_reverse/bit_reverse.cpp":23:0)
#loc8 = loc("tests/app/bit_reverse/bit_reverse.cpp":26:0)
#loc9 = loc("tests/app/bit_reverse/bit_reverse.cpp":28:0)
#loc11 = loc("tests/app/bit_reverse/bit_reverse.cpp":37:0)
#loc12 = loc("tests/app/bit_reverse/bit_reverse.cpp":38:0)
#loc13 = loc("tests/app/bit_reverse/bit_reverse.cpp":41:0)
#loc14 = loc("tests/app/bit_reverse/bit_reverse.cpp":42:0)
#loc15 = loc("tests/app/bit_reverse/bit_reverse.cpp":43:0)
#loc16 = loc("tests/app/bit_reverse/bit_reverse.cpp":46:0)
#loc17 = loc("tests/app/bit_reverse/bit_reverse.cpp":48:0)
#loc18 = loc("tests/app/bit_reverse/main.cpp":5:0)
#loc19 = loc("tests/app/bit_reverse/main.cpp":15:0)
#loc20 = loc("tests/app/bit_reverse/main.cpp":14:0)
#loc21 = loc("tests/app/bit_reverse/main.cpp":13:0)
#loc22 = loc("tests/app/bit_reverse/main.cpp":12:0)
#loc23 = loc("tests/app/bit_reverse/main.cpp":11:0)
#loc24 = loc("tests/app/bit_reverse/main.cpp":10:0)
#loc26 = loc("tests/app/bit_reverse/main.cpp":18:0)
#loc27 = loc("tests/app/bit_reverse/main.cpp":26:0)
#loc28 = loc("tests/app/bit_reverse/main.cpp":29:0)
#loc30 = loc("tests/app/bit_reverse/main.cpp":33:0)
#loc31 = loc("tests/app/bit_reverse/main.cpp":34:0)
#loc32 = loc("tests/app/bit_reverse/main.cpp":35:0)
#loc33 = loc("tests/app/bit_reverse/main.cpp":39:0)
#loc34 = loc("tests/app/bit_reverse/main.cpp":41:0)
#loc35 = loc(fused<#di_subprogram3>[#loc18])
#loc36 = loc(fused<#di_subprogram3>[#loc19])
#loc37 = loc(fused<#di_subprogram3>[#loc20])
#loc38 = loc(fused<#di_subprogram3>[#loc21])
#loc39 = loc(fused<#di_subprogram3>[#loc22])
#loc40 = loc(fused<#di_subprogram3>[#loc23])
#loc41 = loc(fused<#di_subprogram3>[#loc24])
#loc42 = loc(fused<#di_subprogram3>[#loc27])
#loc43 = loc(fused<#di_subprogram3>[#loc28])
#loc44 = loc(fused<#di_subprogram3>[#loc33])
#loc45 = loc(fused<#di_subprogram3>[#loc34])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 17>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 32>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 37>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 17>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 32>
#loc49 = loc(fused<#di_subprogram4>[#loc9])
#loc51 = loc(fused<#di_subprogram5>[#loc17])
#loc52 = loc(fused<#di_lexical_block12>[#loc25])
#loc53 = loc(fused<#di_lexical_block13>[#loc29])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 37>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 33>
#loc55 = loc(fused<#di_lexical_block15>[#loc11])
#loc56 = loc(fused<#di_lexical_block16>[#loc26])
#loc57 = loc(fused[#loc46, #loc52])
#loc58 = loc(fused[#loc47, #loc53])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 37>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 33>
#loc59 = loc(fused<#di_lexical_block18>[#loc3])
#loc60 = loc(fused<#di_lexical_block19>[#loc11])
#loc61 = loc(fused<#di_lexical_block20>[#loc30])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 41>
#loc62 = loc(fused<#di_lexical_block21>[#loc4])
#loc63 = loc(fused<#di_lexical_block21>[#loc8])
#loc64 = loc(fused[#loc54, #loc59])
#loc65 = loc(fused<#di_lexical_block22>[#loc12])
#loc66 = loc(fused<#di_lexical_block22>[#loc16])
#loc67 = loc(fused<#di_lexical_block23>[#loc31])
#loc68 = loc(fused<#di_lexical_block23>[#loc32])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 21>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 41>
#loc70 = loc(fused<#di_lexical_block25>[#loc13])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 21>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 41>
#loc71 = loc(fused<#di_lexical_block26>[#loc5])
#loc72 = loc(fused<#di_lexical_block28>[#loc6])
#loc73 = loc(fused<#di_lexical_block28>[#loc7])
#loc74 = loc(fused[#loc69, #loc71])
#loc75 = loc(fused<#di_lexical_block29>[#loc14])
#loc76 = loc(fused<#di_lexical_block29>[#loc15])
