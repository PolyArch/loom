#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/bitrev/bitrev.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/bitrev/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/bitrev/bitrev.cpp":13:0)
#loc3 = loc("tests/app/bitrev/bitrev.cpp":16:0)
#loc4 = loc("tests/app/bitrev/bitrev.cpp":22:0)
#loc9 = loc("tests/app/bitrev/bitrev.cpp":34:0)
#loc17 = loc("tests/app/bitrev/main.cpp":15:0)
#loc21 = loc("tests/app/bitrev/main.cpp":23:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 4096, elements = #llvm.di_subrange<count = 128 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 16>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 39>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 15>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 23>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 16>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 39>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 11, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 12, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 16>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 39>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 14, arg = 2, type = #di_derived_type4>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 16, type = #di_derived_type5>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 35, arg = 2, type = #di_derived_type4>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 39, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 15, type = #di_derived_type5>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 23, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 13, arg = 1, type = #di_derived_type6>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 15, arg = 3, type = #di_derived_type7>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "j", file = #di_file, line = 17, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "k", file = #di_file, line = 18, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "m", file = #di_file, line = 19, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 34, arg = 1, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 36, arg = 3, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "j", file = #di_file, line = 40, type = #di_derived_type5>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "k", file = #di_file, line = 41, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "m", file = #di_file, line = 42, type = #di_derived_type5>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 7, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type4, #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bitrev_cpu", linkageName = "_Z10bitrev_cpuPKfPfj", file = #di_file, line = 13, scopeLine = 15, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable9, #di_local_variable3, #di_local_variable10, #di_local_variable4, #di_local_variable11, #di_local_variable12, #di_local_variable13>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bitrev_dsa", linkageName = "_Z10bitrev_dsaPKfPfj", file = #di_file, line = 34, scopeLine = 36, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable14, #di_local_variable5, #di_local_variable15, #di_local_variable6, #di_local_variable16, #di_local_variable17, #di_local_variable18>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable19, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable7, #di_local_variable8>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 16>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 15>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 23>
#loc27 = loc(fused<#di_subprogram3>[#loc1])
#loc29 = loc(fused<#di_subprogram4>[#loc9])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 16>
#loc36 = loc(fused<#di_lexical_block8>[#loc3])
#loc38 = loc(fused<#di_lexical_block10>[#loc17])
#loc39 = loc(fused<#di_lexical_block11>[#loc21])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 16>
#loc44 = loc(fused<#di_lexical_block16>[#loc4])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 105, 116, 114, 101, 118, 47, 98, 105, 116, 114, 101, 118, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<15xi8> = dense<[98, 105, 116, 114, 101, 118, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[98, 105, 116, 114, 101, 118, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z10bitrev_cpuPKfPfj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg2: i32 loc(fused<#di_subprogram3>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc40)
    scf.if %0 {
    } else {
      %1 = arith.shrui %arg2, %c1_i32 : i32 loc(#loc2)
      %2 = arith.cmpi eq, %1, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg2 : i32 to i64 loc(#loc40)
      %4 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %5 = scf.if %2 -> (i64) {
          scf.yield %c0_i64 : i64 loc(#loc44)
        } else {
          %11 = arith.trunci %arg3 : i64 to i32 loc(#loc44)
          %12:3 = scf.while (%arg4 = %1, %arg5 = %11, %arg6 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
            %14 = arith.shli %arg6, %c1_i32 : i32 loc(#loc53)
            %15 = arith.andi %arg5, %c1_i32 : i32 loc(#loc53)
            %16 = arith.ori %15, %14 : i32 loc(#loc53)
            %17 = arith.shrui %arg5, %c1_i32 : i32 loc(#loc54)
            %18 = arith.shrui %arg4, %c1_i32 : i32 loc(#loc45)
            %19 = arith.cmpi ne, %18, %c0_i32 : i32 loc(#loc44)
            scf.condition(%19) %18, %17, %16 : i32, i32, i32 loc(#loc44)
          } do {
          ^bb0(%arg4: i32 loc(fused<#di_lexical_block16>[#loc4]), %arg5: i32 loc(fused<#di_lexical_block16>[#loc4]), %arg6: i32 loc(fused<#di_lexical_block16>[#loc4])):
            scf.yield %arg4, %arg5, %arg6 : i32, i32, i32 loc(#loc44)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc44)
          %13 = arith.extui %12#2 : i32 to i64 loc(#loc46)
          scf.yield %13 : i64 loc(#loc46)
        } loc(#loc44)
        %6 = arith.index_cast %arg3 : i64 to index loc(#loc46)
        %7 = memref.load %arg0[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc46)
        %8 = arith.index_cast %5 : i64 to index loc(#loc46)
        memref.store %7, %arg1[%8] : memref<?xf32, strided<[1], offset: ?>> loc(#loc46)
        %9 = arith.addi %arg3, %c1_i64 : i64 loc(#loc40)
        %10 = arith.cmpi ne, %9, %3 : i64 loc(#loc47)
        scf.condition(%10) %9 : i64 loc(#loc36)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block8>[#loc3])):
        scf.yield %arg3 : i64 loc(#loc36)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc36)
    } loc(#loc36)
    return loc(#loc28)
  } loc(#loc27)
  handshake.func @_Z10bitrev_dsaPKfPfj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg2: i32 loc(fused<#di_subprogram4>[#loc9]), %arg3: i1 loc(fused<#di_subprogram4>[#loc9]), ...) -> i1 attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc29)
    %1 = handshake.join %0 : none loc(#loc29)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc41)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc37)
    %6 = arith.shrui %arg2, %2 : i32 loc(#loc2)
    %7 = arith.cmpi eq, %6, %3 : i32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc37)
    %9 = arith.index_cast %4 : i64 to index loc(#loc37)
    %10 = arith.index_cast %arg2 : i32 to index loc(#loc37)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc37)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc37)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc37)
    %12 = arith.index_cast %afterValue : index to i64 loc(#loc37)
    %13 = dataflow.invariant %afterCond, %7 : i1, i1 -> i1 loc(#loc48)
    %trueResult_0, %falseResult_1 = handshake.cond_br %13, %11 : none loc(#loc48)
    %14 = arith.trunci %12 : i64 to i32 loc(#loc48)
    %15 = dataflow.invariant %afterCond, %6 : i1, i32 -> i32 loc(#loc48)
    %16 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc48)
    %17 = arith.index_cast %15 : i32 to index loc(#loc48)
    %18 = arith.index_cast %3 : i32 to index loc(#loc48)
    %index_2, %willContinue_3 = dataflow.stream %17, %16, %18 {step_op = ">>=", stop_cond = "!="} loc(#loc48)
    %19 = dataflow.carry %willContinue_3, %14, %24 : i1, i32, i32 -> i32 loc(#loc48)
    %afterValue_4, %afterCond_5 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc48)
    handshake.sink %afterCond_5 : i1 loc(#loc48)
    %20 = dataflow.carry %willContinue_3, %3, %23 : i1, i32, i32 -> i32 loc(#loc48)
    %afterValue_6, %afterCond_7 = dataflow.gate %20, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc48)
    handshake.sink %afterCond_7 : i1 loc(#loc48)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %20 : i32 loc(#loc48)
    %21 = arith.shli %afterValue_6, %2 : i32 loc(#loc55)
    %22 = arith.andi %afterValue_4, %2 : i32 loc(#loc55)
    %23 = arith.ori %22, %21 : i32 loc(#loc55)
    %24 = arith.shrui %afterValue_4, %2 : i32 loc(#loc56)
    %25 = arith.extui %falseResult_9 : i32 to i64 loc(#loc49)
    %26 = handshake.constant %11 {value = 0 : index} : index loc(#loc48)
    %27 = handshake.constant %11 {value = 1 : index} : index loc(#loc48)
    %28 = arith.select %13, %27, %26 : index loc(#loc48)
    %29 = handshake.mux %28 [%25, %4] : index, i64 loc(#loc48)
    %dataResult, %addressResults = handshake.load [%afterValue] %31#0, %33 : index, f32 loc(#loc49)
    %30 = arith.index_cast %29 : i64 to index loc(#loc49)
    %dataResult_10, %addressResult = handshake.store [%30] %dataResult, %38 : index, f32 loc(#loc49)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc29)
    %32 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc29)
    %33 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc37)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %31#1 : none loc(#loc37)
    %34 = handshake.constant %1 {value = 0 : index} : index loc(#loc37)
    %35 = handshake.constant %1 {value = 1 : index} : index loc(#loc37)
    %36 = arith.select %5, %35, %34 : index loc(#loc37)
    %37 = handshake.mux %36 [%falseResult_12, %trueResult] : index, none loc(#loc37)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc37)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %32 : none loc(#loc37)
    %39 = handshake.mux %36 [%falseResult_14, %trueResult] : index, none loc(#loc37)
    %40 = handshake.join %37, %39 : none, none loc(#loc29)
    %41 = handshake.constant %40 {value = true} : i1 loc(#loc29)
    handshake.return %41 : i1 loc(#loc29)
  } loc(#loc29)
  handshake.func @_Z10bitrev_dsaPKfPfj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg2: i32 loc(fused<#di_subprogram4>[#loc9]), %arg3: none loc(fused<#di_subprogram4>[#loc9]), ...) -> none attributes {argNames = ["input", "output", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc29)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc41)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc37)
    %5 = arith.shrui %arg2, %1 : i32 loc(#loc2)
    %6 = arith.cmpi eq, %5, %2 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc37)
    %8 = arith.index_cast %3 : i64 to index loc(#loc37)
    %9 = arith.index_cast %arg2 : i32 to index loc(#loc37)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc37)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc37)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc37)
    %11 = arith.index_cast %afterValue : index to i64 loc(#loc37)
    %12 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc48)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %10 : none loc(#loc48)
    %13 = arith.trunci %11 : i64 to i32 loc(#loc48)
    %14 = dataflow.invariant %afterCond, %5 : i1, i32 -> i32 loc(#loc48)
    %15 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc48)
    %16 = arith.index_cast %14 : i32 to index loc(#loc48)
    %17 = arith.index_cast %2 : i32 to index loc(#loc48)
    %index_2, %willContinue_3 = dataflow.stream %16, %15, %17 {step_op = ">>=", stop_cond = "!="} loc(#loc48)
    %18 = dataflow.carry %willContinue_3, %13, %23 : i1, i32, i32 -> i32 loc(#loc48)
    %afterValue_4, %afterCond_5 = dataflow.gate %18, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc48)
    handshake.sink %afterCond_5 : i1 loc(#loc48)
    %19 = dataflow.carry %willContinue_3, %2, %22 : i1, i32, i32 -> i32 loc(#loc48)
    %afterValue_6, %afterCond_7 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc48)
    handshake.sink %afterCond_7 : i1 loc(#loc48)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %19 : i32 loc(#loc48)
    %20 = arith.shli %afterValue_6, %1 : i32 loc(#loc55)
    %21 = arith.andi %afterValue_4, %1 : i32 loc(#loc55)
    %22 = arith.ori %21, %20 : i32 loc(#loc55)
    %23 = arith.shrui %afterValue_4, %1 : i32 loc(#loc56)
    %24 = arith.extui %falseResult_9 : i32 to i64 loc(#loc49)
    %25 = handshake.constant %10 {value = 0 : index} : index loc(#loc48)
    %26 = handshake.constant %10 {value = 1 : index} : index loc(#loc48)
    %27 = arith.select %12, %26, %25 : index loc(#loc48)
    %28 = handshake.mux %27 [%24, %3] : index, i64 loc(#loc48)
    %dataResult, %addressResults = handshake.load [%afterValue] %30#0, %32 : index, f32 loc(#loc49)
    %29 = arith.index_cast %28 : i64 to index loc(#loc49)
    %dataResult_10, %addressResult = handshake.store [%29] %dataResult, %37 : index, f32 loc(#loc49)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc29)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc29)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc37)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %30#1 : none loc(#loc37)
    %33 = handshake.constant %0 {value = 0 : index} : index loc(#loc37)
    %34 = handshake.constant %0 {value = 1 : index} : index loc(#loc37)
    %35 = arith.select %4, %34, %33 : index loc(#loc37)
    %36 = handshake.mux %35 [%falseResult_12, %trueResult] : index, none loc(#loc37)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc37)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %31 : none loc(#loc37)
    %38 = handshake.mux %35 [%falseResult_14, %trueResult] : index, none loc(#loc37)
    %39 = handshake.join %36, %38 : none, none loc(#loc29)
    handshake.return %39 : none loc(#loc30)
  } loc(#loc29)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc31)
    %false = arith.constant false loc(#loc31)
    %0 = seq.const_clock  low loc(#loc31)
    %c2_i32 = arith.constant 2 : i32 loc(#loc31)
    %1 = ub.poison : i64 loc(#loc31)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c128_i64 = arith.constant 128 : i64 loc(#loc2)
    %c128_i32 = arith.constant 128 : i32 loc(#loc2)
    %cst = arith.constant 9.99999997E-7 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<128xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc50)
      %11 = arith.uitofp %10 : i32 to f32 loc(#loc50)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc50)
      memref.store %11, %alloca[%12] : memref<128xf32> loc(#loc50)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc42)
      %14 = arith.cmpi ne, %13, %c128_i64 : i64 loc(#loc51)
      scf.condition(%14) %13 : i64 loc(#loc38)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc17])):
      scf.yield %arg0 : i64 loc(#loc38)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc38)
    %cast = memref.cast %alloca : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    %cast_2 = memref.cast %alloca_0 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    call @_Z10bitrev_cpuPKfPfj(%cast, %cast_2, %c128_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc32)
    %cast_3 = memref.cast %alloca_1 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c128_i32, %true : i32 loc(#loc33)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc33)
    %5 = handshake.esi_instance @_Z10bitrev_dsaPKfPfj_esi "_Z10bitrev_dsaPKfPfj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc33)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc33)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %11 = memref.load %alloca_0[%10] : memref<128xf32> loc(#loc57)
      %12 = memref.load %alloca_1[%10] : memref<128xf32> loc(#loc57)
      %13 = arith.subf %11, %12 : f32 loc(#loc57)
      %14 = math.absf %13 : f32 loc(#loc57)
      %15 = arith.cmpf ule, %14, %cst : f32 loc(#loc57)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc43)
        %19 = arith.cmpi eq, %18, %c128_i64 : i64 loc(#loc43)
        %20 = arith.extui %19 : i1 to i32 loc(#loc39)
        %21 = arith.cmpi ne, %18, %c128_i64 : i64 loc(#loc52)
        %22 = arith.extui %21 : i1 to i32 loc(#loc39)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc39)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc39)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc21]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc21]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc21])):
      scf.yield %arg0 : i64 loc(#loc39)
    } loc(#loc39)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc39)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc39)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc58)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc58)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc58)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc58)
      scf.yield %c1_i32 : i32 loc(#loc59)
    } loc(#loc39)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc34)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc34)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc34)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc34)
    } loc(#loc2)
    return %9 : i32 loc(#loc35)
  } loc(#loc31)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/bitrev/bitrev.cpp":0:0)
#loc2 = loc(unknown)
#loc5 = loc("tests/app/bitrev/bitrev.cpp":23:0)
#loc6 = loc("tests/app/bitrev/bitrev.cpp":24:0)
#loc7 = loc("tests/app/bitrev/bitrev.cpp":28:0)
#loc8 = loc("tests/app/bitrev/bitrev.cpp":30:0)
#loc10 = loc("tests/app/bitrev/bitrev.cpp":39:0)
#loc11 = loc("tests/app/bitrev/bitrev.cpp":44:0)
#loc12 = loc("tests/app/bitrev/bitrev.cpp":45:0)
#loc13 = loc("tests/app/bitrev/bitrev.cpp":46:0)
#loc14 = loc("tests/app/bitrev/bitrev.cpp":50:0)
#loc15 = loc("tests/app/bitrev/bitrev.cpp":52:0)
#loc16 = loc("tests/app/bitrev/main.cpp":6:0)
#loc18 = loc("tests/app/bitrev/main.cpp":16:0)
#loc19 = loc("tests/app/bitrev/main.cpp":20:0)
#loc20 = loc("tests/app/bitrev/main.cpp":21:0)
#loc22 = loc("tests/app/bitrev/main.cpp":24:0)
#loc23 = loc("tests/app/bitrev/main.cpp":25:0)
#loc24 = loc("tests/app/bitrev/main.cpp":26:0)
#loc25 = loc("tests/app/bitrev/main.cpp":30:0)
#loc26 = loc("tests/app/bitrev/main.cpp":32:0)
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 39>
#loc28 = loc(fused<#di_subprogram3>[#loc8])
#loc30 = loc(fused<#di_subprogram4>[#loc15])
#loc31 = loc(fused<#di_subprogram5>[#loc16])
#loc32 = loc(fused<#di_subprogram5>[#loc19])
#loc33 = loc(fused<#di_subprogram5>[#loc20])
#loc34 = loc(fused<#di_subprogram5>[#loc25])
#loc35 = loc(fused<#di_subprogram5>[#loc26])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 39>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 15>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 23>
#loc37 = loc(fused<#di_lexical_block9>[#loc10])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 39>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 15>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 23>
#loc40 = loc(fused<#di_lexical_block12>[#loc3])
#loc41 = loc(fused<#di_lexical_block13>[#loc10])
#loc42 = loc(fused<#di_lexical_block14>[#loc17])
#loc43 = loc(fused<#di_lexical_block15>[#loc21])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 22>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 44>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 24>
#loc45 = loc(fused<#di_lexical_block16>[#loc])
#loc46 = loc(fused<#di_lexical_block16>[#loc7])
#loc47 = loc(fused[#loc36, #loc40])
#loc48 = loc(fused<#di_lexical_block17>[#loc11])
#loc49 = loc(fused<#di_lexical_block17>[#loc14])
#loc50 = loc(fused<#di_lexical_block18>[#loc18])
#loc51 = loc(fused[#loc38, #loc42])
#loc52 = loc(fused[#loc39, #loc43])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 24>
#loc53 = loc(fused<#di_lexical_block20>[#loc5])
#loc54 = loc(fused<#di_lexical_block20>[#loc6])
#loc55 = loc(fused<#di_lexical_block21>[#loc12])
#loc56 = loc(fused<#di_lexical_block21>[#loc13])
#loc57 = loc(fused<#di_lexical_block22>[#loc22])
#loc58 = loc(fused<#di_lexical_block23>[#loc23])
#loc59 = loc(fused<#di_lexical_block23>[#loc24])
