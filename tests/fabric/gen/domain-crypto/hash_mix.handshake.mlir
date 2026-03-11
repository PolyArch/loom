#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/hash_mix/hash_mix.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/hash_mix/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/hash_mix/hash_mix.cpp":14:0)
#loc3 = loc("tests/app/hash_mix/hash_mix.cpp":18:0)
#loc13 = loc("tests/app/hash_mix/hash_mix.cpp":35:0)
#loc25 = loc("tests/app/hash_mix/main.cpp":17:0)
#loc30 = loc("tests/app/hash_mix/main.cpp":29:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 41>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 29>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 18>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 41>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 18>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 41>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 41, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 29, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 17, arg = 4, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "s", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "d", file = #di_file, line = 20, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 38, arg = 4, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "s", file = #di_file, line = 42, type = #di_derived_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "d", file = #di_file, line = 43, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_state", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_data", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 13, type = #di_composite_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 14, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_state", file = #di_file, line = 16, arg = 3, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_state", file = #di_file, line = 37, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable2, #di_local_variable3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 17>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 29>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_state", file = #di_file, line = 14, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 15, arg = 2, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_state", file = #di_file, line = 35, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 36, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "hash_mix_cpu", linkageName = "_Z12hash_mix_cpuPKjS0_Pjj", file = #di_file, line = 14, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable15, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable6>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "hash_mix_dsa", linkageName = "_Z12hash_mix_dsaPKjS0_Pjj", file = #di_file, line = 35, scopeLine = 38, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable16, #di_local_variable7, #di_local_variable1, #di_local_variable8, #di_local_variable9>
#loc41 = loc(fused<#di_lexical_block8>[#loc25])
#loc42 = loc(fused<#di_lexical_block9>[#loc30])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc43 = loc(fused<#di_subprogram4>[#loc1])
#loc45 = loc(fused<#di_subprogram5>[#loc13])
#loc49 = loc(fused<#di_lexical_block12>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<32xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 104, 97, 115, 104, 95, 109, 105, 120, 47, 104, 97, 115, 104, 95, 109, 105, 120, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<17xi8> = dense<[104, 97, 115, 104, 95, 109, 105, 120, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<17xi8> = dense<[104, 97, 115, 104, 95, 109, 105, 120, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z12hash_mix_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c25_i32 = arith.constant 25 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1540483477_i32 = arith.constant 1540483477 : i32 loc(#loc2)
    %c1026727936_i32 = arith.constant 1026727936 : i32 loc(#loc2)
    %c19_i32 = arith.constant 19 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc55)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc55)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc58)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc58)
        %5 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc59)
        %6 = arith.addi %5, %4 : i32 loc(#loc60)
        %7 = arith.shli %6, %c7_i32 : i32 loc(#loc61)
        %8 = arith.shrui %6, %c25_i32 : i32 loc(#loc61)
        %9 = arith.ori %7, %8 : i32 loc(#loc61)
        %10 = arith.xori %9, %5 : i32 loc(#loc62)
        %11 = arith.muli %10, %c1540483477_i32 : i32 loc(#loc63)
        %12 = arith.muli %10, %c1026727936_i32 : i32 loc(#loc64)
        %13 = arith.shrui %11, %c19_i32 : i32 loc(#loc64)
        %14 = arith.ori %13, %12 : i32 loc(#loc64)
        memref.store %14, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc65)
        %15 = arith.addi %arg4, %c1_i64 : i64 loc(#loc55)
        %16 = arith.cmpi ne, %15, %1 : i64 loc(#loc66)
        scf.condition(%16) %15 : i64 loc(#loc49)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block12>[#loc3])):
        scf.yield %arg4 : i64 loc(#loc49)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc49)
    } loc(#loc49)
    return loc(#loc44)
  } loc(#loc43)
  handshake.func @_Z12hash_mix_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg3: i32 loc(fused<#di_subprogram5>[#loc13]), %arg4: i1 loc(fused<#di_subprogram5>[#loc13]), ...) -> i1 attributes {argNames = ["input_state", "input_data", "output_state", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc45)
    %1 = handshake.join %0 : none loc(#loc45)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 25 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 7 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %1 {value = 1540483477 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %1 {value = 1026727936 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %1 {value = 19 : i32} : i32 loc(#loc2)
    %9 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %9, %1 : none loc(#loc50)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc50)
    %11 = arith.index_cast %5 : i64 to index loc(#loc50)
    %12 = arith.index_cast %arg3 : i32 to index loc(#loc50)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc50)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc50)
    %dataResult, %addressResults = handshake.load [%afterValue] %22#0, %25 : index, i32 loc(#loc67)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %23#0, %32 : index, i32 loc(#loc68)
    %13 = arith.addi %dataResult_0, %dataResult : i32 loc(#loc69)
    %14 = arith.shli %13, %4 : i32 loc(#loc70)
    %15 = arith.shrui %13, %3 : i32 loc(#loc70)
    %16 = arith.ori %14, %15 : i32 loc(#loc70)
    %17 = arith.xori %16, %dataResult_0 : i32 loc(#loc71)
    %18 = arith.muli %17, %6 : i32 loc(#loc72)
    %19 = arith.muli %17, %7 : i32 loc(#loc73)
    %20 = arith.shrui %18, %8 : i32 loc(#loc73)
    %21 = arith.ori %20, %19 : i32 loc(#loc73)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %21, %30 : index, i32 loc(#loc74)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc45)
    %24 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc45)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc50)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %22#1 : none loc(#loc50)
    %26 = handshake.constant %1 {value = 0 : index} : index loc(#loc50)
    %27 = handshake.constant %1 {value = 1 : index} : index loc(#loc50)
    %28 = arith.select %9, %27, %26 : index loc(#loc50)
    %29 = handshake.mux %28 [%falseResult_4, %trueResult] : index, none loc(#loc50)
    %30 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc50)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %24 : none loc(#loc50)
    %31 = handshake.mux %28 [%falseResult_6, %trueResult] : index, none loc(#loc50)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc50)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %23#1 : none loc(#loc50)
    %33 = handshake.mux %28 [%falseResult_8, %trueResult] : index, none loc(#loc50)
    %34 = handshake.join %29, %31, %33 : none, none, none loc(#loc45)
    %35 = handshake.constant %34 {value = true} : i1 loc(#loc45)
    handshake.return %35 : i1 loc(#loc45)
  } loc(#loc45)
  handshake.func @_Z12hash_mix_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc13]), %arg3: i32 loc(fused<#di_subprogram5>[#loc13]), %arg4: none loc(fused<#di_subprogram5>[#loc13]), ...) -> none attributes {argNames = ["input_state", "input_data", "output_state", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc45)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 25 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 7 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %0 {value = 1540483477 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %0 {value = 1026727936 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %0 {value = 19 : i32} : i32 loc(#loc2)
    %8 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %8, %0 : none loc(#loc50)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc50)
    %10 = arith.index_cast %4 : i64 to index loc(#loc50)
    %11 = arith.index_cast %arg3 : i32 to index loc(#loc50)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc50)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc50)
    %dataResult, %addressResults = handshake.load [%afterValue] %21#0, %24 : index, i32 loc(#loc67)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %22#0, %31 : index, i32 loc(#loc68)
    %12 = arith.addi %dataResult_0, %dataResult : i32 loc(#loc69)
    %13 = arith.shli %12, %3 : i32 loc(#loc70)
    %14 = arith.shrui %12, %2 : i32 loc(#loc70)
    %15 = arith.ori %13, %14 : i32 loc(#loc70)
    %16 = arith.xori %15, %dataResult_0 : i32 loc(#loc71)
    %17 = arith.muli %16, %5 : i32 loc(#loc72)
    %18 = arith.muli %16, %6 : i32 loc(#loc73)
    %19 = arith.shrui %17, %7 : i32 loc(#loc73)
    %20 = arith.ori %19, %18 : i32 loc(#loc73)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %20, %29 : index, i32 loc(#loc74)
    %21:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc45)
    %23 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc45)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc50)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %21#1 : none loc(#loc50)
    %25 = handshake.constant %0 {value = 0 : index} : index loc(#loc50)
    %26 = handshake.constant %0 {value = 1 : index} : index loc(#loc50)
    %27 = arith.select %8, %26, %25 : index loc(#loc50)
    %28 = handshake.mux %27 [%falseResult_4, %trueResult] : index, none loc(#loc50)
    %29 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc50)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %23 : none loc(#loc50)
    %30 = handshake.mux %27 [%falseResult_6, %trueResult] : index, none loc(#loc50)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc50)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %22#1 : none loc(#loc50)
    %32 = handshake.mux %27 [%falseResult_8, %trueResult] : index, none loc(#loc50)
    %33 = handshake.join %28, %30, %32 : none, none, none loc(#loc45)
    handshake.return %33 : none loc(#loc46)
  } loc(#loc45)
  func.func private @llvm.fshl.i32(i32, i32, i32) -> i32 loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc36)
    %false = arith.constant false loc(#loc36)
    %0 = seq.const_clock  low loc(#loc36)
    %c2_i32 = arith.constant 2 : i32 loc(#loc36)
    %1 = ub.poison : i64 loc(#loc36)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1732584193_i32 = arith.constant 1732584193 : i32 loc(#loc2)
    %c13_i32 = arith.constant 13 : i32 loc(#loc2)
    %c-271733879_i32 = arith.constant -271733879 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<17xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<17xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc51)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc51)
      %12 = arith.addi %11, %c1732584193_i32 : i32 loc(#loc51)
      memref.store %12, %alloca[%10] : memref<1024xi32> loc(#loc51)
      %13 = arith.muli %11, %c13_i32 : i32 loc(#loc52)
      %14 = arith.addi %13, %c-271733879_i32 : i32 loc(#loc52)
      memref.store %14, %alloca_0[%10] : memref<1024xi32> loc(#loc52)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc47)
      %16 = arith.cmpi ne, %15, %c1024_i64 : i64 loc(#loc53)
      scf.condition(%16) %15 : i64 loc(#loc41)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc25])):
      scf.yield %arg0 : i64 loc(#loc41)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc41)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc37)
    call @_Z12hash_mix_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc37)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc38)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc38)
    %5 = handshake.esi_instance @_Z12hash_mix_dsaPKjS0_Pjj_esi "_Z12hash_mix_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc38)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc38)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %11 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc57)
      %12 = memref.load %alloca_2[%10] : memref<1024xi32> loc(#loc57)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc57)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc48)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc48)
        %18 = arith.extui %17 : i1 to i32 loc(#loc42)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc54)
        %20 = arith.extui %19 : i1 to i32 loc(#loc42)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc42)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc30]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc30]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc30])):
      scf.yield %arg0 : i64 loc(#loc42)
    } loc(#loc42)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc42)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc42)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<17xi8> -> index loc(#loc75)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc75)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc75)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc75)
      scf.yield %c1_i32 : i32 loc(#loc76)
    } loc(#loc42)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<17xi8> -> index loc(#loc39)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc39)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc39)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc39)
    } loc(#loc2)
    return %9 : i32 loc(#loc40)
  } loc(#loc36)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/hash_mix/hash_mix.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/hash_mix/hash_mix.cpp":19:0)
#loc5 = loc("tests/app/hash_mix/hash_mix.cpp":20:0)
#loc6 = loc("tests/app/hash_mix/hash_mix.cpp":23:0)
#loc7 = loc("tests/app/hash_mix/hash_mix.cpp":24:0)
#loc8 = loc("tests/app/hash_mix/hash_mix.cpp":25:0)
#loc9 = loc("tests/app/hash_mix/hash_mix.cpp":26:0)
#loc10 = loc("tests/app/hash_mix/hash_mix.cpp":27:0)
#loc11 = loc("tests/app/hash_mix/hash_mix.cpp":29:0)
#loc12 = loc("tests/app/hash_mix/hash_mix.cpp":31:0)
#loc14 = loc("tests/app/hash_mix/hash_mix.cpp":41:0)
#loc15 = loc("tests/app/hash_mix/hash_mix.cpp":42:0)
#loc16 = loc("tests/app/hash_mix/hash_mix.cpp":43:0)
#loc17 = loc("tests/app/hash_mix/hash_mix.cpp":46:0)
#loc18 = loc("tests/app/hash_mix/hash_mix.cpp":47:0)
#loc19 = loc("tests/app/hash_mix/hash_mix.cpp":48:0)
#loc20 = loc("tests/app/hash_mix/hash_mix.cpp":49:0)
#loc21 = loc("tests/app/hash_mix/hash_mix.cpp":50:0)
#loc22 = loc("tests/app/hash_mix/hash_mix.cpp":52:0)
#loc23 = loc("tests/app/hash_mix/hash_mix.cpp":54:0)
#loc24 = loc("tests/app/hash_mix/main.cpp":5:0)
#loc26 = loc("tests/app/hash_mix/main.cpp":18:0)
#loc27 = loc("tests/app/hash_mix/main.cpp":19:0)
#loc28 = loc("tests/app/hash_mix/main.cpp":23:0)
#loc29 = loc("tests/app/hash_mix/main.cpp":26:0)
#loc31 = loc("tests/app/hash_mix/main.cpp":30:0)
#loc32 = loc("tests/app/hash_mix/main.cpp":31:0)
#loc33 = loc("tests/app/hash_mix/main.cpp":32:0)
#loc34 = loc("tests/app/hash_mix/main.cpp":36:0)
#loc35 = loc("tests/app/hash_mix/main.cpp":38:0)
#loc36 = loc(fused<#di_subprogram3>[#loc24])
#loc37 = loc(fused<#di_subprogram3>[#loc28])
#loc38 = loc(fused<#di_subprogram3>[#loc29])
#loc39 = loc(fused<#di_subprogram3>[#loc34])
#loc40 = loc(fused<#di_subprogram3>[#loc35])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 17>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 29>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 41>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 17>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 29>
#loc44 = loc(fused<#di_subprogram4>[#loc12])
#loc46 = loc(fused<#di_subprogram5>[#loc23])
#loc47 = loc(fused<#di_lexical_block10>[#loc25])
#loc48 = loc(fused<#di_lexical_block11>[#loc30])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 18>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 41>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 30>
#loc50 = loc(fused<#di_lexical_block13>[#loc14])
#loc51 = loc(fused<#di_lexical_block14>[#loc26])
#loc52 = loc(fused<#di_lexical_block14>[#loc27])
#loc53 = loc(fused[#loc41, #loc47])
#loc54 = loc(fused[#loc42, #loc48])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 18>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 41>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 30>
#loc55 = loc(fused<#di_lexical_block16>[#loc3])
#loc56 = loc(fused<#di_lexical_block17>[#loc14])
#loc57 = loc(fused<#di_lexical_block18>[#loc31])
#loc58 = loc(fused<#di_lexical_block19>[#loc4])
#loc59 = loc(fused<#di_lexical_block19>[#loc5])
#loc60 = loc(fused<#di_lexical_block19>[#loc6])
#loc61 = loc(fused<#di_lexical_block19>[#loc7])
#loc62 = loc(fused<#di_lexical_block19>[#loc8])
#loc63 = loc(fused<#di_lexical_block19>[#loc9])
#loc64 = loc(fused<#di_lexical_block19>[#loc10])
#loc65 = loc(fused<#di_lexical_block19>[#loc11])
#loc66 = loc(fused[#loc49, #loc55])
#loc67 = loc(fused<#di_lexical_block20>[#loc15])
#loc68 = loc(fused<#di_lexical_block20>[#loc16])
#loc69 = loc(fused<#di_lexical_block20>[#loc17])
#loc70 = loc(fused<#di_lexical_block20>[#loc18])
#loc71 = loc(fused<#di_lexical_block20>[#loc19])
#loc72 = loc(fused<#di_lexical_block20>[#loc20])
#loc73 = loc(fused<#di_lexical_block20>[#loc21])
#loc74 = loc(fused<#di_lexical_block20>[#loc22])
#loc75 = loc(fused<#di_lexical_block21>[#loc32])
#loc76 = loc(fused<#di_lexical_block21>[#loc33])
