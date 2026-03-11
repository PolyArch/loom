#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/find_first_set/find_first_set.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/find_first_set/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/find_first_set/find_first_set.cpp":15:0)
#loc3 = loc("tests/app/find_first_set/find_first_set.cpp":18:0)
#loc6 = loc("tests/app/find_first_set/find_first_set.cpp":26:0)
#loc10 = loc("tests/app/find_first_set/find_first_set.cpp":38:0)
#loc26 = loc("tests/app/find_first_set/main.cpp":18:0)
#loc30 = loc("tests/app/find_first_set/main.cpp":33:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 43>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 18>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 33>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 18>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 43>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 18>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 43>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 43, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 18, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 33, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 21>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 46>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 17, arg = 3, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 40, arg = 3, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file, line = 44, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_position", file = #di_file1, line = 23, type = #di_composite_type>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_position", file = #di_file1, line = 24, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 23>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 48>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_position", file = #di_file, line = 16, arg = 2, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_position", file = #di_file, line = 39, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable8, #di_local_variable9, #di_local_variable2, #di_local_variable10, #di_local_variable11, #di_local_variable3>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 18>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 33>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 15, arg = 1, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "position", file = #di_file, line = 24, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file, line = 38, arg = 1, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "position", file = #di_file, line = 49, type = #di_derived_type1>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "find_first_set_cpu", linkageName = "_Z18find_first_set_cpuPKjPjj", file = #di_file, line = 15, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable14, #di_local_variable12, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable15>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "find_first_set_dsa", linkageName = "_Z18find_first_set_dsaPKjPjj", file = #di_file, line = 38, scopeLine = 40, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable13, #di_local_variable6, #di_local_variable1, #di_local_variable7, #di_local_variable17>
#loc48 = loc(fused<#di_lexical_block12>[#loc26])
#loc49 = loc(fused<#di_lexical_block13>[#loc30])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 18>
#loc50 = loc(fused<#di_subprogram4>[#loc1])
#loc52 = loc(fused<#di_subprogram5>[#loc10])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 18>
#loc56 = loc(fused<#di_lexical_block16>[#loc3])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 18>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 21>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 23>
#loc73 = loc(fused<#di_lexical_block28>[#loc6])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<44xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 102, 105, 110, 100, 95, 102, 105, 114, 115, 116, 95, 115, 101, 116, 47, 102, 105, 110, 100, 95, 102, 105, 114, 115, 116, 95, 115, 101, 116, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<23xi8> = dense<[102, 105, 110, 100, 95, 102, 105, 114, 115, 116, 95, 115, 101, 116, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<23xi8> = dense<[102, 105, 110, 100, 95, 102, 105, 114, 115, 116, 95, 115, 101, 116, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z18find_first_set_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc61)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc61)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc64)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc64)
        %5 = arith.cmpi eq, %4, %c0_i32 : i32 loc(#loc69)
        %6 = scf.if %5 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc69)
        } else {
          %9 = arith.andi %4, %c1_i32 : i32 loc(#loc73)
          %10 = arith.cmpi eq, %9, %c0_i32 : i32 loc(#loc73)
          %11 = scf.if %10 -> (i32) {
            %12:2 = scf.while (%arg4 = %c1_i32, %arg5 = %4) : (i32, i32) -> (i32, i32) {
              %13 = arith.addi %arg4, %c1_i32 : i32 loc(#loc75)
              %14 = arith.shrui %arg5, %c1_i32 : i32 loc(#loc76)
              %15 = arith.andi %arg5, %c2_i32 : i32 loc(#loc73)
              %16 = arith.cmpi eq, %15, %c0_i32 : i32 loc(#loc73)
              scf.condition(%16) %13, %14 : i32, i32 loc(#loc73)
            } do {
            ^bb0(%arg4: i32 loc(fused<#di_lexical_block28>[#loc6]), %arg5: i32 loc(fused<#di_lexical_block28>[#loc6])):
              scf.yield %arg4, %arg5 : i32, i32 loc(#loc73)
            } loc(#loc73)
            scf.yield %12#0 : i32 loc(#loc73)
          } else {
            scf.yield %c1_i32 : i32 loc(#loc73)
          } loc(#loc73)
          scf.yield %11 : i32 loc(#loc69)
        } loc(#loc69)
        memref.store %6, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc70)
        %7 = arith.addi %arg3, %c1_i64 : i64 loc(#loc61)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc65)
        scf.condition(%8) %7 : i64 loc(#loc56)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block16>[#loc3])):
        scf.yield %arg3 : i64 loc(#loc56)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc56)
    } loc(#loc56)
    return loc(#loc51)
  } loc(#loc50)
  handshake.func @_Z18find_first_set_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: i1 loc(fused<#di_subprogram5>[#loc10]), ...) -> i1 attributes {argNames = ["input_data", "output_position", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc52)
    %1 = handshake.join %0 : none loc(#loc52)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 2 : i32} : i32 loc(#loc2)
    %6 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc62)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc57)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc57)
    %8 = arith.index_cast %4 : i64 to index loc(#loc57)
    %9 = arith.index_cast %arg2 : i32 to index loc(#loc57)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc57)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc57)
    %dataResult, %addressResults = handshake.load [%afterValue] %29#0, %31 : index, i32 loc(#loc66)
    %11 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc71)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %10 : none loc(#loc71)
    %12 = arith.andi %dataResult, %2 : i32 loc(#loc74)
    %13 = arith.cmpi eq, %12, %3 : i32 loc(#loc74)
    %14 = handshake.constant %1 {value = true} : i1 loc(#loc74)
    %15 = dataflow.carry %14, %2, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc74)
    %16 = dataflow.carry %14, %dataResult, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc74)
    %17 = arith.addi %15, %2 : i32 loc(#loc77)
    %18 = arith.shrui %16, %2 : i32 loc(#loc78)
    %19 = arith.andi %16, %5 : i32 loc(#loc74)
    %20 = arith.cmpi eq, %19, %3 : i32 loc(#loc74)
    %trueResult_2, %falseResult_3 = handshake.cond_br %20, %17 : i32 loc(#loc74)
    %trueResult_4, %falseResult_5 = handshake.cond_br %20, %18 : i32 loc(#loc74)
    handshake.sink %falseResult_5 : i32 loc(#loc74)
    %21 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc74)
    %22 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc74)
    %23 = arith.select %13, %22, %21 : index loc(#loc74)
    %24 = handshake.mux %23 [%2, %falseResult_3] : index, i32 loc(#loc74)
    %25 = handshake.constant %10 {value = 0 : index} : index loc(#loc71)
    %26 = handshake.constant %10 {value = 1 : index} : index loc(#loc71)
    %27 = arith.select %11, %26, %25 : index loc(#loc71)
    %28 = handshake.mux %27 [%24, %3] : index, i32 loc(#loc71)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %28, %36 : index, i32 loc(#loc72)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc52)
    %30 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc52)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc57)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %29#1 : none loc(#loc57)
    %32 = handshake.constant %1 {value = 0 : index} : index loc(#loc57)
    %33 = handshake.constant %1 {value = 1 : index} : index loc(#loc57)
    %34 = arith.select %6, %33, %32 : index loc(#loc57)
    %35 = handshake.mux %34 [%falseResult_8, %trueResult] : index, none loc(#loc57)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc57)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %30 : none loc(#loc57)
    %37 = handshake.mux %34 [%falseResult_10, %trueResult] : index, none loc(#loc57)
    %38 = handshake.join %35, %37 : none, none loc(#loc52)
    %39 = handshake.constant %38 {value = true} : i1 loc(#loc52)
    handshake.return %39 : i1 loc(#loc52)
  } loc(#loc52)
  handshake.func @_Z18find_first_set_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc10]), %arg2: i32 loc(fused<#di_subprogram5>[#loc10]), %arg3: none loc(fused<#di_subprogram5>[#loc10]), ...) -> none attributes {argNames = ["input_data", "output_position", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc52)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 2 : i32} : i32 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc62)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc57)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc57)
    %7 = arith.index_cast %3 : i64 to index loc(#loc57)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc57)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc57)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc57)
    %dataResult, %addressResults = handshake.load [%afterValue] %28#0, %30 : index, i32 loc(#loc66)
    %10 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc71)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %9 : none loc(#loc71)
    %11 = arith.andi %dataResult, %1 : i32 loc(#loc74)
    %12 = arith.cmpi eq, %11, %2 : i32 loc(#loc74)
    %13 = handshake.constant %0 {value = true} : i1 loc(#loc74)
    %14 = dataflow.carry %13, %1, %trueResult_2 : i1, i32, i32 -> i32 loc(#loc74)
    %15 = dataflow.carry %13, %dataResult, %trueResult_4 : i1, i32, i32 -> i32 loc(#loc74)
    %16 = arith.addi %14, %1 : i32 loc(#loc77)
    %17 = arith.shrui %15, %1 : i32 loc(#loc78)
    %18 = arith.andi %15, %4 : i32 loc(#loc74)
    %19 = arith.cmpi eq, %18, %2 : i32 loc(#loc74)
    %trueResult_2, %falseResult_3 = handshake.cond_br %19, %16 : i32 loc(#loc74)
    %trueResult_4, %falseResult_5 = handshake.cond_br %19, %17 : i32 loc(#loc74)
    handshake.sink %falseResult_5 : i32 loc(#loc74)
    %20 = handshake.constant %falseResult_1 {value = 0 : index} : index loc(#loc74)
    %21 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc74)
    %22 = arith.select %12, %21, %20 : index loc(#loc74)
    %23 = handshake.mux %22 [%1, %falseResult_3] : index, i32 loc(#loc74)
    %24 = handshake.constant %9 {value = 0 : index} : index loc(#loc71)
    %25 = handshake.constant %9 {value = 1 : index} : index loc(#loc71)
    %26 = arith.select %10, %25, %24 : index loc(#loc71)
    %27 = handshake.mux %26 [%23, %2] : index, i32 loc(#loc71)
    %dataResult_6, %addressResult = handshake.store [%afterValue] %27, %35 : index, i32 loc(#loc72)
    %28:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc52)
    %29 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc52)
    %30 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc57)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %28#1 : none loc(#loc57)
    %31 = handshake.constant %0 {value = 0 : index} : index loc(#loc57)
    %32 = handshake.constant %0 {value = 1 : index} : index loc(#loc57)
    %33 = arith.select %5, %32, %31 : index loc(#loc57)
    %34 = handshake.mux %33 [%falseResult_8, %trueResult] : index, none loc(#loc57)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc57)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %29 : none loc(#loc57)
    %36 = handshake.mux %33 [%falseResult_10, %trueResult] : index, none loc(#loc57)
    %37 = handshake.join %34, %36 : none, none loc(#loc52)
    handshake.return %37 : none loc(#loc53)
  } loc(#loc52)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc36)
    %false = arith.constant false loc(#loc36)
    %0 = seq.const_clock  low loc(#loc36)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %1 = ub.poison : i64 loc(#loc36)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c6 = arith.constant 6 : index loc(#loc37)
    %c5 = arith.constant 5 : index loc(#loc38)
    %c4 = arith.constant 4 : index loc(#loc39)
    %c3 = arith.constant 3 : index loc(#loc40)
    %c2 = arith.constant 2 : index loc(#loc41)
    %c1 = arith.constant 1 : index loc(#loc42)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c4_i32 = arith.constant 4 : i32 loc(#loc2)
    %c-2147483648_i32 = arith.constant -2147483648 : i32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c-16_i32 = arith.constant -16 : i32 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c34661_i32 = arith.constant 34661 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<23xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<23xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    memref.store %c0_i32, %alloca[%c0] : memref<256xi32> loc(#loc43)
    memref.store %c1_i32, %alloca[%c1] : memref<256xi32> loc(#loc42)
    memref.store %c2_i32, %alloca[%c2] : memref<256xi32> loc(#loc41)
    memref.store %c4_i32, %alloca[%c3] : memref<256xi32> loc(#loc40)
    memref.store %c-2147483648_i32, %alloca[%c4] : memref<256xi32> loc(#loc39)
    memref.store %c-1_i32, %alloca[%c5] : memref<256xi32> loc(#loc38)
    memref.store %c-16_i32, %alloca[%c6] : memref<256xi32> loc(#loc37)
    %4 = scf.while (%arg0 = %c7_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc58)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc58)
      %12 = arith.muli %11, %c34661_i32 : i32 loc(#loc58)
      memref.store %12, %alloca[%10] : memref<256xi32> loc(#loc58)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc54)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc59)
      scf.condition(%14) %13 : i64 loc(#loc48)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc26])):
      scf.yield %arg0 : i64 loc(#loc48)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc48)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc44)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc44)
    call @_Z18find_first_set_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc44)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc45)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc45)
    %5 = handshake.esi_instance @_Z18find_first_set_dsaPKjPjj_esi "_Z18find_first_set_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc45)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc45)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc63)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc63)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc63)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc63)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc55)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc55)
        %18 = arith.extui %17 : i1 to i32 loc(#loc49)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc60)
        %20 = arith.extui %19 : i1 to i32 loc(#loc49)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc63)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc63)
      } loc(#loc63)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc49)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc49)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc30]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc30]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc30])):
      scf.yield %arg0 : i64 loc(#loc49)
    } loc(#loc49)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc49)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc49)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<23xi8> -> index loc(#loc67)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc67)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc67)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc67)
      scf.yield %c1_i32 : i32 loc(#loc68)
    } loc(#loc49)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<23xi8> -> index loc(#loc46)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc46)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc46)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc46)
    } loc(#loc2)
    return %9 : i32 loc(#loc47)
  } loc(#loc36)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/find_first_set/find_first_set.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/find_first_set/find_first_set.cpp":19:0)
#loc5 = loc("tests/app/find_first_set/find_first_set.cpp":21:0)
#loc7 = loc("tests/app/find_first_set/find_first_set.cpp":27:0)
#loc8 = loc("tests/app/find_first_set/find_first_set.cpp":28:0)
#loc9 = loc("tests/app/find_first_set/find_first_set.cpp":34:0)
#loc11 = loc("tests/app/find_first_set/find_first_set.cpp":43:0)
#loc12 = loc("tests/app/find_first_set/find_first_set.cpp":44:0)
#loc13 = loc("tests/app/find_first_set/find_first_set.cpp":46:0)
#loc14 = loc("tests/app/find_first_set/find_first_set.cpp":51:0)
#loc15 = loc("tests/app/find_first_set/find_first_set.cpp":52:0)
#loc16 = loc("tests/app/find_first_set/find_first_set.cpp":53:0)
#loc17 = loc("tests/app/find_first_set/find_first_set.cpp":59:0)
#loc18 = loc("tests/app/find_first_set/main.cpp":5:0)
#loc19 = loc("tests/app/find_first_set/main.cpp":16:0)
#loc20 = loc("tests/app/find_first_set/main.cpp":15:0)
#loc21 = loc("tests/app/find_first_set/main.cpp":14:0)
#loc22 = loc("tests/app/find_first_set/main.cpp":13:0)
#loc23 = loc("tests/app/find_first_set/main.cpp":12:0)
#loc24 = loc("tests/app/find_first_set/main.cpp":11:0)
#loc25 = loc("tests/app/find_first_set/main.cpp":10:0)
#loc27 = loc("tests/app/find_first_set/main.cpp":19:0)
#loc28 = loc("tests/app/find_first_set/main.cpp":27:0)
#loc29 = loc("tests/app/find_first_set/main.cpp":30:0)
#loc31 = loc("tests/app/find_first_set/main.cpp":34:0)
#loc32 = loc("tests/app/find_first_set/main.cpp":35:0)
#loc33 = loc("tests/app/find_first_set/main.cpp":36:0)
#loc34 = loc("tests/app/find_first_set/main.cpp":40:0)
#loc35 = loc("tests/app/find_first_set/main.cpp":42:0)
#loc36 = loc(fused<#di_subprogram3>[#loc18])
#loc37 = loc(fused<#di_subprogram3>[#loc19])
#loc38 = loc(fused<#di_subprogram3>[#loc20])
#loc39 = loc(fused<#di_subprogram3>[#loc21])
#loc40 = loc(fused<#di_subprogram3>[#loc22])
#loc41 = loc(fused<#di_subprogram3>[#loc23])
#loc42 = loc(fused<#di_subprogram3>[#loc24])
#loc43 = loc(fused<#di_subprogram3>[#loc25])
#loc44 = loc(fused<#di_subprogram3>[#loc28])
#loc45 = loc(fused<#di_subprogram3>[#loc29])
#loc46 = loc(fused<#di_subprogram3>[#loc34])
#loc47 = loc(fused<#di_subprogram3>[#loc35])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 18>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 33>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 43>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 18>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 33>
#loc51 = loc(fused<#di_subprogram4>[#loc9])
#loc53 = loc(fused<#di_subprogram5>[#loc17])
#loc54 = loc(fused<#di_lexical_block14>[#loc26])
#loc55 = loc(fused<#di_lexical_block15>[#loc30])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 43>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 34>
#loc57 = loc(fused<#di_lexical_block17>[#loc11])
#loc58 = loc(fused<#di_lexical_block18>[#loc27])
#loc59 = loc(fused[#loc48, #loc54])
#loc60 = loc(fused[#loc49, #loc55])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 43>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 34>
#loc61 = loc(fused<#di_lexical_block20>[#loc3])
#loc62 = loc(fused<#di_lexical_block21>[#loc11])
#loc63 = loc(fused<#di_lexical_block22>[#loc31])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 46>
#loc64 = loc(fused<#di_lexical_block23>[#loc4])
#loc65 = loc(fused[#loc56, #loc61])
#loc66 = loc(fused<#di_lexical_block24>[#loc12])
#loc67 = loc(fused<#di_lexical_block25>[#loc32])
#loc68 = loc(fused<#di_lexical_block25>[#loc33])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 48>
#loc69 = loc(fused<#di_lexical_block26>[#loc5])
#loc70 = loc(fused<#di_lexical_block26>[#loc])
#loc71 = loc(fused<#di_lexical_block27>[#loc13])
#loc72 = loc(fused<#di_lexical_block27>[#loc])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 26>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 51>
#loc74 = loc(fused<#di_lexical_block29>[#loc14])
#loc75 = loc(fused<#di_lexical_block30>[#loc7])
#loc76 = loc(fused<#di_lexical_block30>[#loc8])
#loc77 = loc(fused<#di_lexical_block31>[#loc15])
#loc78 = loc(fused<#di_lexical_block31>[#loc16])
