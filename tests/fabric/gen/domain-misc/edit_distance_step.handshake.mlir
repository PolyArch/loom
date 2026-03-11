#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/edit_distance_step/edit_distance_step.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/edit_distance_step/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":15:0)
#loc3 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":22:0)
#loc12 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":39:0)
#loc23 = loc("tests/app/edit_distance_step/main.cpp":15:0)
#loc31 = loc("tests/app/edit_distance_step/main.cpp":34:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 22>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 48>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 15>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 34>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 22>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 48>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 2048, elements = #llvm.di_subrange<count = 64 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 22>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 48>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 48, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 15, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 34, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 21, arg = 7, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "cost", file = #di_file, line = 23, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "insert_cost", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "delete_cost", file = #di_file, line = 27, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "subst_cost", file = #di_file, line = 28, type = #di_derived_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "min_val", file = #di_file, line = 30, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 45, arg = 7, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "cost", file = #di_file, line = 49, type = #di_derived_type1>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "insert_cost", file = #di_file, line = 52, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "delete_cost", file = #di_file, line = 53, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "subst_cost", file = #di_file, line = 54, type = #di_derived_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "min_val", file = #di_file, line = 56, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "left", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "top", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "diag", file = #di_file1, line = 11, type = #di_composite_type>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "char_a", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "char_b", file = #di_file1, line = 13, type = #di_composite_type>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_result", file = #di_file1, line = 24, type = #di_composite_type>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_result", file = #di_file1, line = 25, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_result", file = #di_file, line = 20, arg = 6, type = #di_derived_type5>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_result", file = #di_file, line = 44, arg = 6, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable16, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable2, #di_local_variable22, #di_local_variable23, #di_local_variable3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 15>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 34>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_left", file = #di_file, line = 15, arg = 1, type = #di_derived_type6>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_top", file = #di_file, line = 16, arg = 2, type = #di_derived_type6>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_diag", file = #di_file, line = 17, arg = 3, type = #di_derived_type6>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_char_a", file = #di_file, line = 18, arg = 4, type = #di_derived_type6>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_char_b", file = #di_file, line = 19, arg = 5, type = #di_derived_type6>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_left", file = #di_file, line = 39, arg = 1, type = #di_derived_type6>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_top", file = #di_file, line = 40, arg = 2, type = #di_derived_type6>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_diag", file = #di_file, line = 41, arg = 3, type = #di_derived_type6>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_char_a", file = #di_file, line = 42, arg = 4, type = #di_derived_type6>
#di_local_variable35 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_char_b", file = #di_file, line = 43, arg = 5, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "edit_distance_step_cpu", linkageName = "_Z22edit_distance_step_cpuPKjS0_S0_S0_S0_Pjj", file = #di_file, line = 15, scopeLine = 21, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable26, #di_local_variable27, #di_local_variable28, #di_local_variable29, #di_local_variable30, #di_local_variable24, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "edit_distance_step_dsa", linkageName = "_Z22edit_distance_step_dsaPKjS0_S0_S0_S0_Pjj", file = #di_file, line = 39, scopeLine = 45, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable31, #di_local_variable32, #di_local_variable33, #di_local_variable34, #di_local_variable35, #di_local_variable25, #di_local_variable10, #di_local_variable1, #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15>
#loc42 = loc(fused<#di_lexical_block8>[#loc23])
#loc43 = loc(fused<#di_lexical_block9>[#loc31])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 22>
#loc44 = loc(fused<#di_subprogram4>[#loc1])
#loc46 = loc(fused<#di_subprogram5>[#loc12])
#loc50 = loc(fused<#di_lexical_block12>[#loc3])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<52xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 101, 100, 105, 116, 95, 100, 105, 115, 116, 97, 110, 99, 101, 95, 115, 116, 101, 112, 47, 101, 100, 105, 116, 95, 100, 105, 115, 116, 97, 110, 99, 101, 95, 115, 116, 101, 112, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<27xi8> = dense<[101, 100, 105, 116, 95, 100, 105, 115, 116, 97, 110, 99, 101, 95, 115, 116, 101, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<27xi8> = dense<[101, 100, 105, 116, 95, 100, 105, 115, 116, 97, 110, 99, 101, 95, 115, 116, 101, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z22edit_distance_step_cpuPKjS0_S0_S0_S0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg4: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg5: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg6: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg6, %c0_i32 : i32 loc(#loc59)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg6 : i32 to i64 loc(#loc59)
      %2 = scf.while (%arg7 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg7 : i64 to index loc(#loc62)
        %4 = memref.load %arg3[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
        %5 = memref.load %arg4[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
        %6 = arith.cmpi ne, %4, %5 : i32 loc(#loc62)
        %7 = arith.extui %6 : i1 to i32 loc(#loc62)
        %8 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc63)
        %9 = arith.addi %8, %c1_i32 : i32 loc(#loc63)
        %10 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc64)
        %11 = arith.addi %10, %c1_i32 : i32 loc(#loc64)
        %12 = memref.load %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc65)
        %13 = arith.addi %12, %7 : i32 loc(#loc65)
        %14 = arith.cmpi ult, %11, %9 : i32 loc(#loc66)
        %15 = arith.select %14, %11, %9 : i32 loc(#loc66)
        %16 = arith.cmpi ult, %13, %15 : i32 loc(#loc67)
        %17 = arith.select %16, %13, %15 : i32 loc(#loc67)
        memref.store %17, %arg5[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc68)
        %18 = arith.addi %arg7, %c1_i64 : i64 loc(#loc59)
        %19 = arith.cmpi ne, %18, %1 : i64 loc(#loc69)
        scf.condition(%19) %18 : i64 loc(#loc50)
      } do {
      ^bb0(%arg7: i64 loc(fused<#di_lexical_block12>[#loc3])):
        scf.yield %arg7 : i64 loc(#loc50)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc50)
    } loc(#loc50)
    return loc(#loc45)
  } loc(#loc44)
  handshake.func @_Z22edit_distance_step_dsaPKjS0_S0_S0_S0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg3: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg6: i32 loc(fused<#di_subprogram5>[#loc12]), %arg7: i1 loc(fused<#di_subprogram5>[#loc12]), ...) -> i1 attributes {argNames = ["input_left", "input_top", "input_diag", "input_char_a", "input_char_b", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg7 : i1 loc(#loc46)
    %1 = handshake.join %0 : none loc(#loc46)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg6, %3 : i32 loc(#loc60)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc51)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc51)
    %7 = arith.index_cast %4 : i64 to index loc(#loc51)
    %8 = arith.index_cast %arg6 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %dataResult, %addressResults = handshake.load [%afterValue] %21#0, %24 : index, i32 loc(#loc70)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %19#0, %33 : index, i32 loc(#loc70)
    %9 = arith.cmpi ne, %dataResult, %dataResult_0 : i32 loc(#loc70)
    %10 = arith.extui %9 : i1 to i32 loc(#loc70)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %18#0, %29 : index, i32 loc(#loc71)
    %11 = arith.addi %dataResult_2, %2 : i32 loc(#loc71)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %22#0, %35 : index, i32 loc(#loc72)
    %12 = arith.addi %dataResult_4, %2 : i32 loc(#loc72)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue] %20#0, %31 : index, i32 loc(#loc73)
    %13 = arith.addi %dataResult_6, %10 : i32 loc(#loc73)
    %14 = arith.cmpi ult, %12, %11 : i32 loc(#loc74)
    %15 = arith.select %14, %12, %11 : i32 loc(#loc74)
    %16 = arith.cmpi ult, %13, %15 : i32 loc(#loc75)
    %17 = arith.select %16, %13, %15 : i32 loc(#loc75)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %17, %37 : index, i32 loc(#loc76)
    %18:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %19:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %20:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_7) {id = 2 : i32} : (index) -> (i32, none) loc(#loc46)
    %21:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 3 : i32} : (index) -> (i32, none) loc(#loc46)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_5) {id = 4 : i32} : (index) -> (i32, none) loc(#loc46)
    %23 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 5 : i32} : (i32, index) -> none loc(#loc46)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc51)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %21#1 : none loc(#loc51)
    %25 = handshake.constant %1 {value = 0 : index} : index loc(#loc51)
    %26 = handshake.constant %1 {value = 1 : index} : index loc(#loc51)
    %27 = arith.select %5, %26, %25 : index loc(#loc51)
    %28 = handshake.mux %27 [%falseResult_10, %trueResult] : index, none loc(#loc51)
    %29 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc51)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %18#1 : none loc(#loc51)
    %30 = handshake.mux %27 [%falseResult_12, %trueResult] : index, none loc(#loc51)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc51)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %20#1 : none loc(#loc51)
    %32 = handshake.mux %27 [%falseResult_14, %trueResult] : index, none loc(#loc51)
    %33 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc51)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %19#1 : none loc(#loc51)
    %34 = handshake.mux %27 [%falseResult_16, %trueResult] : index, none loc(#loc51)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc51)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %22#1 : none loc(#loc51)
    %36 = handshake.mux %27 [%falseResult_18, %trueResult] : index, none loc(#loc51)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc51)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %23 : none loc(#loc51)
    %38 = handshake.mux %27 [%falseResult_20, %trueResult] : index, none loc(#loc51)
    %39 = handshake.join %28, %30, %32, %34, %36, %38 : none, none, none, none, none, none loc(#loc46)
    %40 = handshake.constant %39 {value = true} : i1 loc(#loc46)
    handshake.return %40 : i1 loc(#loc46)
  } loc(#loc46)
  handshake.func @_Z22edit_distance_step_dsaPKjS0_S0_S0_S0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg3: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc12]), %arg6: i32 loc(fused<#di_subprogram5>[#loc12]), %arg7: none loc(fused<#di_subprogram5>[#loc12]), ...) -> none attributes {argNames = ["input_left", "input_top", "input_diag", "input_char_a", "input_char_b", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg7 : none loc(#loc46)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg6, %2 : i32 loc(#loc60)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc51)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc51)
    %6 = arith.index_cast %3 : i64 to index loc(#loc51)
    %7 = arith.index_cast %arg6 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %dataResult, %addressResults = handshake.load [%afterValue] %20#0, %23 : index, i32 loc(#loc70)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %18#0, %32 : index, i32 loc(#loc70)
    %8 = arith.cmpi ne, %dataResult, %dataResult_0 : i32 loc(#loc70)
    %9 = arith.extui %8 : i1 to i32 loc(#loc70)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %17#0, %28 : index, i32 loc(#loc71)
    %10 = arith.addi %dataResult_2, %1 : i32 loc(#loc71)
    %dataResult_4, %addressResults_5 = handshake.load [%afterValue] %21#0, %34 : index, i32 loc(#loc72)
    %11 = arith.addi %dataResult_4, %1 : i32 loc(#loc72)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue] %19#0, %30 : index, i32 loc(#loc73)
    %12 = arith.addi %dataResult_6, %9 : i32 loc(#loc73)
    %13 = arith.cmpi ult, %11, %10 : i32 loc(#loc74)
    %14 = arith.select %13, %11, %10 : i32 loc(#loc74)
    %15 = arith.cmpi ult, %12, %14 : i32 loc(#loc75)
    %16 = arith.select %15, %12, %14 : i32 loc(#loc75)
    %dataResult_8, %addressResult = handshake.store [%afterValue] %16, %36 : index, i32 loc(#loc76)
    %17:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %18:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %19:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_7) {id = 2 : i32} : (index) -> (i32, none) loc(#loc46)
    %20:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 3 : i32} : (index) -> (i32, none) loc(#loc46)
    %21:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_5) {id = 4 : i32} : (index) -> (i32, none) loc(#loc46)
    %22 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 5 : i32} : (i32, index) -> none loc(#loc46)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc51)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %20#1 : none loc(#loc51)
    %24 = handshake.constant %0 {value = 0 : index} : index loc(#loc51)
    %25 = handshake.constant %0 {value = 1 : index} : index loc(#loc51)
    %26 = arith.select %4, %25, %24 : index loc(#loc51)
    %27 = handshake.mux %26 [%falseResult_10, %trueResult] : index, none loc(#loc51)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc51)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %17#1 : none loc(#loc51)
    %29 = handshake.mux %26 [%falseResult_12, %trueResult] : index, none loc(#loc51)
    %30 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc51)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %19#1 : none loc(#loc51)
    %31 = handshake.mux %26 [%falseResult_14, %trueResult] : index, none loc(#loc51)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc51)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %18#1 : none loc(#loc51)
    %33 = handshake.mux %26 [%falseResult_16, %trueResult] : index, none loc(#loc51)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc51)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %21#1 : none loc(#loc51)
    %35 = handshake.mux %26 [%falseResult_18, %trueResult] : index, none loc(#loc51)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc51)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %22 : none loc(#loc51)
    %37 = handshake.mux %26 [%falseResult_20, %trueResult] : index, none loc(#loc51)
    %38 = handshake.join %27, %29, %31, %33, %35, %37 : none, none, none, none, none, none loc(#loc46)
    handshake.return %38 : none loc(#loc47)
  } loc(#loc46)
  func.func private @llvm.umin.i32(i32, i32) -> i32 loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc37)
    %false = arith.constant false loc(#loc37)
    %0 = seq.const_clock  low loc(#loc37)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %1 = ub.poison : i64 loc(#loc37)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c97_i32 = arith.constant 97 : i32 loc(#loc2)
    %c64_i64 = arith.constant 64 : i64 loc(#loc2)
    %c64_i32 = arith.constant 64 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<27xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<27xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<64xi32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<64xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc52)
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      %12 = arith.trunci %10 : i64 to i32 loc(#loc52)
      memref.store %12, %alloca[%11] : memref<64xi32> loc(#loc52)
      %13 = arith.trunci %arg0 : i64 to i32 loc(#loc53)
      %14 = arith.addi %13, %c2_i32 : i32 loc(#loc53)
      memref.store %14, %alloca_0[%11] : memref<64xi32> loc(#loc53)
      memref.store %13, %alloca_1[%11] : memref<64xi32> loc(#loc54)
      %15 = arith.andi %13, %c1_i32 : i32 loc(#loc55)
      %16 = arith.addi %15, %c97_i32 : i32 loc(#loc55)
      memref.store %16, %alloca_2[%11] : memref<64xi32> loc(#loc55)
      %17 = arith.andi %12, %c1_i32 : i32 loc(#loc56)
      %18 = arith.addi %17, %c97_i32 : i32 loc(#loc56)
      memref.store %18, %alloca_3[%11] : memref<64xi32> loc(#loc56)
      %19 = arith.cmpi ne, %10, %c64_i64 : i64 loc(#loc57)
      scf.condition(%19) %10 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc23])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %cast = memref.cast %alloca : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_6 = memref.cast %alloca_0 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_7 = memref.cast %alloca_1 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_8 = memref.cast %alloca_2 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_9 = memref.cast %alloca_3 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_10 = memref.cast %alloca_4 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    call @_Z22edit_distance_step_cpuPKjS0_S0_S0_S0_Pjj(%cast, %cast_6, %cast_7, %cast_8, %cast_9, %cast_10, %c64_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc38)
    %cast_11 = memref.cast %alloca_5 : memref<64xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_12, %ready_13 = esi.wrap.vr %cast_6, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_14, %ready_15 = esi.wrap.vr %cast_7, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_16, %ready_17 = esi.wrap.vr %cast_8, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_18, %ready_19 = esi.wrap.vr %cast_9, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_20, %ready_21 = esi.wrap.vr %cast_11, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_22, %ready_23 = esi.wrap.vr %c64_i32, %true : i32 loc(#loc39)
    %chanOutput_24, %ready_25 = esi.wrap.vr %true, %true : i1 loc(#loc39)
    %5 = handshake.esi_instance @_Z22edit_distance_step_dsaPKjS0_S0_S0_S0_Pjj_esi "_Z22edit_distance_step_dsaPKjS0_S0_S0_S0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18, %chanOutput_20, %chanOutput_22, %chanOutput_24) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc39)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc39)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc61)
      %11 = memref.load %alloca_4[%10] : memref<64xi32> loc(#loc61)
      %12 = memref.load %alloca_5[%10] : memref<64xi32> loc(#loc61)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc61)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc49)
        %17 = arith.cmpi eq, %16, %c64_i64 : i64 loc(#loc49)
        %18 = arith.extui %17 : i1 to i32 loc(#loc43)
        %19 = arith.cmpi ne, %16, %c64_i64 : i64 loc(#loc58)
        %20 = arith.extui %19 : i1 to i32 loc(#loc43)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc61)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc61)
      } loc(#loc61)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc43)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc31]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc31]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc31])):
      scf.yield %arg0 : i64 loc(#loc43)
    } loc(#loc43)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc43)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc43)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<27xi8> -> index loc(#loc77)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc77)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc77)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc77)
      scf.yield %c1_i32 : i32 loc(#loc78)
    } loc(#loc43)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<27xi8> -> index loc(#loc40)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc40)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc40)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc40)
    } loc(#loc2)
    return %9 : i32 loc(#loc41)
  } loc(#loc37)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/edit_distance_step/edit_distance_step.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":23:0)
#loc5 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":26:0)
#loc6 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":27:0)
#loc7 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":28:0)
#loc8 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":30:0)
#loc9 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":31:0)
#loc10 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":33:0)
#loc11 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":35:0)
#loc13 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":48:0)
#loc14 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":49:0)
#loc15 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":52:0)
#loc16 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":53:0)
#loc17 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":54:0)
#loc18 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":56:0)
#loc19 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":57:0)
#loc20 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":59:0)
#loc21 = loc("tests/app/edit_distance_step/edit_distance_step.cpp":61:0)
#loc22 = loc("tests/app/edit_distance_step/main.cpp":5:0)
#loc24 = loc("tests/app/edit_distance_step/main.cpp":16:0)
#loc25 = loc("tests/app/edit_distance_step/main.cpp":17:0)
#loc26 = loc("tests/app/edit_distance_step/main.cpp":18:0)
#loc27 = loc("tests/app/edit_distance_step/main.cpp":19:0)
#loc28 = loc("tests/app/edit_distance_step/main.cpp":20:0)
#loc29 = loc("tests/app/edit_distance_step/main.cpp":28:0)
#loc30 = loc("tests/app/edit_distance_step/main.cpp":31:0)
#loc32 = loc("tests/app/edit_distance_step/main.cpp":35:0)
#loc33 = loc("tests/app/edit_distance_step/main.cpp":36:0)
#loc34 = loc("tests/app/edit_distance_step/main.cpp":37:0)
#loc35 = loc("tests/app/edit_distance_step/main.cpp":41:0)
#loc36 = loc("tests/app/edit_distance_step/main.cpp":43:0)
#loc37 = loc(fused<#di_subprogram3>[#loc22])
#loc38 = loc(fused<#di_subprogram3>[#loc29])
#loc39 = loc(fused<#di_subprogram3>[#loc30])
#loc40 = loc(fused<#di_subprogram3>[#loc35])
#loc41 = loc(fused<#di_subprogram3>[#loc36])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 15>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 34>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 48>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 15>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 34>
#loc45 = loc(fused<#di_subprogram4>[#loc11])
#loc47 = loc(fused<#di_subprogram5>[#loc21])
#loc48 = loc(fused<#di_lexical_block10>[#loc23])
#loc49 = loc(fused<#di_lexical_block11>[#loc31])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 22>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 48>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 35>
#loc51 = loc(fused<#di_lexical_block13>[#loc13])
#loc52 = loc(fused<#di_lexical_block14>[#loc24])
#loc53 = loc(fused<#di_lexical_block14>[#loc25])
#loc54 = loc(fused<#di_lexical_block14>[#loc26])
#loc55 = loc(fused<#di_lexical_block14>[#loc27])
#loc56 = loc(fused<#di_lexical_block14>[#loc28])
#loc57 = loc(fused[#loc42, #loc48])
#loc58 = loc(fused[#loc43, #loc49])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 22>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 48>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 35>
#loc59 = loc(fused<#di_lexical_block16>[#loc3])
#loc60 = loc(fused<#di_lexical_block17>[#loc13])
#loc61 = loc(fused<#di_lexical_block18>[#loc32])
#loc62 = loc(fused<#di_lexical_block19>[#loc4])
#loc63 = loc(fused<#di_lexical_block19>[#loc5])
#loc64 = loc(fused<#di_lexical_block19>[#loc6])
#loc65 = loc(fused<#di_lexical_block19>[#loc7])
#loc66 = loc(fused<#di_lexical_block19>[#loc8])
#loc67 = loc(fused<#di_lexical_block19>[#loc9])
#loc68 = loc(fused<#di_lexical_block19>[#loc10])
#loc69 = loc(fused[#loc50, #loc59])
#loc70 = loc(fused<#di_lexical_block20>[#loc14])
#loc71 = loc(fused<#di_lexical_block20>[#loc15])
#loc72 = loc(fused<#di_lexical_block20>[#loc16])
#loc73 = loc(fused<#di_lexical_block20>[#loc17])
#loc74 = loc(fused<#di_lexical_block20>[#loc18])
#loc75 = loc(fused<#di_lexical_block20>[#loc19])
#loc76 = loc(fused<#di_lexical_block20>[#loc20])
#loc77 = loc(fused<#di_lexical_block21>[#loc33])
#loc78 = loc(fused<#di_lexical_block21>[#loc34])
