#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/bitrev_complex/bitrev_complex.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/bitrev_complex/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":12:0)
#loc3 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":17:0)
#loc4 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":22:0)
#loc10 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":35:0)
#loc19 = loc("tests/app/bitrev_complex/main.cpp":20:0)
#loc24 = loc("tests/app/bitrev_complex/main.cpp":29:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 4096, elements = #llvm.di_subrange<count = 128 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 42>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 20>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 29>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 42>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_real", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_imag", file = #di_file1, line = 11, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_real", file = #di_file1, line = 14, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_imag", file = #di_file1, line = 15, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_real", file = #di_file1, line = 16, type = #di_composite_type>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_imag", file = #di_file1, line = 17, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 42>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_real", file = #di_file, line = 14, arg = 3, type = #di_derived_type4>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_imag", file = #di_file, line = 15, arg = 4, type = #di_derived_type4>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_real", file = #di_file, line = 37, arg = 3, type = #di_derived_type4>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_imag", file = #di_file, line = 38, arg = 4, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 42, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 20, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 29, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_real", file = #di_file, line = 12, arg = 1, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_imag", file = #di_file, line = 13, arg = 2, type = #di_derived_type6>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 5, type = #di_derived_type7>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "j", file = #di_file, line = 18, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "k", file = #di_file, line = 19, type = #di_derived_type5>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "m", file = #di_file, line = 20, type = #di_derived_type5>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_real", file = #di_file, line = 35, arg = 1, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_imag", file = #di_file, line = 36, arg = 2, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 39, arg = 5, type = #di_derived_type7>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "j", file = #di_file, line = 43, type = #di_derived_type5>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "k", file = #di_file, line = 44, type = #di_derived_type5>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "m", file = #di_file, line = 45, type = #di_derived_type5>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 7, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_derived_type4, #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bitrev_complex_cpu", linkageName = "_Z18bitrev_complex_cpuPKfS0_PfS1_j", file = #di_file, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable14, #di_local_variable15, #di_local_variable6, #di_local_variable7, #di_local_variable16, #di_local_variable8, #di_local_variable17, #di_local_variable18, #di_local_variable19>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "bitrev_complex_dsa", linkageName = "_Z18bitrev_complex_dsaPKfS0_PfS1_j", file = #di_file, line = 35, scopeLine = 39, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable20, #di_local_variable21, #di_local_variable9, #di_local_variable10, #di_local_variable22, #di_local_variable11, #di_local_variable23, #di_local_variable24, #di_local_variable25>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable26, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5, #di_local_variable12, #di_local_variable13>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 20>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 29>
#loc31 = loc(fused<#di_subprogram3>[#loc1])
#loc33 = loc(fused<#di_subprogram4>[#loc10])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 17>
#loc40 = loc(fused<#di_lexical_block8>[#loc3])
#loc42 = loc(fused<#di_lexical_block10>[#loc19])
#loc43 = loc(fused<#di_lexical_block11>[#loc24])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 17>
#loc48 = loc(fused<#di_lexical_block16>[#loc4])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<44xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 105, 116, 114, 101, 118, 95, 99, 111, 109, 112, 108, 101, 120, 47, 98, 105, 116, 114, 101, 118, 95, 99, 111, 109, 112, 108, 101, 120, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<23xi8> = dense<[98, 105, 116, 114, 101, 118, 95, 99, 111, 109, 112, 108, 101, 120, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<23xi8> = dense<[98, 105, 116, 114, 101, 118, 95, 99, 111, 109, 112, 108, 101, 120, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z18bitrev_complex_cpuPKfS0_PfS1_j(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg3: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram3>[#loc1]), %arg4: i32 loc(fused<#di_subprogram3>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc44)
    scf.if %0 {
    } else {
      %1 = arith.shrui %arg4, %c1_i32 : i32 loc(#loc2)
      %2 = arith.cmpi eq, %1, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg4 : i32 to i64 loc(#loc44)
      %4 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %5 = scf.if %2 -> (i64) {
          scf.yield %c0_i64 : i64 loc(#loc48)
        } else {
          %12 = arith.trunci %arg5 : i64 to i32 loc(#loc48)
          %13:3 = scf.while (%arg6 = %1, %arg7 = %12, %arg8 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
            %15 = arith.shli %arg8, %c1_i32 : i32 loc(#loc60)
            %16 = arith.andi %arg7, %c1_i32 : i32 loc(#loc60)
            %17 = arith.ori %16, %15 : i32 loc(#loc60)
            %18 = arith.shrui %arg7, %c1_i32 : i32 loc(#loc61)
            %19 = arith.shrui %arg6, %c1_i32 : i32 loc(#loc49)
            %20 = arith.cmpi ne, %19, %c0_i32 : i32 loc(#loc48)
            scf.condition(%20) %19, %18, %17 : i32, i32, i32 loc(#loc48)
          } do {
          ^bb0(%arg6: i32 loc(fused<#di_lexical_block16>[#loc4]), %arg7: i32 loc(fused<#di_lexical_block16>[#loc4]), %arg8: i32 loc(fused<#di_lexical_block16>[#loc4])):
            scf.yield %arg6, %arg7, %arg8 : i32, i32, i32 loc(#loc48)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc48)
          %14 = arith.extui %13#2 : i32 to i64 loc(#loc50)
          scf.yield %14 : i64 loc(#loc50)
        } loc(#loc48)
        %6 = arith.index_cast %arg5 : i64 to index loc(#loc50)
        %7 = memref.load %arg0[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %8 = arith.index_cast %5 : i64 to index loc(#loc50)
        memref.store %7, %arg2[%8] : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
        %9 = memref.load %arg1[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc51)
        memref.store %9, %arg3[%8] : memref<?xf32, strided<[1], offset: ?>> loc(#loc51)
        %10 = arith.addi %arg5, %c1_i64 : i64 loc(#loc44)
        %11 = arith.cmpi ne, %10, %3 : i64 loc(#loc52)
        scf.condition(%11) %10 : i64 loc(#loc40)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block8>[#loc3])):
        scf.yield %arg5 : i64 loc(#loc40)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc40)
    } loc(#loc40)
    return loc(#loc32)
  } loc(#loc31)
  handshake.func @_Z18bitrev_complex_dsaPKfS0_PfS1_j_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg4: i32 loc(fused<#di_subprogram4>[#loc10]), %arg5: i1 loc(fused<#di_subprogram4>[#loc10]), ...) -> i1 attributes {argNames = ["input_real", "input_imag", "output_real", "output_imag", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc33)
    %1 = handshake.join %0 : none loc(#loc33)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc45)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc41)
    %6 = arith.shrui %arg4, %2 : i32 loc(#loc2)
    %7 = arith.cmpi eq, %6, %3 : i32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc41)
    %9 = arith.index_cast %4 : i64 to index loc(#loc41)
    %10 = arith.index_cast %arg4 : i32 to index loc(#loc41)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc41)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc41)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc41)
    %12 = arith.index_cast %afterValue : index to i64 loc(#loc41)
    %13 = dataflow.invariant %afterCond, %7 : i1, i1 -> i1 loc(#loc53)
    %trueResult_0, %falseResult_1 = handshake.cond_br %13, %11 : none loc(#loc53)
    %14 = arith.trunci %12 : i64 to i32 loc(#loc53)
    %15 = dataflow.invariant %afterCond, %6 : i1, i32 -> i32 loc(#loc53)
    %16 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc53)
    %17 = arith.index_cast %15 : i32 to index loc(#loc53)
    %18 = arith.index_cast %3 : i32 to index loc(#loc53)
    %index_2, %willContinue_3 = dataflow.stream %17, %16, %18 {step_op = ">>=", stop_cond = "!="} loc(#loc53)
    %19 = dataflow.carry %willContinue_3, %14, %24 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_4, %afterCond_5 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_5 : i1 loc(#loc53)
    %20 = dataflow.carry %willContinue_3, %3, %23 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_6, %afterCond_7 = dataflow.gate %20, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_7 : i1 loc(#loc53)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %20 : i32 loc(#loc53)
    %21 = arith.shli %afterValue_6, %2 : i32 loc(#loc62)
    %22 = arith.andi %afterValue_4, %2 : i32 loc(#loc62)
    %23 = arith.ori %22, %21 : i32 loc(#loc62)
    %24 = arith.shrui %afterValue_4, %2 : i32 loc(#loc63)
    %25 = arith.extui %falseResult_9 : i32 to i64 loc(#loc54)
    %26 = handshake.constant %11 {value = 0 : index} : index loc(#loc53)
    %27 = handshake.constant %11 {value = 1 : index} : index loc(#loc53)
    %28 = arith.select %13, %27, %26 : index loc(#loc53)
    %29 = handshake.mux %28 [%25, %4] : index, i64 loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %32#0, %35 : index, f32 loc(#loc54)
    %30 = arith.index_cast %29 : i64 to index loc(#loc54)
    %dataResult_10, %addressResult = handshake.store [%30] %dataResult, %42 : index, f32 loc(#loc54)
    %dataResult_11, %addressResults_12 = handshake.load [%afterValue] %31#0, %40 : index, f32 loc(#loc55)
    %dataResult_13, %addressResult_14 = handshake.store [%30] %dataResult_11, %44 : index, f32 loc(#loc55)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_12) {id = 0 : i32} : (index) -> (f32, none) loc(#loc33)
    %32:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc33)
    %33 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc33)
    %34 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_13, %addressResult_14) {id = 3 : i32} : (f32, index) -> none loc(#loc33)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc41)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %32#1 : none loc(#loc41)
    %36 = handshake.constant %1 {value = 0 : index} : index loc(#loc41)
    %37 = handshake.constant %1 {value = 1 : index} : index loc(#loc41)
    %38 = arith.select %5, %37, %36 : index loc(#loc41)
    %39 = handshake.mux %38 [%falseResult_16, %trueResult] : index, none loc(#loc41)
    %40 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc41)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %31#1 : none loc(#loc41)
    %41 = handshake.mux %38 [%falseResult_18, %trueResult] : index, none loc(#loc41)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc41)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %33 : none loc(#loc41)
    %43 = handshake.mux %38 [%falseResult_20, %trueResult] : index, none loc(#loc41)
    %44 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc41)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %34 : none loc(#loc41)
    %45 = handshake.mux %38 [%falseResult_22, %trueResult] : index, none loc(#loc41)
    %46 = handshake.join %39, %41, %43, %45 : none, none, none, none loc(#loc33)
    %47 = handshake.constant %46 {value = true} : i1 loc(#loc33)
    handshake.return %47 : i1 loc(#loc33)
  } loc(#loc33)
  handshake.func @_Z18bitrev_complex_dsaPKfS0_PfS1_j(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc10]), %arg4: i32 loc(fused<#di_subprogram4>[#loc10]), %arg5: none loc(fused<#di_subprogram4>[#loc10]), ...) -> none attributes {argNames = ["input_real", "input_imag", "output_real", "output_imag", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc33)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc45)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc41)
    %5 = arith.shrui %arg4, %1 : i32 loc(#loc2)
    %6 = arith.cmpi eq, %5, %2 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc41)
    %8 = arith.index_cast %3 : i64 to index loc(#loc41)
    %9 = arith.index_cast %arg4 : i32 to index loc(#loc41)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc41)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc41)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc41)
    %11 = arith.index_cast %afterValue : index to i64 loc(#loc41)
    %12 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc53)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %10 : none loc(#loc53)
    %13 = arith.trunci %11 : i64 to i32 loc(#loc53)
    %14 = dataflow.invariant %afterCond, %5 : i1, i32 -> i32 loc(#loc53)
    %15 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc53)
    %16 = arith.index_cast %14 : i32 to index loc(#loc53)
    %17 = arith.index_cast %2 : i32 to index loc(#loc53)
    %index_2, %willContinue_3 = dataflow.stream %16, %15, %17 {step_op = ">>=", stop_cond = "!="} loc(#loc53)
    %18 = dataflow.carry %willContinue_3, %13, %23 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_4, %afterCond_5 = dataflow.gate %18, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_5 : i1 loc(#loc53)
    %19 = dataflow.carry %willContinue_3, %2, %22 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_6, %afterCond_7 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_7 : i1 loc(#loc53)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %19 : i32 loc(#loc53)
    %20 = arith.shli %afterValue_6, %1 : i32 loc(#loc62)
    %21 = arith.andi %afterValue_4, %1 : i32 loc(#loc62)
    %22 = arith.ori %21, %20 : i32 loc(#loc62)
    %23 = arith.shrui %afterValue_4, %1 : i32 loc(#loc63)
    %24 = arith.extui %falseResult_9 : i32 to i64 loc(#loc54)
    %25 = handshake.constant %10 {value = 0 : index} : index loc(#loc53)
    %26 = handshake.constant %10 {value = 1 : index} : index loc(#loc53)
    %27 = arith.select %12, %26, %25 : index loc(#loc53)
    %28 = handshake.mux %27 [%24, %3] : index, i64 loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %31#0, %34 : index, f32 loc(#loc54)
    %29 = arith.index_cast %28 : i64 to index loc(#loc54)
    %dataResult_10, %addressResult = handshake.store [%29] %dataResult, %41 : index, f32 loc(#loc54)
    %dataResult_11, %addressResults_12 = handshake.load [%afterValue] %30#0, %39 : index, f32 loc(#loc55)
    %dataResult_13, %addressResult_14 = handshake.store [%29] %dataResult_11, %43 : index, f32 loc(#loc55)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_12) {id = 0 : i32} : (index) -> (f32, none) loc(#loc33)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc33)
    %32 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc33)
    %33 = handshake.extmemory[ld = 0, st = 1] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_13, %addressResult_14) {id = 3 : i32} : (f32, index) -> none loc(#loc33)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc41)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %31#1 : none loc(#loc41)
    %35 = handshake.constant %0 {value = 0 : index} : index loc(#loc41)
    %36 = handshake.constant %0 {value = 1 : index} : index loc(#loc41)
    %37 = arith.select %4, %36, %35 : index loc(#loc41)
    %38 = handshake.mux %37 [%falseResult_16, %trueResult] : index, none loc(#loc41)
    %39 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc41)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %30#1 : none loc(#loc41)
    %40 = handshake.mux %37 [%falseResult_18, %trueResult] : index, none loc(#loc41)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc41)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %32 : none loc(#loc41)
    %42 = handshake.mux %37 [%falseResult_20, %trueResult] : index, none loc(#loc41)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc41)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %33 : none loc(#loc41)
    %44 = handshake.mux %37 [%falseResult_22, %trueResult] : index, none loc(#loc41)
    %45 = handshake.join %38, %40, %42, %44 : none, none, none, none loc(#loc33)
    handshake.return %45 : none loc(#loc34)
  } loc(#loc33)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc35)
    %c2_i32 = arith.constant 2 : i32 loc(#loc35)
    %1 = ub.poison : i64 loc(#loc35)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c128_i32 = arith.constant 128 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c128_i64 = arith.constant 128 : i64 loc(#loc2)
    %cst = arith.constant 9.99999997E-7 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<23xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<23xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<128xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc56)
      %11 = arith.uitofp %10 : i32 to f32 loc(#loc56)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc56)
      memref.store %11, %alloca[%12] : memref<128xf32> loc(#loc56)
      %13 = arith.subi %c128_i32, %10 : i32 loc(#loc57)
      %14 = arith.uitofp %13 : i32 to f32 loc(#loc57)
      memref.store %14, %alloca_0[%12] : memref<128xf32> loc(#loc57)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc46)
      %16 = arith.cmpi ne, %15, %c128_i64 : i64 loc(#loc58)
      scf.condition(%16) %15 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc19])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %cast = memref.cast %alloca : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc36)
    %cast_5 = memref.cast %alloca_0 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc36)
    %cast_6 = memref.cast %alloca_1 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc36)
    %cast_7 = memref.cast %alloca_2 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc36)
    call @_Z18bitrev_complex_cpuPKfS0_PfS1_j(%cast, %cast_5, %cast_6, %cast_7, %c128_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc36)
    %cast_8 = memref.cast %alloca_3 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %cast_9 = memref.cast %alloca_4 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %chanOutput_10, %ready_11 = esi.wrap.vr %cast_5, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %chanOutput_12, %ready_13 = esi.wrap.vr %cast_8, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %chanOutput_14, %ready_15 = esi.wrap.vr %cast_9, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %chanOutput_16, %ready_17 = esi.wrap.vr %c128_i32, %true : i32 loc(#loc37)
    %chanOutput_18, %ready_19 = esi.wrap.vr %true, %true : i1 loc(#loc37)
    %5 = handshake.esi_instance @_Z18bitrev_complex_dsaPKfS0_PfS1_j_esi "_Z18bitrev_complex_dsaPKfS0_PfS1_j_inst0" clk %0 rst %false(%chanOutput, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc37)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc37)
    %6:2 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc64)
      %11 = memref.load %alloca_1[%10] : memref<128xf32> loc(#loc64)
      %12 = memref.load %alloca_3[%10] : memref<128xf32> loc(#loc64)
      %13 = arith.subf %11, %12 : f32 loc(#loc64)
      %14 = math.absf %13 : f32 loc(#loc64)
      %15 = arith.cmpf ogt, %14, %cst : f32 loc(#loc64)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc64)
      } else {
        %18 = memref.load %alloca_2[%10] : memref<128xf32> loc(#loc65)
        %19 = memref.load %alloca_4[%10] : memref<128xf32> loc(#loc65)
        %20 = arith.subf %18, %19 : f32 loc(#loc65)
        %21 = math.absf %20 : f32 loc(#loc65)
        %22 = arith.cmpf ogt, %21, %cst : f32 loc(#loc65)
        %23:3 = scf.if %22 -> (i64, i32, i32) {
          scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc64)
        } else {
          %24 = arith.addi %arg0, %c1_i64 : i64 loc(#loc47)
          %25 = arith.cmpi eq, %24, %c128_i64 : i64 loc(#loc47)
          %26 = arith.extui %25 : i1 to i32 loc(#loc43)
          %27 = arith.cmpi ne, %24, %c128_i64 : i64 loc(#loc59)
          %28 = arith.extui %27 : i1 to i32 loc(#loc43)
          scf.yield %24, %26, %28 : i64, i32, i32 loc(#loc64)
        } loc(#loc64)
        scf.yield %23#0, %23#1, %23#2 : i64, i32, i32 loc(#loc64)
      } loc(#loc64)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc43)
      scf.condition(%17) %16#0, %16#1 : i64, i32 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc24]), %arg1: i32 loc(fused<#di_lexical_block11>[#loc24])):
      scf.yield %arg0 : i64 loc(#loc43)
    } loc(#loc43)
    %7 = arith.index_castui %6#1 : i32 to index loc(#loc43)
    %8:2 = scf.index_switch %7 -> i1, i32 
    case 1 {
      scf.yield %true, %c0_i32 : i1, i32 loc(#loc43)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<23xi8> -> index loc(#loc66)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc66)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc66)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc66)
      scf.yield %false, %c1_i32 : i1, i32 loc(#loc67)
    } loc(#loc43)
    %9 = arith.select %8#0, %c0_i32, %8#1 : i32 loc(#loc2)
    scf.if %8#0 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<23xi8> -> index loc(#loc38)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc38)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc38)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc38)
    } loc(#loc2)
    return %9 : i32 loc(#loc39)
  } loc(#loc35)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/bitrev_complex/bitrev_complex.cpp":0:0)
#loc2 = loc(unknown)
#loc5 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":23:0)
#loc6 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":24:0)
#loc7 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":28:0)
#loc8 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":29:0)
#loc9 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":31:0)
#loc11 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":42:0)
#loc12 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":47:0)
#loc13 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":48:0)
#loc14 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":49:0)
#loc15 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":53:0)
#loc16 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":54:0)
#loc17 = loc("tests/app/bitrev_complex/bitrev_complex.cpp":56:0)
#loc18 = loc("tests/app/bitrev_complex/main.cpp":6:0)
#loc20 = loc("tests/app/bitrev_complex/main.cpp":21:0)
#loc21 = loc("tests/app/bitrev_complex/main.cpp":22:0)
#loc22 = loc("tests/app/bitrev_complex/main.cpp":26:0)
#loc23 = loc("tests/app/bitrev_complex/main.cpp":27:0)
#loc25 = loc("tests/app/bitrev_complex/main.cpp":30:0)
#loc26 = loc("tests/app/bitrev_complex/main.cpp":31:0)
#loc27 = loc("tests/app/bitrev_complex/main.cpp":32:0)
#loc28 = loc("tests/app/bitrev_complex/main.cpp":33:0)
#loc29 = loc("tests/app/bitrev_complex/main.cpp":37:0)
#loc30 = loc("tests/app/bitrev_complex/main.cpp":39:0)
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 42>
#loc32 = loc(fused<#di_subprogram3>[#loc9])
#loc34 = loc(fused<#di_subprogram4>[#loc17])
#loc35 = loc(fused<#di_subprogram5>[#loc18])
#loc36 = loc(fused<#di_subprogram5>[#loc22])
#loc37 = loc(fused<#di_subprogram5>[#loc23])
#loc38 = loc(fused<#di_subprogram5>[#loc29])
#loc39 = loc(fused<#di_subprogram5>[#loc30])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 42>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 20>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 29>
#loc41 = loc(fused<#di_lexical_block9>[#loc11])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 42>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 20>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 29>
#loc44 = loc(fused<#di_lexical_block12>[#loc3])
#loc45 = loc(fused<#di_lexical_block13>[#loc11])
#loc46 = loc(fused<#di_lexical_block14>[#loc19])
#loc47 = loc(fused<#di_lexical_block15>[#loc24])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 22>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 47>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 30>
#loc49 = loc(fused<#di_lexical_block16>[#loc])
#loc50 = loc(fused<#di_lexical_block16>[#loc7])
#loc51 = loc(fused<#di_lexical_block16>[#loc8])
#loc52 = loc(fused[#loc40, #loc44])
#loc53 = loc(fused<#di_lexical_block17>[#loc12])
#loc54 = loc(fused<#di_lexical_block17>[#loc15])
#loc55 = loc(fused<#di_lexical_block17>[#loc16])
#loc56 = loc(fused<#di_lexical_block18>[#loc20])
#loc57 = loc(fused<#di_lexical_block18>[#loc21])
#loc58 = loc(fused[#loc42, #loc46])
#loc59 = loc(fused[#loc43, #loc47])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 31>
#loc60 = loc(fused<#di_lexical_block20>[#loc5])
#loc61 = loc(fused<#di_lexical_block20>[#loc6])
#loc62 = loc(fused<#di_lexical_block21>[#loc13])
#loc63 = loc(fused<#di_lexical_block21>[#loc14])
#loc64 = loc(fused<#di_lexical_block22>[#loc25])
#loc65 = loc(fused<#di_lexical_block22>[#loc26])
#loc66 = loc(fused<#di_lexical_block23>[#loc27])
#loc67 = loc(fused<#di_lexical_block23>[#loc28])
