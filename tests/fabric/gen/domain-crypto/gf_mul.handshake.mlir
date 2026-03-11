#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/gf_mul/gf_mul.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/gf_mul/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/gf_mul/gf_mul.cpp":13:0)
#loc3 = loc("tests/app/gf_mul/gf_mul.cpp":17:0)
#loc6 = loc("tests/app/gf_mul/gf_mul.cpp":22:0)
#loc14 = loc("tests/app/gf_mul/gf_mul.cpp":40:0)
#loc27 = loc("tests/app/gf_mul/main.cpp":17:0)
#loc32 = loc("tests/app/gf_mul/main.cpp":29:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 46>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 29>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 46>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 46>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 46, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 29, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 22>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 51>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 16, arg = 4, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "a", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "b", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "p", file = #di_file, line = 20, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 43, arg = 4, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "a", file = #di_file, line = 47, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "b", file = #di_file, line = 48, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "p", file = #di_file, line = 49, type = #di_derived_type1>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_A", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_B", file = #di_file1, line = 10, type = #di_composite_type>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 13, type = #di_composite_type>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 14, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 22>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 51>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_C", file = #di_file, line = 15, arg = 3, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_C", file = #di_file, line = 42, arg = 3, type = #di_derived_type5>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file, line = 51, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable2, #di_local_variable3>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 22>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 51>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 17>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 29>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_A", file = #di_file, line = 13, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_B", file = #di_file, line = 14, arg = 2, type = #di_derived_type6>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_A", file = #di_file, line = 40, arg = 1, type = #di_derived_type6>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_B", file = #di_file, line = 41, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_lexical_block12, name = "hi_bit_set", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block13, name = "hi_bit_set", file = #di_file, line = 55, type = #di_derived_type1>
#loc43 = loc(fused<#di_lexical_block14>[#loc27])
#loc44 = loc(fused<#di_lexical_block15>[#loc32])
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gf_mul_cpu", linkageName = "_Z10gf_mul_cpuPKjS0_Pjj", file = #di_file, line = 13, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable17, #di_local_variable4, #di_local_variable, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable18, #di_local_variable25>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gf_mul_dsa", linkageName = "_Z10gf_mul_dsaPKjS0_Pjj", file = #di_file, line = 40, scopeLine = 43, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable23, #di_local_variable24, #di_local_variable19, #di_local_variable8, #di_local_variable1, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable20, #di_local_variable26>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 17>
#loc47 = loc(fused<#di_subprogram4>[#loc1])
#loc49 = loc(fused<#di_subprogram5>[#loc14])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 17>
#loc55 = loc(fused<#di_lexical_block20>[#loc3])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 17>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 22>
#loc69 = loc(fused<#di_lexical_block28>[#loc6])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 103, 102, 95, 109, 117, 108, 47, 103, 102, 95, 109, 117, 108, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<15xi8> = dense<[103, 102, 95, 109, 117, 108, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[103, 102, 95, 109, 117, 108, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z10gf_mul_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c128_i32 = arith.constant 128 : i32 loc(#loc2)
    %c27_i32 = arith.constant 27 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc58)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc58)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc62)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
        %5 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc63)
        %6 = arith.andi %5, %c255_i32 : i32 loc(#loc63)
        %7:4 = scf.while (%arg5 = %c0_i32, %arg6 = %c0_i32, %arg7 = %6, %arg8 = %4) : (i32, i32, i32, i32) -> (i32, i32, i32, i32) {
          %11 = arith.andi %arg7, %c1_i32 : i32 loc(#loc79)
          %12 = arith.cmpi eq, %11, %c0_i32 : i32 loc(#loc79)
          %13 = arith.select %12, %c0_i32, %arg8 : i32 loc(#loc79)
          %14 = arith.xori %13, %arg6 : i32 loc(#loc79)
          %15 = arith.andi %arg8, %c128_i32 : i32 loc(#loc72)
          %16 = arith.shli %arg8, %c1_i32 : i32 loc(#loc73)
          %17 = arith.cmpi eq, %15, %c0_i32 : i32 loc(#loc80)
          %18 = arith.xori %16, %c27_i32 : i32 loc(#loc80)
          %19 = arith.select %17, %16, %18 : i32 loc(#loc80)
          %20 = arith.shrui %arg7, %c1_i32 : i32 loc(#loc74)
          %21 = arith.addi %arg5, %c1_i32 : i32 loc(#loc71)
          %22 = arith.cmpi ne, %21, %c8_i32 : i32 loc(#loc75)
          scf.condition(%22) %21, %14, %20, %19 : i32, i32, i32, i32 loc(#loc69)
        } do {
        ^bb0(%arg5: i32 loc(fused<#di_lexical_block28>[#loc6]), %arg6: i32 loc(fused<#di_lexical_block28>[#loc6]), %arg7: i32 loc(fused<#di_lexical_block28>[#loc6]), %arg8: i32 loc(fused<#di_lexical_block28>[#loc6])):
          scf.yield %arg5, %arg6, %arg7, %arg8 : i32, i32, i32, i32 loc(#loc69)
        } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc69)
        %8 = arith.andi %7#1, %c255_i32 : i32 loc(#loc64)
        memref.store %8, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc64)
        %9 = arith.addi %arg4, %c1_i64 : i64 loc(#loc58)
        %10 = arith.cmpi ne, %9, %1 : i64 loc(#loc65)
        scf.condition(%10) %9 : i64 loc(#loc55)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block20>[#loc3])):
        scf.yield %arg4 : i64 loc(#loc55)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc55)
    } loc(#loc55)
    return loc(#loc48)
  } loc(#loc47)
  handshake.func @_Z10gf_mul_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg3: i32 loc(fused<#di_subprogram5>[#loc14]), %arg4: i1 loc(fused<#di_subprogram5>[#loc14]), ...) -> i1 attributes {argNames = ["input_A", "input_B", "output_C", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc49)
    %1 = handshake.join %0 : none loc(#loc49)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 8 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %1 {value = 255 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %1 {value = 128 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %1 {value = 27 : i32} : i32 loc(#loc2)
    %9 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc59)
    %trueResult, %falseResult = handshake.cond_br %9, %1 : none loc(#loc56)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc56)
    %11 = arith.index_cast %5 : i64 to index loc(#loc56)
    %12 = arith.index_cast %arg3 : i32 to index loc(#loc56)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc56)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc56)
    %13 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc56)
    %dataResult, %addressResults = handshake.load [%afterValue] %32#0, %35 : index, i32 loc(#loc66)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %33#0, %42 : index, i32 loc(#loc67)
    %14 = arith.andi %dataResult_0, %6 : i32 loc(#loc67)
    %15 = handshake.constant %13 {value = 1 : index} : index loc(#loc70)
    %16 = arith.index_cast %3 : i32 to index loc(#loc70)
    %17 = arith.index_cast %4 : i32 to index loc(#loc70)
    %index_2, %willContinue_3 = dataflow.stream %16, %15, %17 {step_op = "+=", stop_cond = "!="} loc(#loc70)
    %18 = dataflow.carry %willContinue_3, %3, %24 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_4, %afterCond_5 = dataflow.gate %18, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_5 : i1 loc(#loc70)
    %trueResult_6, %falseResult_7 = handshake.cond_br %willContinue_3, %18 : i32 loc(#loc70)
    %19 = dataflow.carry %willContinue_3, %14, %30 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_8, %afterCond_9 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_9 : i1 loc(#loc70)
    %20 = dataflow.carry %willContinue_3, %dataResult, %29 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_10, %afterCond_11 = dataflow.gate %20, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_11 : i1 loc(#loc70)
    %21 = arith.andi %afterValue_8, %2 : i32 loc(#loc81)
    %22 = arith.cmpi eq, %21, %3 : i32 loc(#loc81)
    %23 = arith.select %22, %3, %afterValue_10 : i32 loc(#loc81)
    %24 = arith.xori %23, %afterValue_4 : i32 loc(#loc81)
    %25 = arith.andi %afterValue_10, %7 : i32 loc(#loc76)
    %26 = arith.shli %afterValue_10, %2 : i32 loc(#loc77)
    %27 = arith.cmpi eq, %25, %3 : i32 loc(#loc82)
    %28 = arith.xori %26, %8 : i32 loc(#loc82)
    %29 = arith.select %27, %26, %28 : i32 loc(#loc82)
    %30 = arith.shrui %afterValue_8, %2 : i32 loc(#loc78)
    %31 = arith.andi %falseResult_7, %6 : i32 loc(#loc68)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %31, %40 : index, i32 loc(#loc68)
    %32:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc49)
    %33:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc49)
    %34 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc49)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc56)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %32#1 : none loc(#loc56)
    %36 = handshake.constant %1 {value = 0 : index} : index loc(#loc56)
    %37 = handshake.constant %1 {value = 1 : index} : index loc(#loc56)
    %38 = arith.select %9, %37, %36 : index loc(#loc56)
    %39 = handshake.mux %38 [%falseResult_14, %trueResult] : index, none loc(#loc56)
    %40 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc56)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %34 : none loc(#loc56)
    %41 = handshake.mux %38 [%falseResult_16, %trueResult] : index, none loc(#loc56)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc56)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %33#1 : none loc(#loc56)
    %43 = handshake.mux %38 [%falseResult_18, %trueResult] : index, none loc(#loc56)
    %44 = handshake.join %39, %41, %43 : none, none, none loc(#loc49)
    %45 = handshake.constant %44 {value = true} : i1 loc(#loc49)
    handshake.return %45 : i1 loc(#loc49)
  } loc(#loc49)
  handshake.func @_Z10gf_mul_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg3: i32 loc(fused<#di_subprogram5>[#loc14]), %arg4: none loc(fused<#di_subprogram5>[#loc14]), ...) -> none attributes {argNames = ["input_A", "input_B", "output_C", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc49)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 8 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %0 {value = 255 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %0 {value = 128 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %0 {value = 27 : i32} : i32 loc(#loc2)
    %8 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc59)
    %trueResult, %falseResult = handshake.cond_br %8, %0 : none loc(#loc56)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc56)
    %10 = arith.index_cast %4 : i64 to index loc(#loc56)
    %11 = arith.index_cast %arg3 : i32 to index loc(#loc56)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc56)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc56)
    %12 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc56)
    %dataResult, %addressResults = handshake.load [%afterValue] %31#0, %34 : index, i32 loc(#loc66)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %32#0, %41 : index, i32 loc(#loc67)
    %13 = arith.andi %dataResult_0, %5 : i32 loc(#loc67)
    %14 = handshake.constant %12 {value = 1 : index} : index loc(#loc70)
    %15 = arith.index_cast %2 : i32 to index loc(#loc70)
    %16 = arith.index_cast %3 : i32 to index loc(#loc70)
    %index_2, %willContinue_3 = dataflow.stream %15, %14, %16 {step_op = "+=", stop_cond = "!="} loc(#loc70)
    %17 = dataflow.carry %willContinue_3, %2, %23 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_4, %afterCond_5 = dataflow.gate %17, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_5 : i1 loc(#loc70)
    %trueResult_6, %falseResult_7 = handshake.cond_br %willContinue_3, %17 : i32 loc(#loc70)
    %18 = dataflow.carry %willContinue_3, %13, %29 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_8, %afterCond_9 = dataflow.gate %18, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_9 : i1 loc(#loc70)
    %19 = dataflow.carry %willContinue_3, %dataResult, %28 : i1, i32, i32 -> i32 loc(#loc70)
    %afterValue_10, %afterCond_11 = dataflow.gate %19, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc70)
    handshake.sink %afterCond_11 : i1 loc(#loc70)
    %20 = arith.andi %afterValue_8, %1 : i32 loc(#loc81)
    %21 = arith.cmpi eq, %20, %2 : i32 loc(#loc81)
    %22 = arith.select %21, %2, %afterValue_10 : i32 loc(#loc81)
    %23 = arith.xori %22, %afterValue_4 : i32 loc(#loc81)
    %24 = arith.andi %afterValue_10, %6 : i32 loc(#loc76)
    %25 = arith.shli %afterValue_10, %1 : i32 loc(#loc77)
    %26 = arith.cmpi eq, %24, %2 : i32 loc(#loc82)
    %27 = arith.xori %25, %7 : i32 loc(#loc82)
    %28 = arith.select %26, %25, %27 : i32 loc(#loc82)
    %29 = arith.shrui %afterValue_8, %1 : i32 loc(#loc78)
    %30 = arith.andi %falseResult_7, %5 : i32 loc(#loc68)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %30, %39 : index, i32 loc(#loc68)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc49)
    %32:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc49)
    %33 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc49)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc56)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %31#1 : none loc(#loc56)
    %35 = handshake.constant %0 {value = 0 : index} : index loc(#loc56)
    %36 = handshake.constant %0 {value = 1 : index} : index loc(#loc56)
    %37 = arith.select %8, %36, %35 : index loc(#loc56)
    %38 = handshake.mux %37 [%falseResult_14, %trueResult] : index, none loc(#loc56)
    %39 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc56)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %33 : none loc(#loc56)
    %40 = handshake.mux %37 [%falseResult_16, %trueResult] : index, none loc(#loc56)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc56)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %32#1 : none loc(#loc56)
    %42 = handshake.mux %37 [%falseResult_18, %trueResult] : index, none loc(#loc56)
    %43 = handshake.join %38, %40, %42 : none, none, none loc(#loc49)
    handshake.return %43 : none loc(#loc50)
  } loc(#loc49)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc38)
    %false = arith.constant false loc(#loc38)
    %0 = seq.const_clock  low loc(#loc38)
    %c2_i32 = arith.constant 2 : i32 loc(#loc38)
    %1 = ub.poison : i64 loc(#loc38)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<256xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc51)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc51)
      memref.store %11, %alloca[%10] : memref<256xi32> loc(#loc51)
      %12 = arith.muli %11, %c3_i32 : i32 loc(#loc52)
      %13 = arith.addi %12, %c7_i32 : i32 loc(#loc52)
      %14 = arith.andi %13, %c255_i32 : i32 loc(#loc52)
      memref.store %14, %alloca_0[%10] : memref<256xi32> loc(#loc52)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
      %16 = arith.cmpi ne, %15, %c256_i64 : i64 loc(#loc53)
      scf.condition(%16) %15 : i64 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block14>[#loc27])):
      scf.yield %arg0 : i64 loc(#loc43)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %cast_3 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %cast_4 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    call @_Z10gf_mul_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc39)
    %cast_5 = memref.cast %alloca_2 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc40)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc40)
    %5 = handshake.esi_instance @_Z10gf_mul_dsaPKjS0_Pjj_esi "_Z10gf_mul_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc40)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc40)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %11 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc57)
      %12 = memref.load %alloca_2[%10] : memref<256xi32> loc(#loc57)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc57)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc46)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc46)
        %18 = arith.extui %17 : i1 to i32 loc(#loc44)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc54)
        %20 = arith.extui %19 : i1 to i32 loc(#loc44)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc44)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc44)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc32]), %arg1: i1 loc(fused<#di_lexical_block15>[#loc32]), %arg2: i32 loc(fused<#di_lexical_block15>[#loc32])):
      scf.yield %arg0 : i64 loc(#loc44)
    } loc(#loc44)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc44)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc44)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc60)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc60)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc60)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc60)
      scf.yield %c1_i32 : i32 loc(#loc61)
    } loc(#loc44)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc41)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc41)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc41)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc41)
    } loc(#loc2)
    return %9 : i32 loc(#loc42)
  } loc(#loc38)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/gf_mul/gf_mul.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/gf_mul/gf_mul.cpp":18:0)
#loc5 = loc("tests/app/gf_mul/gf_mul.cpp":19:0)
#loc7 = loc("tests/app/gf_mul/gf_mul.cpp":23:0)
#loc8 = loc("tests/app/gf_mul/gf_mul.cpp":26:0)
#loc9 = loc("tests/app/gf_mul/gf_mul.cpp":27:0)
#loc10 = loc("tests/app/gf_mul/gf_mul.cpp":28:0)
#loc11 = loc("tests/app/gf_mul/gf_mul.cpp":31:0)
#loc12 = loc("tests/app/gf_mul/gf_mul.cpp":34:0)
#loc13 = loc("tests/app/gf_mul/gf_mul.cpp":36:0)
#loc15 = loc("tests/app/gf_mul/gf_mul.cpp":46:0)
#loc16 = loc("tests/app/gf_mul/gf_mul.cpp":47:0)
#loc17 = loc("tests/app/gf_mul/gf_mul.cpp":48:0)
#loc18 = loc("tests/app/gf_mul/gf_mul.cpp":51:0)
#loc19 = loc("tests/app/gf_mul/gf_mul.cpp":52:0)
#loc20 = loc("tests/app/gf_mul/gf_mul.cpp":55:0)
#loc21 = loc("tests/app/gf_mul/gf_mul.cpp":56:0)
#loc22 = loc("tests/app/gf_mul/gf_mul.cpp":57:0)
#loc23 = loc("tests/app/gf_mul/gf_mul.cpp":60:0)
#loc24 = loc("tests/app/gf_mul/gf_mul.cpp":63:0)
#loc25 = loc("tests/app/gf_mul/gf_mul.cpp":65:0)
#loc26 = loc("tests/app/gf_mul/main.cpp":5:0)
#loc28 = loc("tests/app/gf_mul/main.cpp":18:0)
#loc29 = loc("tests/app/gf_mul/main.cpp":19:0)
#loc30 = loc("tests/app/gf_mul/main.cpp":23:0)
#loc31 = loc("tests/app/gf_mul/main.cpp":26:0)
#loc33 = loc("tests/app/gf_mul/main.cpp":30:0)
#loc34 = loc("tests/app/gf_mul/main.cpp":31:0)
#loc35 = loc("tests/app/gf_mul/main.cpp":32:0)
#loc36 = loc("tests/app/gf_mul/main.cpp":36:0)
#loc37 = loc("tests/app/gf_mul/main.cpp":38:0)
#loc38 = loc(fused<#di_subprogram3>[#loc26])
#loc39 = loc(fused<#di_subprogram3>[#loc30])
#loc40 = loc(fused<#di_subprogram3>[#loc31])
#loc41 = loc(fused<#di_subprogram3>[#loc36])
#loc42 = loc(fused<#di_subprogram3>[#loc37])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 17>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 29>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 17>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 29>
#loc45 = loc(fused<#di_lexical_block16>[#loc27])
#loc46 = loc(fused<#di_lexical_block17>[#loc32])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 46>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 30>
#loc48 = loc(fused<#di_subprogram4>[#loc13])
#loc50 = loc(fused<#di_subprogram5>[#loc25])
#loc51 = loc(fused<#di_lexical_block18>[#loc28])
#loc52 = loc(fused<#di_lexical_block18>[#loc29])
#loc53 = loc(fused[#loc43, #loc45])
#loc54 = loc(fused[#loc44, #loc46])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 46>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 30>
#loc56 = loc(fused<#di_lexical_block21>[#loc15])
#loc57 = loc(fused<#di_lexical_block22>[#loc33])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 46>
#loc58 = loc(fused<#di_lexical_block23>[#loc3])
#loc59 = loc(fused<#di_lexical_block24>[#loc15])
#loc60 = loc(fused<#di_lexical_block25>[#loc34])
#loc61 = loc(fused<#di_lexical_block25>[#loc35])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 51>
#loc62 = loc(fused<#di_lexical_block26>[#loc4])
#loc63 = loc(fused<#di_lexical_block26>[#loc5])
#loc64 = loc(fused<#di_lexical_block26>[#loc12])
#loc65 = loc(fused[#loc55, #loc58])
#loc66 = loc(fused<#di_lexical_block27>[#loc16])
#loc67 = loc(fused<#di_lexical_block27>[#loc17])
#loc68 = loc(fused<#di_lexical_block27>[#loc24])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 22>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 51>
#loc70 = loc(fused<#di_lexical_block29>[#loc18])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 22>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 51>
#loc71 = loc(fused<#di_lexical_block30>[#loc6])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 23>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 28>
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 52>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 57>
#loc72 = loc(fused<#di_lexical_block32>[#loc8])
#loc73 = loc(fused<#di_lexical_block32>[#loc9])
#loc74 = loc(fused<#di_lexical_block32>[#loc11])
#loc75 = loc(fused[#loc69, #loc71])
#loc76 = loc(fused<#di_lexical_block33>[#loc20])
#loc77 = loc(fused<#di_lexical_block33>[#loc21])
#loc78 = loc(fused<#di_lexical_block33>[#loc23])
#loc79 = loc(fused<#di_lexical_block34>[#loc7])
#loc80 = loc(fused<#di_lexical_block35>[#loc10])
#loc81 = loc(fused<#di_lexical_block36>[#loc19])
#loc82 = loc(fused<#di_lexical_block37>[#loc22])
