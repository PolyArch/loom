#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/convolve_1d/convolve_1d.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/convolve_1d/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/convolve_1d/convolve_1d.cpp":16:0)
#loc4 = loc("tests/app/convolve_1d/convolve_1d.cpp":23:0)
#loc5 = loc("tests/app/convolve_1d/convolve_1d.cpp":25:0)
#loc9 = loc("tests/app/convolve_1d/convolve_1d.cpp":34:0)
#loc17 = loc("tests/app/convolve_1d/main.cpp":22:0)
#loc19 = loc("tests/app/convolve_1d/main.cpp":27:0)
#loc23 = loc("tests/app/convolve_1d/main.cpp":35:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 4096, elements = #llvm.di_subrange<count = 128 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 224, elements = #llvm.di_subrange<count = 7 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 3904, elements = #llvm.di_subrange<count = 122 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 23>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 43>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 22>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 27>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 35>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 23>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 43>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "kernel", file = #di_file1, line = 15, type = #di_composite_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 18, type = #di_composite_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 19, type = #di_composite_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 23>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 43>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 18, arg = 3, type = #di_derived_type4>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_size", file = #di_file, line = 21, type = #di_derived_type5>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block, name = "n", file = #di_file, line = 23, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 36, arg = 3, type = #di_derived_type4>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_size", file = #di_file, line = 39, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "n", file = #di_file, line = 43, type = #di_derived_type5>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 22, type = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 27, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 35, type = #di_derived_type5>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 25>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 45>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 16, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "kernel", file = #di_file, line = 17, arg = 2, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_size", file = #di_file, line = 19, arg = 4, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "kernel_size", file = #di_file, line = 20, arg = 5, type = #di_derived_type7>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "sum", file = #di_file, line = 24, type = #di_basic_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 34, arg = 1, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "kernel", file = #di_file, line = 35, arg = 2, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_size", file = #di_file, line = 37, arg = 4, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "kernel_size", file = #di_file, line = 38, arg = 5, type = #di_derived_type7>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "sum", file = #di_file, line = 44, type = #di_basic_type>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_size", file = #di_file1, line = 7, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "kernel_size", file = #di_file1, line = 8, type = #di_derived_type7>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_size", file = #di_file1, line = 9, type = #di_derived_type7>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_derived_type7, #di_derived_type7>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "k", file = #di_file, line = 25, type = #di_derived_type5>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "k", file = #di_file, line = 45, type = #di_derived_type5>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable23, #di_local_variable24, #di_local_variable25, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable10, #di_local_variable11, #di_local_variable12>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 22>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 27>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 35>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "convolve_1d_cpu", linkageName = "_Z15convolve_1d_cpuPKfS0_Pfjj", file = #di_file, line = 16, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable4, #di_local_variable15, #di_local_variable16, #di_local_variable5, #di_local_variable6, #di_local_variable17, #di_local_variable26>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "convolve_1d_dsa", linkageName = "_Z15convolve_1d_dsaPKfS0_Pfjj", file = #di_file, line = 34, scopeLine = 38, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable18, #di_local_variable19, #di_local_variable7, #di_local_variable20, #di_local_variable21, #di_local_variable8, #di_local_variable9, #di_local_variable22, #di_local_variable27>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 23>
#loc36 = loc(fused<#di_subprogram5>[#loc1])
#loc39 = loc(fused<#di_subprogram6>[#loc9])
#loc42 = loc(fused<#di_lexical_block11>[#loc17])
#loc43 = loc(fused<#di_lexical_block12>[#loc19])
#loc44 = loc(fused<#di_lexical_block13>[#loc23])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 23>
#loc45 = loc(fused<#di_lexical_block14>[#loc4])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 23>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 25>
#loc61 = loc(fused<#di_lexical_block27>[#loc5])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 47, 99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<20xi8> = dense<[99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z15convolve_1d_cpuPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: i32 loc(fused<#di_subprogram5>[#loc1]), %arg4: i32 loc(fused<#di_subprogram5>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c4294967295_i64 = arith.constant 4294967295 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.addi %arg3, %c1_i32 : i32 loc(#loc37)
    %1 = arith.cmpi eq, %0, %arg4 : i32 loc(#loc50)
    scf.if %1 {
    } else {
      %2 = arith.subi %0, %arg4 : i32 loc(#loc37)
      %3 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %4 = arith.extui %2 : i32 to i64 loc(#loc50)
      %5 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %6 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %7 = scf.if %3 -> (f32) {
          scf.yield %cst : f32 loc(#loc61)
        } else {
          %11:2 = scf.while (%arg6 = %c0_i64, %arg7 = %cst) : (i64, f32) -> (i64, f32) {
            %12 = arith.addi %arg6, %arg5 : i64 loc(#loc66)
            %13 = arith.andi %12, %c4294967295_i64 : i64 loc(#loc66)
            %14 = arith.index_cast %13 : i64 to index loc(#loc66)
            %15 = memref.load %arg0[%14] : memref<?xf32, strided<[1], offset: ?>> loc(#loc66)
            %16 = arith.index_cast %arg6 : i64 to index loc(#loc66)
            %17 = memref.load %arg1[%16] : memref<?xf32, strided<[1], offset: ?>> loc(#loc66)
            %18 = math.fma %15, %17, %arg7 : f32 loc(#loc66)
            %19 = arith.addi %arg6, %c1_i64 : i64 loc(#loc65)
            %20 = arith.cmpi ne, %19, %5 : i64 loc(#loc67)
            scf.condition(%20) %19, %18 : i64, f32 loc(#loc61)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block27>[#loc5]), %arg7: f32 loc(fused<#di_lexical_block27>[#loc5])):
            scf.yield %arg6, %arg7 : i64, f32 loc(#loc61)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc61)
          scf.yield %11#1 : f32 loc(#loc61)
        } loc(#loc61)
        %8 = arith.index_cast %arg5 : i64 to index loc(#loc57)
        memref.store %7, %arg2[%8] : memref<?xf32, strided<[1], offset: ?>> loc(#loc57)
        %9 = arith.addi %arg5, %c1_i64 : i64 loc(#loc50)
        %10 = arith.cmpi ne, %9, %4 : i64 loc(#loc58)
        scf.condition(%10) %9 : i64 loc(#loc45)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block14>[#loc4])):
        scf.yield %arg5 : i64 loc(#loc45)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc45)
    } loc(#loc45)
    return loc(#loc38)
  } loc(#loc36)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  handshake.func @_Z15convolve_1d_dsaPKfS0_Pfjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg3: i32 loc(fused<#di_subprogram6>[#loc9]), %arg4: i32 loc(fused<#di_subprogram6>[#loc9]), %arg5: i1 loc(fused<#di_subprogram6>[#loc9]), ...) -> i1 attributes {argNames = ["input", "kernel", "output", "input_size", "kernel_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc39)
    %1 = handshake.join %0 : none loc(#loc39)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %6 = handshake.constant %1 {value = 4294967295 : i64} : i64 loc(#loc2)
    %7 = arith.addi %arg3, %2 : i32 loc(#loc40)
    %8 = arith.cmpi eq, %7, %arg4 : i32 loc(#loc51)
    %trueResult, %falseResult = handshake.cond_br %8, %1 : none loc(#loc46)
    %9 = arith.subi %7, %arg4 : i32 loc(#loc40)
    %10 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc2)
    %11 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc46)
    %12 = arith.index_cast %4 : i64 to index loc(#loc46)
    %13 = arith.index_cast %9 : i32 to index loc(#loc46)
    %index, %willContinue = dataflow.stream %12, %11, %13 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc46)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc46)
    %14 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc46)
    %15 = arith.index_cast %afterValue : index to i64 loc(#loc46)
    %16 = dataflow.invariant %afterCond, %10 : i1, i1 -> i1 loc(#loc62)
    %trueResult_0, %falseResult_1 = handshake.cond_br %16, %14 : none loc(#loc62)
    %17 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc62)
    %18 = arith.index_cast %arg4 : i32 to index loc(#loc62)
    %index_2, %willContinue_3 = dataflow.stream %12, %17, %18 {step_op = "+=", stop_cond = "!="} loc(#loc62)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc62)
    %19 = dataflow.carry %willContinue_3, %5, %24 : i1, f32, f32 -> f32 loc(#loc62)
    %afterValue_6, %afterCond_7 = dataflow.gate %19, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc62)
    handshake.sink %afterCond_7 : i1 loc(#loc62)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %19 : f32 loc(#loc62)
    %20 = arith.index_cast %afterValue_4 : index to i64 loc(#loc62)
    %21 = arith.addi %20, %15 : i64 loc(#loc68)
    %22 = arith.andi %21, %6 : i64 loc(#loc68)
    %23 = arith.index_cast %22 : i64 to index loc(#loc68)
    %dataResult, %addressResults = handshake.load [%23] %30#0, %33 : index, f32 loc(#loc68)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %29#0, %45 : index, f32 loc(#loc68)
    %24 = math.fma %dataResult, %dataResult_10, %afterValue_6 : f32 loc(#loc68)
    %25 = handshake.constant %14 {value = 0 : index} : index loc(#loc62)
    %26 = handshake.constant %14 {value = 1 : index} : index loc(#loc62)
    %27 = arith.select %16, %26, %25 : index loc(#loc62)
    %28 = handshake.mux %27 [%falseResult_9, %5] : index, f32 loc(#loc62)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %28, %42 : index, f32 loc(#loc59)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (f32, none) loc(#loc39)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc39)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc39)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc46)
    %trueResult_13, %falseResult_14 = handshake.cond_br %16, %32 : none loc(#loc62)
    %33 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc62)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %30#1 : none loc(#loc62)
    %34 = handshake.constant %32 {value = 0 : index} : index loc(#loc62)
    %35 = handshake.constant %32 {value = 1 : index} : index loc(#loc62)
    %36 = arith.select %16, %35, %34 : index loc(#loc62)
    %37 = handshake.mux %36 [%falseResult_16, %trueResult_13] : index, none loc(#loc62)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %37 : none loc(#loc46)
    %38 = handshake.constant %1 {value = 0 : index} : index loc(#loc46)
    %39 = handshake.constant %1 {value = 1 : index} : index loc(#loc46)
    %40 = arith.select %8, %39, %38 : index loc(#loc46)
    %41 = handshake.mux %40 [%falseResult_18, %trueResult] : index, none loc(#loc46)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc46)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %31 : none loc(#loc46)
    %43 = handshake.mux %40 [%falseResult_20, %trueResult] : index, none loc(#loc46)
    %44 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc46)
    %trueResult_21, %falseResult_22 = handshake.cond_br %16, %44 : none loc(#loc62)
    %45 = dataflow.carry %willContinue_3, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc62)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_3, %29#1 : none loc(#loc62)
    %46 = handshake.constant %44 {value = 0 : index} : index loc(#loc62)
    %47 = handshake.constant %44 {value = 1 : index} : index loc(#loc62)
    %48 = arith.select %16, %47, %46 : index loc(#loc62)
    %49 = handshake.mux %48 [%falseResult_24, %trueResult_21] : index, none loc(#loc62)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %49 : none loc(#loc46)
    %50 = handshake.mux %40 [%falseResult_26, %trueResult] : index, none loc(#loc46)
    %51 = handshake.join %41, %43, %50 : none, none, none loc(#loc39)
    %52 = handshake.constant %51 {value = true} : i1 loc(#loc39)
    handshake.return %52 : i1 loc(#loc39)
  } loc(#loc39)
  handshake.func @_Z15convolve_1d_dsaPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc9]), %arg3: i32 loc(fused<#di_subprogram6>[#loc9]), %arg4: i32 loc(fused<#di_subprogram6>[#loc9]), %arg5: none loc(fused<#di_subprogram6>[#loc9]), ...) -> none attributes {argNames = ["input", "kernel", "output", "input_size", "kernel_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc39)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %5 = handshake.constant %0 {value = 4294967295 : i64} : i64 loc(#loc2)
    %6 = arith.addi %arg3, %1 : i32 loc(#loc40)
    %7 = arith.cmpi eq, %6, %arg4 : i32 loc(#loc51)
    %trueResult, %falseResult = handshake.cond_br %7, %0 : none loc(#loc46)
    %8 = arith.subi %6, %arg4 : i32 loc(#loc40)
    %9 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc46)
    %11 = arith.index_cast %3 : i64 to index loc(#loc46)
    %12 = arith.index_cast %8 : i32 to index loc(#loc46)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc46)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc46)
    %13 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc46)
    %14 = arith.index_cast %afterValue : index to i64 loc(#loc46)
    %15 = dataflow.invariant %afterCond, %9 : i1, i1 -> i1 loc(#loc62)
    %trueResult_0, %falseResult_1 = handshake.cond_br %15, %13 : none loc(#loc62)
    %16 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc62)
    %17 = arith.index_cast %arg4 : i32 to index loc(#loc62)
    %index_2, %willContinue_3 = dataflow.stream %11, %16, %17 {step_op = "+=", stop_cond = "!="} loc(#loc62)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc62)
    %18 = dataflow.carry %willContinue_3, %4, %23 : i1, f32, f32 -> f32 loc(#loc62)
    %afterValue_6, %afterCond_7 = dataflow.gate %18, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc62)
    handshake.sink %afterCond_7 : i1 loc(#loc62)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %18 : f32 loc(#loc62)
    %19 = arith.index_cast %afterValue_4 : index to i64 loc(#loc62)
    %20 = arith.addi %19, %14 : i64 loc(#loc68)
    %21 = arith.andi %20, %5 : i64 loc(#loc68)
    %22 = arith.index_cast %21 : i64 to index loc(#loc68)
    %dataResult, %addressResults = handshake.load [%22] %29#0, %32 : index, f32 loc(#loc68)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %28#0, %44 : index, f32 loc(#loc68)
    %23 = math.fma %dataResult, %dataResult_10, %afterValue_6 : f32 loc(#loc68)
    %24 = handshake.constant %13 {value = 0 : index} : index loc(#loc62)
    %25 = handshake.constant %13 {value = 1 : index} : index loc(#loc62)
    %26 = arith.select %15, %25, %24 : index loc(#loc62)
    %27 = handshake.mux %26 [%falseResult_9, %4] : index, f32 loc(#loc62)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %27, %41 : index, f32 loc(#loc59)
    %28:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (f32, none) loc(#loc39)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc39)
    %30 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc39)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc46)
    %trueResult_13, %falseResult_14 = handshake.cond_br %15, %31 : none loc(#loc62)
    %32 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc62)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %29#1 : none loc(#loc62)
    %33 = handshake.constant %31 {value = 0 : index} : index loc(#loc62)
    %34 = handshake.constant %31 {value = 1 : index} : index loc(#loc62)
    %35 = arith.select %15, %34, %33 : index loc(#loc62)
    %36 = handshake.mux %35 [%falseResult_16, %trueResult_13] : index, none loc(#loc62)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %36 : none loc(#loc46)
    %37 = handshake.constant %0 {value = 0 : index} : index loc(#loc46)
    %38 = handshake.constant %0 {value = 1 : index} : index loc(#loc46)
    %39 = arith.select %7, %38, %37 : index loc(#loc46)
    %40 = handshake.mux %39 [%falseResult_18, %trueResult] : index, none loc(#loc46)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc46)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %30 : none loc(#loc46)
    %42 = handshake.mux %39 [%falseResult_20, %trueResult] : index, none loc(#loc46)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc46)
    %trueResult_21, %falseResult_22 = handshake.cond_br %15, %43 : none loc(#loc62)
    %44 = dataflow.carry %willContinue_3, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc62)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_3, %28#1 : none loc(#loc62)
    %45 = handshake.constant %43 {value = 0 : index} : index loc(#loc62)
    %46 = handshake.constant %43 {value = 1 : index} : index loc(#loc62)
    %47 = arith.select %15, %46, %45 : index loc(#loc62)
    %48 = handshake.mux %47 [%falseResult_24, %trueResult_21] : index, none loc(#loc62)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %48 : none loc(#loc46)
    %49 = handshake.mux %39 [%falseResult_26, %trueResult] : index, none loc(#loc46)
    %50 = handshake.join %40, %42, %49 : none, none, none loc(#loc39)
    handshake.return %50 : none loc(#loc41)
  } loc(#loc39)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc31)
    %false = arith.constant false loc(#loc31)
    %0 = seq.const_clock  low loc(#loc31)
    %c2_i32 = arith.constant 2 : i32 loc(#loc31)
    %1 = ub.poison : i64 loc(#loc31)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c122_i64 = arith.constant 122 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 6.283180e+00 : f32 loc(#loc2)
    %cst_0 = arith.constant 3.125000e-02 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c128_i64 = arith.constant 128 : i64 loc(#loc2)
    %cst_1 = arith.constant 0.142857149 : f32 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c128_i32 = arith.constant 128 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %cst_2 = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<7xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<122xf32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<122xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc52)
      %12 = arith.uitofp %11 : i32 to f32 loc(#loc52)
      %13 = arith.mulf %12, %cst : f32 loc(#loc52)
      %14 = arith.mulf %13, %cst_0 : f32 loc(#loc52)
      %15 = math.sin %14 : f32 loc(#loc52)
      %16 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      memref.store %15, %alloca[%16] : memref<128xf32> loc(#loc52)
      %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc47)
      %18 = arith.cmpi ne, %17, %c128_i64 : i64 loc(#loc53)
      scf.condition(%18) %17 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc17])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc54)
      memref.store %cst_1, %alloca_3[%11] : memref<7xf32> loc(#loc54)
      %12 = arith.addi %arg0, %c1_i64 : i64 loc(#loc48)
      %13 = arith.cmpi ne, %12, %c7_i64 : i64 loc(#loc55)
      scf.condition(%13) %12 : i64 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc19])):
      scf.yield %arg0 : i64 loc(#loc43)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    %cast = memref.cast %alloca : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    %cast_6 = memref.cast %alloca_3 : memref<7xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    %cast_7 = memref.cast %alloca_4 : memref<122xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc32)
    call @_Z15convolve_1d_cpuPKfS0_Pfjj(%cast, %cast_6, %cast_7, %c128_i32, %c7_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc32)
    %cast_8 = memref.cast %alloca_5 : memref<122xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_9, %ready_10 = esi.wrap.vr %cast_6, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_11, %ready_12 = esi.wrap.vr %cast_8, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_13, %ready_14 = esi.wrap.vr %c128_i32, %true : i32 loc(#loc33)
    %chanOutput_15, %ready_16 = esi.wrap.vr %c7_i32, %true : i32 loc(#loc33)
    %chanOutput_17, %ready_18 = esi.wrap.vr %true, %true : i1 loc(#loc33)
    %6 = handshake.esi_instance @_Z15convolve_1d_dsaPKfS0_Pfjj_esi "_Z15convolve_1d_dsaPKfS0_Pfjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_9, %chanOutput_11, %chanOutput_13, %chanOutput_15, %chanOutput_17) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc33)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc33)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc60)
      %12 = memref.load %alloca_4[%11] : memref<122xf32> loc(#loc60)
      %13 = memref.load %alloca_5[%11] : memref<122xf32> loc(#loc60)
      %14 = arith.subf %12, %13 : f32 loc(#loc60)
      %15 = math.absf %14 : f32 loc(#loc60)
      %16 = arith.cmpf ule, %15, %cst_2 : f32 loc(#loc60)
      %17:3 = scf.if %16 -> (i64, i32, i32) {
        %19 = arith.addi %arg0, %c1_i64 : i64 loc(#loc49)
        %20 = arith.cmpi eq, %19, %c122_i64 : i64 loc(#loc49)
        %21 = arith.extui %20 : i1 to i32 loc(#loc44)
        %22 = arith.cmpi ne, %19, %c122_i64 : i64 loc(#loc56)
        %23 = arith.extui %22 : i1 to i32 loc(#loc44)
        scf.yield %19, %21, %23 : i64, i32, i32 loc(#loc60)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc60)
      } loc(#loc60)
      %18 = arith.trunci %17#2 : i32 to i1 loc(#loc44)
      scf.condition(%18) %17#0, %16, %17#1 : i64, i1, i32 loc(#loc44)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc23]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc23]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc23])):
      scf.yield %arg0 : i64 loc(#loc44)
    } loc(#loc44)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc44)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc44)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc63)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc63)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc63)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc63)
      scf.yield %c1_i32 : i32 loc(#loc64)
    } loc(#loc44)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc34)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc34)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc34)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc34)
    } loc(#loc2)
    return %10 : i32 loc(#loc35)
  } loc(#loc31)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @sinf(f32) -> f32 loc(#loc30)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#di_file2 = #llvm.di_file<"/usr/include/bits/mathcalls.h" in "">
#loc = loc("tests/app/convolve_1d/convolve_1d.cpp":0:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/convolve_1d/convolve_1d.cpp":21:0)
#loc6 = loc("tests/app/convolve_1d/convolve_1d.cpp":26:0)
#loc7 = loc("tests/app/convolve_1d/convolve_1d.cpp":28:0)
#loc8 = loc("tests/app/convolve_1d/convolve_1d.cpp":30:0)
#loc10 = loc("tests/app/convolve_1d/convolve_1d.cpp":39:0)
#loc11 = loc("tests/app/convolve_1d/convolve_1d.cpp":43:0)
#loc12 = loc("tests/app/convolve_1d/convolve_1d.cpp":45:0)
#loc13 = loc("tests/app/convolve_1d/convolve_1d.cpp":46:0)
#loc14 = loc("tests/app/convolve_1d/convolve_1d.cpp":48:0)
#loc15 = loc("tests/app/convolve_1d/convolve_1d.cpp":50:0)
#loc16 = loc("tests/app/convolve_1d/main.cpp":6:0)
#loc18 = loc("tests/app/convolve_1d/main.cpp":23:0)
#loc20 = loc("tests/app/convolve_1d/main.cpp":28:0)
#loc21 = loc("tests/app/convolve_1d/main.cpp":32:0)
#loc22 = loc("tests/app/convolve_1d/main.cpp":33:0)
#loc24 = loc("tests/app/convolve_1d/main.cpp":36:0)
#loc25 = loc("tests/app/convolve_1d/main.cpp":37:0)
#loc26 = loc("tests/app/convolve_1d/main.cpp":38:0)
#loc27 = loc("tests/app/convolve_1d/main.cpp":42:0)
#loc28 = loc("tests/app/convolve_1d/main.cpp":44:0)
#loc29 = loc("/usr/include/bits/mathcalls.h":64:0)
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type, #di_basic_type>
#di_subprogram3 = #llvm.di_subprogram<scope = #di_file2, name = "sinf", file = #di_file2, line = 64, subprogramFlags = Optimized, type = #di_subroutine_type1>
#loc30 = loc(fused<#di_subprogram3>[#loc29])
#loc31 = loc(fused<#di_subprogram4>[#loc16])
#loc32 = loc(fused<#di_subprogram4>[#loc21])
#loc33 = loc(fused<#di_subprogram4>[#loc22])
#loc34 = loc(fused<#di_subprogram4>[#loc27])
#loc35 = loc(fused<#di_subprogram4>[#loc28])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 43>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 22>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 27>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 35>
#loc37 = loc(fused<#di_subprogram5>[#loc3])
#loc38 = loc(fused<#di_subprogram5>[#loc8])
#loc40 = loc(fused<#di_subprogram6>[#loc10])
#loc41 = loc(fused<#di_subprogram6>[#loc15])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 43>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 22>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 27>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 35>
#loc46 = loc(fused<#di_lexical_block15>[#loc11])
#loc47 = loc(fused<#di_lexical_block16>[#loc17])
#loc48 = loc(fused<#di_lexical_block17>[#loc19])
#loc49 = loc(fused<#di_lexical_block18>[#loc23])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 43>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 36>
#loc50 = loc(fused<#di_lexical_block19>[#loc4])
#loc51 = loc(fused<#di_lexical_block20>[#loc11])
#loc52 = loc(fused<#di_lexical_block21>[#loc18])
#loc53 = loc(fused[#loc42, #loc47])
#loc54 = loc(fused<#di_lexical_block22>[#loc20])
#loc55 = loc(fused[#loc43, #loc48])
#loc56 = loc(fused[#loc44, #loc49])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 45>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 36>
#loc57 = loc(fused<#di_lexical_block24>[#loc7])
#loc58 = loc(fused[#loc45, #loc50])
#loc59 = loc(fused<#di_lexical_block25>[#loc14])
#loc60 = loc(fused<#di_lexical_block26>[#loc24])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 25>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 45>
#loc62 = loc(fused<#di_lexical_block28>[#loc12])
#loc63 = loc(fused<#di_lexical_block29>[#loc25])
#loc64 = loc(fused<#di_lexical_block29>[#loc26])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 25>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 45>
#loc65 = loc(fused<#di_lexical_block30>[#loc5])
#loc66 = loc(fused<#di_lexical_block32>[#loc6])
#loc67 = loc(fused[#loc61, #loc65])
#loc68 = loc(fused<#di_lexical_block33>[#loc13])
