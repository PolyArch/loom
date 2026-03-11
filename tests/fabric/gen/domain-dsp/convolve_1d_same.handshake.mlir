#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/convolve_1d_same/convolve_1d_same.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/convolve_1d_same/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":12:0)
#loc4 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":19:0)
#loc5 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":21:0)
#loc12 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":33:0)
#loc22 = loc("tests/app/convolve_1d_same/main.cpp":21:0)
#loc24 = loc("tests/app/convolve_1d_same/main.cpp":26:0)
#loc28 = loc("tests/app/convolve_1d_same/main.cpp":34:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 4096, elements = #llvm.di_subrange<count = 128 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 224, elements = #llvm.di_subrange<count = 7 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__int32_t", baseType = #di_basic_type2>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 19>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 42>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 21>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 26>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 34>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "int32_t", baseType = #di_derived_type3>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 19>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 42>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 11, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "kernel", file = #di_file1, line = 14, type = #di_composite_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 17, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 18, type = #di_composite_type>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type6>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 19>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 42>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 14, arg = 3, type = #di_derived_type5>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "pad", file = #di_file, line = 17, type = #di_derived_type7>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block, name = "n", file = #di_file, line = 19, type = #di_derived_type6>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 35, arg = 3, type = #di_derived_type5>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "pad", file = #di_file, line = 38, type = #di_derived_type7>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "n", file = #di_file, line = 42, type = #di_derived_type6>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 21, type = #di_derived_type6>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 26, type = #di_derived_type6>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 34, type = #di_derived_type6>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 21>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 44>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 12, arg = 1, type = #di_derived_type8>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "kernel", file = #di_file, line = 13, arg = 2, type = #di_derived_type8>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_size", file = #di_file, line = 15, arg = 4, type = #di_derived_type9>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "kernel_size", file = #di_file, line = 16, arg = 5, type = #di_derived_type9>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "sum", file = #di_file, line = 20, type = #di_basic_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 33, arg = 1, type = #di_derived_type8>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "kernel", file = #di_file, line = 34, arg = 2, type = #di_derived_type8>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_size", file = #di_file, line = 36, arg = 4, type = #di_derived_type9>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "kernel_size", file = #di_file, line = 37, arg = 5, type = #di_derived_type9>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "sum", file = #di_file, line = 43, type = #di_basic_type>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_size", file = #di_file1, line = 7, type = #di_derived_type9>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "kernel_size", file = #di_file1, line = 8, type = #di_derived_type9>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type8, #di_derived_type8, #di_derived_type5, #di_derived_type9, #di_derived_type9>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 21>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 44>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "k", file = #di_file, line = 21, type = #di_derived_type6>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "k", file = #di_file, line = 44, type = #di_derived_type6>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable23, #di_local_variable24, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable10, #di_local_variable11, #di_local_variable12>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 21>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 44>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 21>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 26>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 34>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block13, name = "idx", file = #di_file, line = 22, type = #di_derived_type7>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block14, name = "idx", file = #di_file, line = 45, type = #di_derived_type7>
#loc41 = loc(fused<#di_lexical_block15>[#loc22])
#loc42 = loc(fused<#di_lexical_block16>[#loc24])
#loc43 = loc(fused<#di_lexical_block17>[#loc28])
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "convolve_1d_same_cpu", linkageName = "_Z20convolve_1d_same_cpuPKfS0_Pfjj", file = #di_file, line = 12, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable4, #di_local_variable15, #di_local_variable16, #di_local_variable5, #di_local_variable6, #di_local_variable17, #di_local_variable25, #di_local_variable27>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "convolve_1d_same_dsa", linkageName = "_Z20convolve_1d_same_dsaPKfS0_Pfjj", file = #di_file, line = 33, scopeLine = 37, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable18, #di_local_variable19, #di_local_variable7, #di_local_variable20, #di_local_variable21, #di_local_variable8, #di_local_variable9, #di_local_variable22, #di_local_variable26, #di_local_variable28>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 19>
#loc47 = loc(fused<#di_subprogram5>[#loc1])
#loc50 = loc(fused<#di_subprogram6>[#loc12])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 19>
#loc58 = loc(fused<#di_lexical_block24>[#loc4])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 19>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 21>
#loc68 = loc(fused<#di_lexical_block32>[#loc5])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<48xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 95, 115, 97, 109, 101, 47, 99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 95, 115, 97, 109, 101, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<25xi8> = dense<[99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 95, 115, 97, 109, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<25xi8> = dense<[99, 111, 110, 118, 111, 108, 118, 101, 95, 49, 100, 95, 115, 97, 109, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z20convolve_1d_same_cpuPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: i32 loc(fused<#di_subprogram5>[#loc1]), %arg4: i32 loc(fused<#di_subprogram5>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.shrui %arg4, %c1_i32 : i32 loc(#loc48)
    %1 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc61)
    scf.if %1 {
    } else {
      %2 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg3 : i32 to i64 loc(#loc61)
      %4 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %5 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %6 = scf.if %2 -> (f32) {
          scf.yield %cst : f32 loc(#loc68)
        } else {
          %10 = arith.trunci %arg5 : i64 to i32 loc(#loc2)
          %11 = arith.subi %10, %0 : i32 loc(#loc2)
          %12:2 = scf.while (%arg6 = %c0_i64, %arg7 = %cst) : (i64, f32) -> (i64, f32) {
            %13 = arith.trunci %arg6 : i64 to i32 loc(#loc71)
            %14 = arith.addi %11, %13 : i32 loc(#loc71)
            %15 = arith.cmpi sgt, %14, %c-1_i32 : i32 loc(#loc74)
            %16 = arith.cmpi slt, %14, %arg3 : i32 loc(#loc74)
            %17 = arith.andi %15, %16 : i1 loc(#loc74)
            %18 = scf.if %17 -> (f32) {
              %21 = arith.extui %14 : i32 to i64 loc(#loc76)
              %22 = arith.index_cast %21 : i64 to index loc(#loc76)
              %23 = memref.load %arg0[%22] : memref<?xf32, strided<[1], offset: ?>> loc(#loc76)
              %24 = arith.index_cast %arg6 : i64 to index loc(#loc76)
              %25 = memref.load %arg1[%24] : memref<?xf32, strided<[1], offset: ?>> loc(#loc76)
              %26 = math.fma %23, %25, %arg7 : f32 loc(#loc76)
              scf.yield %26 : f32 loc(#loc77)
            } else {
              scf.yield %arg7 : f32 loc(#loc74)
            } loc(#loc74)
            %19 = arith.addi %arg6, %c1_i64 : i64 loc(#loc70)
            %20 = arith.cmpi ne, %19, %4 : i64 loc(#loc72)
            scf.condition(%20) %19, %18 : i64, f32 loc(#loc68)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block32>[#loc5]), %arg7: f32 loc(fused<#di_lexical_block32>[#loc5])):
            scf.yield %arg6, %arg7 : i64, f32 loc(#loc68)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc68)
          scf.yield %12#1 : f32 loc(#loc68)
        } loc(#loc68)
        %7 = arith.index_cast %arg5 : i64 to index loc(#loc65)
        memref.store %6, %arg2[%7] : memref<?xf32, strided<[1], offset: ?>> loc(#loc65)
        %8 = arith.addi %arg5, %c1_i64 : i64 loc(#loc61)
        %9 = arith.cmpi ne, %8, %3 : i64 loc(#loc66)
        scf.condition(%9) %8 : i64 loc(#loc58)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block24>[#loc4])):
        scf.yield %arg5 : i64 loc(#loc58)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc58)
    } loc(#loc58)
    return loc(#loc49)
  } loc(#loc47)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  handshake.func @_Z20convolve_1d_same_dsaPKfS0_Pfjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: i32 loc(fused<#di_subprogram6>[#loc12]), %arg4: i32 loc(fused<#di_subprogram6>[#loc12]), %arg5: i1 loc(fused<#di_subprogram6>[#loc12]), ...) -> i1 attributes {argNames = ["input", "kernel", "output", "input_size", "kernel_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc50)
    %1 = handshake.join %0 : none loc(#loc50)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %6 = handshake.constant %1 {value = -1 : i32} : i32 loc(#loc2)
    %7 = arith.shrui %arg4, %2 : i32 loc(#loc51)
    %8 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc62)
    %trueResult, %falseResult = handshake.cond_br %8, %1 : none loc(#loc59)
    %9 = arith.cmpi eq, %arg4, %3 : i32 loc(#loc2)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc59)
    %11 = arith.index_cast %4 : i64 to index loc(#loc59)
    %12 = arith.index_cast %arg3 : i32 to index loc(#loc59)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc59)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc59)
    %13 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc59)
    %14 = arith.index_cast %afterValue : index to i64 loc(#loc59)
    %15 = dataflow.invariant %afterCond, %9 : i1, i1 -> i1 loc(#loc69)
    %trueResult_0, %falseResult_1 = handshake.cond_br %15, %13 : none loc(#loc69)
    %16 = arith.trunci %14 : i64 to i32 loc(#loc2)
    %17 = arith.subi %16, %7 : i32 loc(#loc2)
    %18 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc69)
    %19 = arith.index_cast %arg4 : i32 to index loc(#loc69)
    %index_2, %willContinue_3 = dataflow.stream %11, %18, %19 {step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc69)
    %20 = dataflow.carry %willContinue_3, %5, %35 : i1, f32, f32 -> f32 loc(#loc69)
    %afterValue_6, %afterCond_7 = dataflow.gate %20, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc69)
    handshake.sink %afterCond_7 : i1 loc(#loc69)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %20 : f32 loc(#loc69)
    %21 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc69)
    %22 = arith.index_cast %afterValue_4 : index to i64 loc(#loc69)
    %23 = arith.trunci %22 : i64 to i32 loc(#loc73)
    %24 = dataflow.invariant %afterCond_5, %17 : i1, i32 -> i32 loc(#loc73)
    %25 = arith.addi %24, %23 : i32 loc(#loc73)
    %26 = arith.cmpi sgt, %25, %6 : i32 loc(#loc75)
    %27 = arith.cmpi slt, %25, %arg3 : i32 loc(#loc75)
    %28 = arith.andi %26, %27 : i1 loc(#loc75)
    %29 = arith.extui %25 : i32 to i64 loc(#loc78)
    %30 = arith.index_cast %29 : i64 to index loc(#loc78)
    %dataResult, %addressResults = handshake.load [%30] %40#0, %trueResult_15 : index, f32 loc(#loc78)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %41#0, %trueResult_25 : index, f32 loc(#loc78)
    %31 = math.fma %dataResult, %dataResult_10, %afterValue_6 : f32 loc(#loc78)
    %32 = handshake.constant %21 {value = 0 : index} : index loc(#loc75)
    %33 = handshake.constant %21 {value = 1 : index} : index loc(#loc75)
    %34 = arith.select %28, %33, %32 : index loc(#loc75)
    %35 = handshake.mux %34 [%afterValue_6, %31] : index, f32 loc(#loc75)
    %36 = handshake.constant %13 {value = 0 : index} : index loc(#loc69)
    %37 = handshake.constant %13 {value = 1 : index} : index loc(#loc69)
    %38 = arith.select %15, %37, %36 : index loc(#loc69)
    %39 = handshake.mux %38 [%falseResult_9, %5] : index, f32 loc(#loc69)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %39, %57 : index, f32 loc(#loc67)
    %40:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc50)
    %41:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 1 : i32} : (index) -> (f32, none) loc(#loc50)
    %42 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc50)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc59)
    %trueResult_13, %falseResult_14 = handshake.cond_br %15, %43 : none loc(#loc69)
    %44 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_17 : i1, none, none -> none loc(#loc69)
    %trueResult_15, %falseResult_16 = handshake.cond_br %28, %44 : none loc(#loc75)
    %45 = handshake.constant %44 {value = 0 : index} : index loc(#loc75)
    %46 = handshake.constant %44 {value = 1 : index} : index loc(#loc75)
    %47 = arith.select %28, %46, %45 : index loc(#loc75)
    %48 = handshake.mux %47 [%falseResult_16, %40#1] : index, none loc(#loc75)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_3, %48 : none loc(#loc69)
    %49 = handshake.constant %43 {value = 0 : index} : index loc(#loc69)
    %50 = handshake.constant %43 {value = 1 : index} : index loc(#loc69)
    %51 = arith.select %15, %50, %49 : index loc(#loc69)
    %52 = handshake.mux %51 [%falseResult_18, %trueResult_13] : index, none loc(#loc69)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %52 : none loc(#loc59)
    %53 = handshake.constant %1 {value = 0 : index} : index loc(#loc59)
    %54 = handshake.constant %1 {value = 1 : index} : index loc(#loc59)
    %55 = arith.select %8, %54, %53 : index loc(#loc59)
    %56 = handshake.mux %55 [%falseResult_20, %trueResult] : index, none loc(#loc59)
    %57 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc59)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %42 : none loc(#loc59)
    %58 = handshake.mux %55 [%falseResult_22, %trueResult] : index, none loc(#loc59)
    %59 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc59)
    %trueResult_23, %falseResult_24 = handshake.cond_br %15, %59 : none loc(#loc69)
    %60 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_27 : i1, none, none -> none loc(#loc69)
    %trueResult_25, %falseResult_26 = handshake.cond_br %28, %60 : none loc(#loc75)
    %61 = handshake.constant %60 {value = 0 : index} : index loc(#loc75)
    %62 = handshake.constant %60 {value = 1 : index} : index loc(#loc75)
    %63 = arith.select %28, %62, %61 : index loc(#loc75)
    %64 = handshake.mux %63 [%falseResult_26, %41#1] : index, none loc(#loc75)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_3, %64 : none loc(#loc69)
    %65 = handshake.constant %59 {value = 0 : index} : index loc(#loc69)
    %66 = handshake.constant %59 {value = 1 : index} : index loc(#loc69)
    %67 = arith.select %15, %66, %65 : index loc(#loc69)
    %68 = handshake.mux %67 [%falseResult_28, %trueResult_23] : index, none loc(#loc69)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %68 : none loc(#loc59)
    %69 = handshake.mux %55 [%falseResult_30, %trueResult] : index, none loc(#loc59)
    %70 = handshake.join %56, %58, %69 : none, none, none loc(#loc50)
    %71 = handshake.constant %70 {value = true} : i1 loc(#loc50)
    handshake.return %71 : i1 loc(#loc50)
  } loc(#loc50)
  handshake.func @_Z20convolve_1d_same_dsaPKfS0_Pfjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: i32 loc(fused<#di_subprogram6>[#loc12]), %arg4: i32 loc(fused<#di_subprogram6>[#loc12]), %arg5: none loc(fused<#di_subprogram6>[#loc12]), ...) -> none attributes {argNames = ["input", "kernel", "output", "input_size", "kernel_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc50)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %5 = handshake.constant %0 {value = -1 : i32} : i32 loc(#loc2)
    %6 = arith.shrui %arg4, %1 : i32 loc(#loc51)
    %7 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc62)
    %trueResult, %falseResult = handshake.cond_br %7, %0 : none loc(#loc59)
    %8 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %9 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc59)
    %10 = arith.index_cast %3 : i64 to index loc(#loc59)
    %11 = arith.index_cast %arg3 : i32 to index loc(#loc59)
    %index, %willContinue = dataflow.stream %10, %9, %11 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc59)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc59)
    %12 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc59)
    %13 = arith.index_cast %afterValue : index to i64 loc(#loc59)
    %14 = dataflow.invariant %afterCond, %8 : i1, i1 -> i1 loc(#loc69)
    %trueResult_0, %falseResult_1 = handshake.cond_br %14, %12 : none loc(#loc69)
    %15 = arith.trunci %13 : i64 to i32 loc(#loc2)
    %16 = arith.subi %15, %6 : i32 loc(#loc2)
    %17 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc69)
    %18 = arith.index_cast %arg4 : i32 to index loc(#loc69)
    %index_2, %willContinue_3 = dataflow.stream %10, %17, %18 {step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc69)
    %19 = dataflow.carry %willContinue_3, %4, %34 : i1, f32, f32 -> f32 loc(#loc69)
    %afterValue_6, %afterCond_7 = dataflow.gate %19, %willContinue_3 : f32, i1 -> f32, i1 loc(#loc69)
    handshake.sink %afterCond_7 : i1 loc(#loc69)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %19 : f32 loc(#loc69)
    %20 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc69)
    %21 = arith.index_cast %afterValue_4 : index to i64 loc(#loc69)
    %22 = arith.trunci %21 : i64 to i32 loc(#loc73)
    %23 = dataflow.invariant %afterCond_5, %16 : i1, i32 -> i32 loc(#loc73)
    %24 = arith.addi %23, %22 : i32 loc(#loc73)
    %25 = arith.cmpi sgt, %24, %5 : i32 loc(#loc75)
    %26 = arith.cmpi slt, %24, %arg3 : i32 loc(#loc75)
    %27 = arith.andi %25, %26 : i1 loc(#loc75)
    %28 = arith.extui %24 : i32 to i64 loc(#loc78)
    %29 = arith.index_cast %28 : i64 to index loc(#loc78)
    %dataResult, %addressResults = handshake.load [%29] %39#0, %trueResult_15 : index, f32 loc(#loc78)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %40#0, %trueResult_25 : index, f32 loc(#loc78)
    %30 = math.fma %dataResult, %dataResult_10, %afterValue_6 : f32 loc(#loc78)
    %31 = handshake.constant %20 {value = 0 : index} : index loc(#loc75)
    %32 = handshake.constant %20 {value = 1 : index} : index loc(#loc75)
    %33 = arith.select %27, %32, %31 : index loc(#loc75)
    %34 = handshake.mux %33 [%afterValue_6, %30] : index, f32 loc(#loc75)
    %35 = handshake.constant %12 {value = 0 : index} : index loc(#loc69)
    %36 = handshake.constant %12 {value = 1 : index} : index loc(#loc69)
    %37 = arith.select %14, %36, %35 : index loc(#loc69)
    %38 = handshake.mux %37 [%falseResult_9, %4] : index, f32 loc(#loc69)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %38, %56 : index, f32 loc(#loc67)
    %39:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc50)
    %40:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 1 : i32} : (index) -> (f32, none) loc(#loc50)
    %41 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc50)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc59)
    %trueResult_13, %falseResult_14 = handshake.cond_br %14, %42 : none loc(#loc69)
    %43 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_17 : i1, none, none -> none loc(#loc69)
    %trueResult_15, %falseResult_16 = handshake.cond_br %27, %43 : none loc(#loc75)
    %44 = handshake.constant %43 {value = 0 : index} : index loc(#loc75)
    %45 = handshake.constant %43 {value = 1 : index} : index loc(#loc75)
    %46 = arith.select %27, %45, %44 : index loc(#loc75)
    %47 = handshake.mux %46 [%falseResult_16, %39#1] : index, none loc(#loc75)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_3, %47 : none loc(#loc69)
    %48 = handshake.constant %42 {value = 0 : index} : index loc(#loc69)
    %49 = handshake.constant %42 {value = 1 : index} : index loc(#loc69)
    %50 = arith.select %14, %49, %48 : index loc(#loc69)
    %51 = handshake.mux %50 [%falseResult_18, %trueResult_13] : index, none loc(#loc69)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %51 : none loc(#loc59)
    %52 = handshake.constant %0 {value = 0 : index} : index loc(#loc59)
    %53 = handshake.constant %0 {value = 1 : index} : index loc(#loc59)
    %54 = arith.select %7, %53, %52 : index loc(#loc59)
    %55 = handshake.mux %54 [%falseResult_20, %trueResult] : index, none loc(#loc59)
    %56 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc59)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %41 : none loc(#loc59)
    %57 = handshake.mux %54 [%falseResult_22, %trueResult] : index, none loc(#loc59)
    %58 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc59)
    %trueResult_23, %falseResult_24 = handshake.cond_br %14, %58 : none loc(#loc69)
    %59 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_27 : i1, none, none -> none loc(#loc69)
    %trueResult_25, %falseResult_26 = handshake.cond_br %27, %59 : none loc(#loc75)
    %60 = handshake.constant %59 {value = 0 : index} : index loc(#loc75)
    %61 = handshake.constant %59 {value = 1 : index} : index loc(#loc75)
    %62 = arith.select %27, %61, %60 : index loc(#loc75)
    %63 = handshake.mux %62 [%falseResult_26, %40#1] : index, none loc(#loc75)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_3, %63 : none loc(#loc69)
    %64 = handshake.constant %58 {value = 0 : index} : index loc(#loc69)
    %65 = handshake.constant %58 {value = 1 : index} : index loc(#loc69)
    %66 = arith.select %14, %65, %64 : index loc(#loc69)
    %67 = handshake.mux %66 [%falseResult_28, %trueResult_23] : index, none loc(#loc69)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %67 : none loc(#loc59)
    %68 = handshake.mux %54 [%falseResult_30, %trueResult] : index, none loc(#loc59)
    %69 = handshake.join %55, %57, %68 : none, none, none loc(#loc50)
    handshake.return %69 : none loc(#loc52)
  } loc(#loc50)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc36)
    %false = arith.constant false loc(#loc36)
    %0 = seq.const_clock  low loc(#loc36)
    %c2_i32 = arith.constant 2 : i32 loc(#loc36)
    %1 = ub.poison : i64 loc(#loc36)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
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
    %2 = memref.get_global @str : memref<25xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<25xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<7xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<128xf32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<128xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc53)
      %12 = arith.uitofp %11 : i32 to f32 loc(#loc53)
      %13 = arith.mulf %12, %cst : f32 loc(#loc53)
      %14 = arith.mulf %13, %cst_0 : f32 loc(#loc53)
      %15 = math.sin %14 : f32 loc(#loc53)
      %16 = arith.index_cast %arg0 : i64 to index loc(#loc53)
      memref.store %15, %alloca[%16] : memref<128xf32> loc(#loc53)
      %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc44)
      %18 = arith.cmpi ne, %17, %c128_i64 : i64 loc(#loc54)
      scf.condition(%18) %17 : i64 loc(#loc41)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc22])):
      scf.yield %arg0 : i64 loc(#loc41)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc41)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc55)
      memref.store %cst_1, %alloca_3[%11] : memref<7xf32> loc(#loc55)
      %12 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
      %13 = arith.cmpi ne, %12, %c7_i64 : i64 loc(#loc56)
      scf.condition(%13) %12 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block16>[#loc24])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %cast = memref.cast %alloca : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %cast_6 = memref.cast %alloca_3 : memref<7xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    %cast_7 = memref.cast %alloca_4 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc37)
    call @_Z20convolve_1d_same_cpuPKfS0_Pfjj(%cast, %cast_6, %cast_7, %c128_i32, %c7_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc37)
    %cast_8 = memref.cast %alloca_5 : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_9, %ready_10 = esi.wrap.vr %cast_6, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_11, %ready_12 = esi.wrap.vr %cast_8, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    %chanOutput_13, %ready_14 = esi.wrap.vr %c128_i32, %true : i32 loc(#loc38)
    %chanOutput_15, %ready_16 = esi.wrap.vr %c7_i32, %true : i32 loc(#loc38)
    %chanOutput_17, %ready_18 = esi.wrap.vr %true, %true : i1 loc(#loc38)
    %6 = handshake.esi_instance @_Z20convolve_1d_same_dsaPKfS0_Pfjj_esi "_Z20convolve_1d_same_dsaPKfS0_Pfjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_9, %chanOutput_11, %chanOutput_13, %chanOutput_15, %chanOutput_17) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc38)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc38)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc60)
      %12 = memref.load %alloca_4[%11] : memref<128xf32> loc(#loc60)
      %13 = memref.load %alloca_5[%11] : memref<128xf32> loc(#loc60)
      %14 = arith.subf %12, %13 : f32 loc(#loc60)
      %15 = math.absf %14 : f32 loc(#loc60)
      %16 = arith.cmpf ule, %15, %cst_2 : f32 loc(#loc60)
      %17:3 = scf.if %16 -> (i64, i32, i32) {
        %19 = arith.addi %arg0, %c1_i64 : i64 loc(#loc46)
        %20 = arith.cmpi eq, %19, %c128_i64 : i64 loc(#loc46)
        %21 = arith.extui %20 : i1 to i32 loc(#loc43)
        %22 = arith.cmpi ne, %19, %c128_i64 : i64 loc(#loc57)
        %23 = arith.extui %22 : i1 to i32 loc(#loc43)
        scf.yield %19, %21, %23 : i64, i32, i32 loc(#loc60)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc60)
      } loc(#loc60)
      %18 = arith.trunci %17#2 : i32 to i1 loc(#loc43)
      scf.condition(%18) %17#0, %16, %17#1 : i64, i1, i32 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block17>[#loc28]), %arg1: i1 loc(fused<#di_lexical_block17>[#loc28]), %arg2: i32 loc(fused<#di_lexical_block17>[#loc28])):
      scf.yield %arg0 : i64 loc(#loc43)
    } loc(#loc43)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc43)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc43)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<25xi8> -> index loc(#loc63)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc63)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc63)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc63)
      scf.yield %c1_i32 : i32 loc(#loc64)
    } loc(#loc43)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<25xi8> -> index loc(#loc39)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc39)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc39)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc39)
    } loc(#loc2)
    return %10 : i32 loc(#loc40)
  } loc(#loc36)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @sinf(f32) -> f32 loc(#loc35)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#di_file2 = #llvm.di_file<"/usr/include/bits/mathcalls.h" in "">
#loc = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":0:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":17:0)
#loc6 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":22:0)
#loc7 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":23:0)
#loc8 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":24:0)
#loc9 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":25:0)
#loc10 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":27:0)
#loc11 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":29:0)
#loc13 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":38:0)
#loc14 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":42:0)
#loc15 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":44:0)
#loc16 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":45:0)
#loc17 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":46:0)
#loc18 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":47:0)
#loc19 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":50:0)
#loc20 = loc("tests/app/convolve_1d_same/convolve_1d_same.cpp":52:0)
#loc21 = loc("tests/app/convolve_1d_same/main.cpp":6:0)
#loc23 = loc("tests/app/convolve_1d_same/main.cpp":22:0)
#loc25 = loc("tests/app/convolve_1d_same/main.cpp":27:0)
#loc26 = loc("tests/app/convolve_1d_same/main.cpp":31:0)
#loc27 = loc("tests/app/convolve_1d_same/main.cpp":32:0)
#loc29 = loc("tests/app/convolve_1d_same/main.cpp":35:0)
#loc30 = loc("tests/app/convolve_1d_same/main.cpp":36:0)
#loc31 = loc("tests/app/convolve_1d_same/main.cpp":37:0)
#loc32 = loc("tests/app/convolve_1d_same/main.cpp":41:0)
#loc33 = loc("tests/app/convolve_1d_same/main.cpp":43:0)
#loc34 = loc("/usr/include/bits/mathcalls.h":64:0)
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type, #di_basic_type>
#di_subprogram3 = #llvm.di_subprogram<scope = #di_file2, name = "sinf", file = #di_file2, line = 64, subprogramFlags = Optimized, type = #di_subroutine_type1>
#loc35 = loc(fused<#di_subprogram3>[#loc34])
#loc36 = loc(fused<#di_subprogram4>[#loc21])
#loc37 = loc(fused<#di_subprogram4>[#loc26])
#loc38 = loc(fused<#di_subprogram4>[#loc27])
#loc39 = loc(fused<#di_subprogram4>[#loc32])
#loc40 = loc(fused<#di_subprogram4>[#loc33])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 21>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 26>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 34>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 21>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 26>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 34>
#loc44 = loc(fused<#di_lexical_block18>[#loc22])
#loc45 = loc(fused<#di_lexical_block19>[#loc24])
#loc46 = loc(fused<#di_lexical_block20>[#loc28])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 42>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 35>
#loc48 = loc(fused<#di_subprogram5>[#loc3])
#loc49 = loc(fused<#di_subprogram5>[#loc11])
#loc51 = loc(fused<#di_subprogram6>[#loc13])
#loc52 = loc(fused<#di_subprogram6>[#loc20])
#loc53 = loc(fused<#di_lexical_block21>[#loc23])
#loc54 = loc(fused[#loc41, #loc44])
#loc55 = loc(fused<#di_lexical_block22>[#loc25])
#loc56 = loc(fused[#loc42, #loc45])
#loc57 = loc(fused[#loc43, #loc46])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 42>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 35>
#loc59 = loc(fused<#di_lexical_block25>[#loc14])
#loc60 = loc(fused<#di_lexical_block26>[#loc29])
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 42>
#loc61 = loc(fused<#di_lexical_block27>[#loc4])
#loc62 = loc(fused<#di_lexical_block28>[#loc14])
#loc63 = loc(fused<#di_lexical_block29>[#loc30])
#loc64 = loc(fused<#di_lexical_block29>[#loc31])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 44>
#loc65 = loc(fused<#di_lexical_block30>[#loc10])
#loc66 = loc(fused[#loc58, #loc61])
#loc67 = loc(fused<#di_lexical_block31>[#loc19])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 21>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 44>
#loc69 = loc(fused<#di_lexical_block33>[#loc15])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 21>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 44>
#loc70 = loc(fused<#di_lexical_block34>[#loc5])
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 23>
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file, line = 46>
#loc71 = loc(fused<#di_lexical_block36>[#loc6])
#loc72 = loc(fused[#loc68, #loc70])
#loc73 = loc(fused<#di_lexical_block37>[#loc16])
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file, line = 23>
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file, line = 46>
#loc74 = loc(fused<#di_lexical_block38>[#loc7])
#loc75 = loc(fused<#di_lexical_block39>[#loc17])
#loc76 = loc(fused<#di_lexical_block40>[#loc8])
#loc77 = loc(fused<#di_lexical_block40>[#loc9])
#loc78 = loc(fused<#di_lexical_block41>[#loc18])
