#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/batchnorm/batchnorm.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file2 = #llvm.di_file<"tests/app/batchnorm/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/batchnorm/batchnorm.cpp":17:0)
#loc3 = loc("tests/app/batchnorm/batchnorm.cpp":27:0)
#loc5 = loc("tests/app/batchnorm/batchnorm.cpp":30:0)
#loc6 = loc("tests/app/batchnorm/batchnorm.cpp":31:0)
#loc12 = loc("tests/app/batchnorm/batchnorm.cpp":42:0)
#loc22 = loc("tests/app/batchnorm/main.cpp":26:0)
#loc25 = loc("tests/app/batchnorm/main.cpp":31:0)
#loc31 = loc("tests/app/batchnorm/main.cpp":45:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file2, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 128, elements = #llvm.di_subrange<count = 4 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 27>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 54>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file2, line = 26>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file2, line = 31>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file2, line = 45>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 27>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 54>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "epsilon", file = #di_file, line = 26, arg = 10, type = #di_derived_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram1, name = "epsilon", file = #di_file, line = 51, arg = 10, type = #di_derived_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "epsilon", file = #di_file2, line = 10, type = #di_derived_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file2, line = 13, type = #di_composite_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "mean", file = #di_file2, line = 16, type = #di_composite_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "variance", file = #di_file2, line = 17, type = #di_composite_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "gamma", file = #di_file2, line = 18, type = #di_composite_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram2, name = "beta", file = #di_file2, line = 19, type = #di_composite_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file2, line = 22, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file2, line = 23, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 27>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 54>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 22, arg = 6, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block, name = "c", file = #di_file, line = 27, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 47, arg = 6, type = #di_derived_type4>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "c", file = #di_file, line = 54, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file2, line = 26, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "c", file = #di_file2, line = 31, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file2, line = 45, type = #di_derived_type5>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 30>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 57>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 17, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "mean", file = #di_file, line = 18, arg = 2, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "variance", file = #di_file, line = 19, arg = 3, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "gamma", file = #di_file, line = 20, arg = 4, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram, name = "beta", file = #di_file, line = 21, arg = 5, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram, name = "C", file = #di_file, line = 23, arg = 7, type = #di_derived_type7>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram, name = "H", file = #di_file, line = 24, arg = 8, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram, name = "W", file = #di_file, line = 25, arg = 9, type = #di_derived_type7>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "inv_std", file = #di_file, line = 28, type = #di_basic_type>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 42, arg = 1, type = #di_derived_type6>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "mean", file = #di_file, line = 43, arg = 2, type = #di_derived_type6>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram1, name = "variance", file = #di_file, line = 44, arg = 3, type = #di_derived_type6>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram1, name = "gamma", file = #di_file, line = 45, arg = 4, type = #di_derived_type6>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram1, name = "beta", file = #di_file, line = 46, arg = 5, type = #di_derived_type6>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_subprogram1, name = "C", file = #di_file, line = 48, arg = 7, type = #di_derived_type7>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram1, name = "H", file = #di_file, line = 49, arg = 8, type = #di_derived_type7>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram1, name = "W", file = #di_file, line = 50, arg = 9, type = #di_derived_type7>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "inv_std", file = #di_file, line = 55, type = #di_basic_type>
#di_local_variable35 = #llvm.di_local_variable<scope = #di_subprogram2, name = "C", file = #di_file2, line = 7, type = #di_derived_type7>
#di_local_variable36 = #llvm.di_local_variable<scope = #di_subprogram2, name = "H", file = #di_file2, line = 8, type = #di_derived_type7>
#di_local_variable37 = #llvm.di_local_variable<scope = #di_subprogram2, name = "W", file = #di_file2, line = 9, type = #di_derived_type7>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 30>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 57>
#di_local_variable38 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "h", file = #di_file, line = 30, type = #di_derived_type5>
#di_local_variable39 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "h", file = #di_file, line = 57, type = #di_derived_type5>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file2, name = "main", file = #di_file2, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable35, #di_local_variable36, #di_local_variable37, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable14, #di_local_variable15, #di_local_variable16>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 30>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 57>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file2, line = 26>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file2, line = 31>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file2, line = 45>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 31>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 58>
#loc43 = loc(fused<#di_lexical_block15>[#loc22])
#loc44 = loc(fused<#di_lexical_block16>[#loc25])
#loc45 = loc(fused<#di_lexical_block17>[#loc31])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 31>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 58>
#di_local_variable40 = #llvm.di_local_variable<scope = #di_lexical_block18, name = "w", file = #di_file, line = 31, type = #di_derived_type5>
#di_local_variable41 = #llvm.di_local_variable<scope = #di_lexical_block19, name = "w", file = #di_file, line = 58, type = #di_derived_type5>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 31>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 58>
#di_local_variable42 = #llvm.di_local_variable<scope = #di_lexical_block28, name = "idx", file = #di_file, line = 32, type = #di_derived_type5>
#di_local_variable43 = #llvm.di_local_variable<scope = #di_lexical_block28, name = "normalized", file = #di_file, line = 33, type = #di_basic_type>
#di_local_variable44 = #llvm.di_local_variable<scope = #di_lexical_block29, name = "idx", file = #di_file, line = 59, type = #di_derived_type5>
#di_local_variable45 = #llvm.di_local_variable<scope = #di_lexical_block29, name = "normalized", file = #di_file, line = 60, type = #di_basic_type>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "batchnorm_cpu", linkageName = "_Z13batchnorm_cpuPKfS0_S0_S0_S0_Pfjjjf", file = #di_file, line = 17, scopeLine = 26, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable10, #di_local_variable22, #di_local_variable23, #di_local_variable24, #di_local_variable, #di_local_variable11, #di_local_variable25, #di_local_variable38, #di_local_variable40, #di_local_variable42, #di_local_variable43>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "batchnorm_dsa", linkageName = "_Z13batchnorm_dsaPKfS0_S0_S0_S0_Pfjjjf", file = #di_file, line = 42, scopeLine = 51, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable26, #di_local_variable27, #di_local_variable28, #di_local_variable29, #di_local_variable30, #di_local_variable12, #di_local_variable31, #di_local_variable32, #di_local_variable33, #di_local_variable1, #di_local_variable13, #di_local_variable34, #di_local_variable39, #di_local_variable41, #di_local_variable44, #di_local_variable45>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 27>
#loc60 = loc(fused<#di_subprogram5>[#loc1])
#loc62 = loc(fused<#di_subprogram6>[#loc12])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 27>
#loc64 = loc(fused<#di_lexical_block32>[#loc3])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 27>
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 30>
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file, line = 30>
#loc71 = loc(fused<#di_lexical_block38>[#loc5])
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file, line = 30>
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_lexical_block42, file = #di_file, line = 31>
#loc75 = loc(fused<#di_lexical_block44>[#loc6])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<34xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 98, 97, 116, 99, 104, 110, 111, 114, 109, 47, 98, 97, 116, 99, 104, 110, 111, 114, 109, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @str : memref<18xi8> = dense<[98, 97, 116, 99, 104, 110, 111, 114, 109, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<18xi8> = dense<[98, 97, 116, 99, 104, 110, 111, 114, 109, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z13batchnorm_cpuPKfS0_S0_S0_S0_Pfjjjf(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg4: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg5: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg6: i32 loc(fused<#di_subprogram5>[#loc1]), %arg7: i32 loc(fused<#di_subprogram5>[#loc1]), %arg8: i32 loc(fused<#di_subprogram5>[#loc1]), %arg9: f32 loc(fused<#di_subprogram5>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg6, %c0_i32 : i32 loc(#loc66)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc2)
      %2 = arith.cmpi eq, %arg8, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg6 : i32 to i64 loc(#loc66)
      %4 = arith.extui %arg8 : i32 to i64 loc(#loc2)
      %5 = scf.while (%arg10 = %c0_i64) : (i64) -> i64 {
        %6 = arith.index_cast %arg10 : i64 to index loc(#loc68)
        %7 = memref.load %arg2[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc68)
        %8 = arith.addf %arg9, %7 : f32 loc(#loc68)
        %9 = math.sqrt %8 : f32 loc(#loc68)
        %10 = arith.divf %cst, %9 : f32 loc(#loc68)
        scf.if %1 {
        } else {
          %13 = arith.trunci %arg10 : i64 to i32 loc(#loc2)
          %14 = arith.muli %arg7, %13 : i32 loc(#loc2)
          %15 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
            scf.if %2 {
            } else {
              %18 = arith.addi %arg11, %14 : i32 loc(#loc2)
              %19 = arith.muli %18, %arg8 : i32 loc(#loc2)
              %20 = memref.load %arg1[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc2)
              %21 = memref.load %arg3[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc2)
              %22 = memref.load %arg4[%6] : memref<?xf32, strided<[1], offset: ?>> loc(#loc2)
              %23 = scf.while (%arg12 = %c0_i64) : (i64) -> i64 {
                %24 = arith.trunci %arg12 : i64 to i32 loc(#loc78)
                %25 = arith.addi %19, %24 : i32 loc(#loc78)
                %26 = arith.extui %25 : i32 to i64 loc(#loc79)
                %27 = arith.index_cast %26 : i64 to index loc(#loc79)
                %28 = memref.load %arg0[%27] : memref<?xf32, strided<[1], offset: ?>> loc(#loc79)
                %29 = arith.subf %28, %20 : f32 loc(#loc79)
                %30 = arith.mulf %10, %29 : f32 loc(#loc79)
                %31 = math.fma %21, %30, %22 : f32 loc(#loc80)
                memref.store %31, %arg5[%27] : memref<?xf32, strided<[1], offset: ?>> loc(#loc80)
                %32 = arith.addi %arg12, %c1_i64 : i64 loc(#loc77)
                %33 = arith.cmpi ne, %32, %4 : i64 loc(#loc81)
                scf.condition(%33) %32 : i64 loc(#loc75)
              } do {
              ^bb0(%arg12: i64 loc(fused<#di_lexical_block44>[#loc6])):
                scf.yield %arg12 : i64 loc(#loc75)
              } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc75)
            } loc(#loc75)
            %16 = arith.addi %arg11, %c1_i32 : i32 loc(#loc73)
            %17 = arith.cmpi ne, %16, %arg7 : i32 loc(#loc74)
            scf.condition(%17) %16 : i32 loc(#loc71)
          } do {
          ^bb0(%arg11: i32 loc(fused<#di_lexical_block38>[#loc5])):
            scf.yield %arg11 : i32 loc(#loc71)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc71)
        } loc(#loc71)
        %11 = arith.addi %arg10, %c1_i64 : i64 loc(#loc66)
        %12 = arith.cmpi ne, %11, %3 : i64 loc(#loc69)
        scf.condition(%12) %11 : i64 loc(#loc64)
      } do {
      ^bb0(%arg10: i64 loc(fused<#di_lexical_block32>[#loc3])):
        scf.yield %arg10 : i64 loc(#loc64)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc64)
    } loc(#loc64)
    return loc(#loc61)
  } loc(#loc60)
  func.func private @sqrtf(f32) -> f32 loc(#loc37)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  handshake.func @_Z13batchnorm_dsaPKfS0_S0_S0_S0_Pfjjjf_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg4: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg5: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg6: i32 loc(fused<#di_subprogram6>[#loc12]), %arg7: i32 loc(fused<#di_subprogram6>[#loc12]), %arg8: i32 loc(fused<#di_subprogram6>[#loc12]), %arg9: f32 loc(fused<#di_subprogram6>[#loc12]), %arg10: i1 loc(fused<#di_subprogram6>[#loc12]), ...) -> i1 attributes {argNames = ["input", "mean", "variance", "gamma", "beta", "output", "C", "H", "W", "epsilon", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg10 : i1 loc(#loc62)
    %1 = handshake.join %0 : none loc(#loc62)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %1 {value = 1.000000e+00 : f32} : f32 loc(#loc2)
    %5 = arith.cmpi eq, %arg6, %2 : i32 loc(#loc67)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc65)
    %6 = arith.cmpi eq, %arg7, %2 : i32 loc(#loc2)
    %7 = arith.cmpi eq, %arg8, %2 : i32 loc(#loc2)
    %8 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc65)
    %9 = arith.index_cast %3 : i64 to index loc(#loc65)
    %10 = arith.index_cast %arg6 : i32 to index loc(#loc65)
    %index, %willContinue = dataflow.stream %9, %8, %10 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc65)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc65)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc65)
    %12 = arith.index_cast %afterValue : index to i64 loc(#loc65)
    %dataResult, %addressResults = handshake.load [%afterValue] %45#0, %48 : index, f32 loc(#loc70)
    %13 = arith.addf %arg9, %dataResult : f32 loc(#loc70)
    %14 = math.sqrt %13 : f32 loc(#loc70)
    %15 = arith.divf %4, %14 : f32 loc(#loc70)
    %16 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc72)
    %trueResult_0, %falseResult_1 = handshake.cond_br %16, %11 : none loc(#loc72)
    %17 = arith.trunci %12 : i64 to i32 loc(#loc2)
    %18 = arith.muli %arg7, %17 : i32 loc(#loc2)
    %19 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc72)
    %20 = arith.index_cast %2 : i32 to index loc(#loc72)
    %21 = arith.index_cast %arg7 : i32 to index loc(#loc72)
    %index_2, %willContinue_3 = dataflow.stream %20, %19, %21 {step_op = "+=", stop_cond = "!="} loc(#loc72)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc72)
    %22 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc72)
    %23 = arith.index_cast %afterValue_4 : index to i32 loc(#loc72)
    %24 = dataflow.invariant %afterCond, %7 : i1, i1 -> i1 loc(#loc76)
    %trueResult_6, %falseResult_7 = handshake.cond_br %24, %22 : none loc(#loc76)
    %25 = dataflow.invariant %afterCond_5, %18 : i1, i32 -> i32 loc(#loc2)
    %26 = arith.addi %23, %25 : i32 loc(#loc2)
    %27 = arith.muli %26, %arg8 : i32 loc(#loc2)
    %dataResult_8, %addressResults_9 = handshake.load [%afterValue] %44#0, %falseResult_44 : index, f32 loc(#loc2)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue] %42#0, %falseResult_26 : index, f32 loc(#loc2)
    %dataResult_12, %addressResults_13 = handshake.load [%afterValue] %43#0, %falseResult_52 : index, f32 loc(#loc2)
    %28 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc76)
    %29 = arith.index_cast %arg8 : i32 to index loc(#loc76)
    %index_14, %willContinue_15 = dataflow.stream %9, %28, %29 {step_op = "+=", stop_cond = "!="} loc(#loc76)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc76)
    %30 = arith.index_cast %afterValue_16 : index to i64 loc(#loc76)
    %31 = arith.trunci %30 : i64 to i32 loc(#loc82)
    %32 = dataflow.invariant %afterCond_17, %27 : i1, i32 -> i32 loc(#loc82)
    %33 = arith.addi %32, %31 : i32 loc(#loc82)
    %34 = arith.extui %33 : i32 to i64 loc(#loc83)
    %35 = arith.index_cast %34 : i64 to index loc(#loc83)
    %dataResult_18, %addressResults_19 = handshake.load [%35] %46#0, %66 : index, f32 loc(#loc83)
    %36 = dataflow.invariant %afterCond_17, %dataResult_8 : i1, f32 -> f32 loc(#loc83)
    %37 = arith.subf %dataResult_18, %36 : f32 loc(#loc83)
    %38 = arith.mulf %15, %37 : f32 loc(#loc83)
    %39 = dataflow.invariant %afterCond_17, %dataResult_10 : i1, f32 -> f32 loc(#loc84)
    %40 = dataflow.invariant %afterCond_17, %dataResult_12 : i1, f32 -> f32 loc(#loc84)
    %41 = math.fma %39, %38, %40 : f32 loc(#loc84)
    %dataResult_20, %addressResult = handshake.store [%35] %41, %100 : index, f32 loc(#loc84)
    %42:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (f32, none) loc(#loc62)
    %43:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (f32, none) loc(#loc62)
    %44:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_9) {id = 2 : i32} : (index) -> (f32, none) loc(#loc62)
    %45:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 3 : i32} : (index) -> (f32, none) loc(#loc62)
    %46:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_19) {id = 4 : i32} : (index) -> (f32, none) loc(#loc62)
    %47 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_20, %addressResult) {id = 5 : i32} : (f32, index) -> none loc(#loc62)
    %48 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc65)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %45#1 : none loc(#loc65)
    %49 = handshake.constant %1 {value = 0 : index} : index loc(#loc65)
    %50 = handshake.constant %1 {value = 1 : index} : index loc(#loc65)
    %51 = arith.select %5, %50, %49 : index loc(#loc65)
    %52 = handshake.mux %51 [%falseResult_22, %trueResult] : index, none loc(#loc65)
    %53 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc65)
    %trueResult_23, %falseResult_24 = handshake.cond_br %16, %53 : none loc(#loc72)
    %54 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_27 : i1, none, none -> none loc(#loc72)
    %trueResult_25, %falseResult_26 = handshake.cond_br %24, %54 : none loc(#loc76)
    %55 = handshake.constant %54 {value = 0 : index} : index loc(#loc76)
    %56 = handshake.constant %54 {value = 1 : index} : index loc(#loc76)
    %57 = arith.select %24, %56, %55 : index loc(#loc76)
    %58 = handshake.mux %57 [%42#1, %trueResult_25] : index, none loc(#loc76)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_3, %58 : none loc(#loc72)
    %59 = handshake.constant %53 {value = 0 : index} : index loc(#loc72)
    %60 = handshake.constant %53 {value = 1 : index} : index loc(#loc72)
    %61 = arith.select %16, %60, %59 : index loc(#loc72)
    %62 = handshake.mux %61 [%falseResult_28, %trueResult_23] : index, none loc(#loc72)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %62 : none loc(#loc65)
    %63 = handshake.mux %51 [%falseResult_30, %trueResult] : index, none loc(#loc65)
    %64 = dataflow.carry %willContinue, %falseResult, %trueResult_39 : i1, none, none -> none loc(#loc65)
    %trueResult_31, %falseResult_32 = handshake.cond_br %16, %64 : none loc(#loc72)
    %65 = dataflow.carry %willContinue_3, %falseResult_32, %trueResult_37 : i1, none, none -> none loc(#loc72)
    %trueResult_33, %falseResult_34 = handshake.cond_br %24, %65 : none loc(#loc76)
    %66 = dataflow.carry %willContinue_15, %falseResult_34, %trueResult_35 : i1, none, none -> none loc(#loc76)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_15, %46#1 : none loc(#loc76)
    %67 = handshake.constant %65 {value = 0 : index} : index loc(#loc76)
    %68 = handshake.constant %65 {value = 1 : index} : index loc(#loc76)
    %69 = arith.select %24, %68, %67 : index loc(#loc76)
    %70 = handshake.mux %69 [%falseResult_36, %trueResult_33] : index, none loc(#loc76)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_3, %70 : none loc(#loc72)
    %71 = handshake.constant %64 {value = 0 : index} : index loc(#loc72)
    %72 = handshake.constant %64 {value = 1 : index} : index loc(#loc72)
    %73 = arith.select %16, %72, %71 : index loc(#loc72)
    %74 = handshake.mux %73 [%falseResult_38, %trueResult_31] : index, none loc(#loc72)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue, %74 : none loc(#loc65)
    %75 = handshake.mux %51 [%falseResult_40, %trueResult] : index, none loc(#loc65)
    %76 = dataflow.carry %willContinue, %falseResult, %trueResult_47 : i1, none, none -> none loc(#loc65)
    %trueResult_41, %falseResult_42 = handshake.cond_br %16, %76 : none loc(#loc72)
    %77 = dataflow.carry %willContinue_3, %falseResult_42, %trueResult_45 : i1, none, none -> none loc(#loc72)
    %trueResult_43, %falseResult_44 = handshake.cond_br %24, %77 : none loc(#loc76)
    %78 = handshake.constant %77 {value = 0 : index} : index loc(#loc76)
    %79 = handshake.constant %77 {value = 1 : index} : index loc(#loc76)
    %80 = arith.select %24, %79, %78 : index loc(#loc76)
    %81 = handshake.mux %80 [%44#1, %trueResult_43] : index, none loc(#loc76)
    %trueResult_45, %falseResult_46 = handshake.cond_br %willContinue_3, %81 : none loc(#loc72)
    %82 = handshake.constant %76 {value = 0 : index} : index loc(#loc72)
    %83 = handshake.constant %76 {value = 1 : index} : index loc(#loc72)
    %84 = arith.select %16, %83, %82 : index loc(#loc72)
    %85 = handshake.mux %84 [%falseResult_46, %trueResult_41] : index, none loc(#loc72)
    %trueResult_47, %falseResult_48 = handshake.cond_br %willContinue, %85 : none loc(#loc65)
    %86 = handshake.mux %51 [%falseResult_48, %trueResult] : index, none loc(#loc65)
    %87 = dataflow.carry %willContinue, %falseResult, %trueResult_55 : i1, none, none -> none loc(#loc65)
    %trueResult_49, %falseResult_50 = handshake.cond_br %16, %87 : none loc(#loc72)
    %88 = dataflow.carry %willContinue_3, %falseResult_50, %trueResult_53 : i1, none, none -> none loc(#loc72)
    %trueResult_51, %falseResult_52 = handshake.cond_br %24, %88 : none loc(#loc76)
    %89 = handshake.constant %88 {value = 0 : index} : index loc(#loc76)
    %90 = handshake.constant %88 {value = 1 : index} : index loc(#loc76)
    %91 = arith.select %24, %90, %89 : index loc(#loc76)
    %92 = handshake.mux %91 [%43#1, %trueResult_51] : index, none loc(#loc76)
    %trueResult_53, %falseResult_54 = handshake.cond_br %willContinue_3, %92 : none loc(#loc72)
    %93 = handshake.constant %87 {value = 0 : index} : index loc(#loc72)
    %94 = handshake.constant %87 {value = 1 : index} : index loc(#loc72)
    %95 = arith.select %16, %94, %93 : index loc(#loc72)
    %96 = handshake.mux %95 [%falseResult_54, %trueResult_49] : index, none loc(#loc72)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue, %96 : none loc(#loc65)
    %97 = handshake.mux %51 [%falseResult_56, %trueResult] : index, none loc(#loc65)
    %98 = dataflow.carry %willContinue, %falseResult, %trueResult_65 : i1, none, none -> none loc(#loc65)
    %trueResult_57, %falseResult_58 = handshake.cond_br %16, %98 : none loc(#loc72)
    %99 = dataflow.carry %willContinue_3, %falseResult_58, %trueResult_63 : i1, none, none -> none loc(#loc72)
    %trueResult_59, %falseResult_60 = handshake.cond_br %24, %99 : none loc(#loc76)
    %100 = dataflow.carry %willContinue_15, %falseResult_60, %trueResult_61 : i1, none, none -> none loc(#loc76)
    %trueResult_61, %falseResult_62 = handshake.cond_br %willContinue_15, %47 : none loc(#loc76)
    %101 = handshake.constant %99 {value = 0 : index} : index loc(#loc76)
    %102 = handshake.constant %99 {value = 1 : index} : index loc(#loc76)
    %103 = arith.select %24, %102, %101 : index loc(#loc76)
    %104 = handshake.mux %103 [%falseResult_62, %trueResult_59] : index, none loc(#loc76)
    %trueResult_63, %falseResult_64 = handshake.cond_br %willContinue_3, %104 : none loc(#loc72)
    %105 = handshake.constant %98 {value = 0 : index} : index loc(#loc72)
    %106 = handshake.constant %98 {value = 1 : index} : index loc(#loc72)
    %107 = arith.select %16, %106, %105 : index loc(#loc72)
    %108 = handshake.mux %107 [%falseResult_64, %trueResult_57] : index, none loc(#loc72)
    %trueResult_65, %falseResult_66 = handshake.cond_br %willContinue, %108 : none loc(#loc65)
    %109 = handshake.mux %51 [%falseResult_66, %trueResult] : index, none loc(#loc65)
    %110 = handshake.join %52, %63, %75, %86, %97, %109 : none, none, none, none, none, none loc(#loc62)
    %111 = handshake.constant %110 {value = true} : i1 loc(#loc62)
    handshake.return %111 : i1 loc(#loc62)
  } loc(#loc62)
  handshake.func @_Z13batchnorm_dsaPKfS0_S0_S0_S0_Pfjjjf(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg4: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg5: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg6: i32 loc(fused<#di_subprogram6>[#loc12]), %arg7: i32 loc(fused<#di_subprogram6>[#loc12]), %arg8: i32 loc(fused<#di_subprogram6>[#loc12]), %arg9: f32 loc(fused<#di_subprogram6>[#loc12]), %arg10: none loc(fused<#di_subprogram6>[#loc12]), ...) -> none attributes {argNames = ["input", "mean", "variance", "gamma", "beta", "output", "C", "H", "W", "epsilon", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg10 : none loc(#loc62)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = handshake.constant %0 {value = 1.000000e+00 : f32} : f32 loc(#loc2)
    %4 = arith.cmpi eq, %arg6, %1 : i32 loc(#loc67)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc65)
    %5 = arith.cmpi eq, %arg7, %1 : i32 loc(#loc2)
    %6 = arith.cmpi eq, %arg8, %1 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc65)
    %8 = arith.index_cast %2 : i64 to index loc(#loc65)
    %9 = arith.index_cast %arg6 : i32 to index loc(#loc65)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll=auto"], step_op = "+=", stop_cond = "!="} loc(#loc65)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc65)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc65)
    %11 = arith.index_cast %afterValue : index to i64 loc(#loc65)
    %dataResult, %addressResults = handshake.load [%afterValue] %44#0, %47 : index, f32 loc(#loc70)
    %12 = arith.addf %arg9, %dataResult : f32 loc(#loc70)
    %13 = math.sqrt %12 : f32 loc(#loc70)
    %14 = arith.divf %3, %13 : f32 loc(#loc70)
    %15 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc72)
    %trueResult_0, %falseResult_1 = handshake.cond_br %15, %10 : none loc(#loc72)
    %16 = arith.trunci %11 : i64 to i32 loc(#loc2)
    %17 = arith.muli %arg7, %16 : i32 loc(#loc2)
    %18 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc72)
    %19 = arith.index_cast %1 : i32 to index loc(#loc72)
    %20 = arith.index_cast %arg7 : i32 to index loc(#loc72)
    %index_2, %willContinue_3 = dataflow.stream %19, %18, %20 {step_op = "+=", stop_cond = "!="} loc(#loc72)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc72)
    %21 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc72)
    %22 = arith.index_cast %afterValue_4 : index to i32 loc(#loc72)
    %23 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc76)
    %trueResult_6, %falseResult_7 = handshake.cond_br %23, %21 : none loc(#loc76)
    %24 = dataflow.invariant %afterCond_5, %17 : i1, i32 -> i32 loc(#loc2)
    %25 = arith.addi %22, %24 : i32 loc(#loc2)
    %26 = arith.muli %25, %arg8 : i32 loc(#loc2)
    %dataResult_8, %addressResults_9 = handshake.load [%afterValue] %43#0, %falseResult_44 : index, f32 loc(#loc2)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue] %41#0, %falseResult_26 : index, f32 loc(#loc2)
    %dataResult_12, %addressResults_13 = handshake.load [%afterValue] %42#0, %falseResult_52 : index, f32 loc(#loc2)
    %27 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc76)
    %28 = arith.index_cast %arg8 : i32 to index loc(#loc76)
    %index_14, %willContinue_15 = dataflow.stream %8, %27, %28 {step_op = "+=", stop_cond = "!="} loc(#loc76)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc76)
    %29 = arith.index_cast %afterValue_16 : index to i64 loc(#loc76)
    %30 = arith.trunci %29 : i64 to i32 loc(#loc82)
    %31 = dataflow.invariant %afterCond_17, %26 : i1, i32 -> i32 loc(#loc82)
    %32 = arith.addi %31, %30 : i32 loc(#loc82)
    %33 = arith.extui %32 : i32 to i64 loc(#loc83)
    %34 = arith.index_cast %33 : i64 to index loc(#loc83)
    %dataResult_18, %addressResults_19 = handshake.load [%34] %45#0, %65 : index, f32 loc(#loc83)
    %35 = dataflow.invariant %afterCond_17, %dataResult_8 : i1, f32 -> f32 loc(#loc83)
    %36 = arith.subf %dataResult_18, %35 : f32 loc(#loc83)
    %37 = arith.mulf %14, %36 : f32 loc(#loc83)
    %38 = dataflow.invariant %afterCond_17, %dataResult_10 : i1, f32 -> f32 loc(#loc84)
    %39 = dataflow.invariant %afterCond_17, %dataResult_12 : i1, f32 -> f32 loc(#loc84)
    %40 = math.fma %38, %37, %39 : f32 loc(#loc84)
    %dataResult_20, %addressResult = handshake.store [%34] %40, %99 : index, f32 loc(#loc84)
    %41:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (f32, none) loc(#loc62)
    %42:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (f32, none) loc(#loc62)
    %43:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_9) {id = 2 : i32} : (index) -> (f32, none) loc(#loc62)
    %44:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 3 : i32} : (index) -> (f32, none) loc(#loc62)
    %45:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_19) {id = 4 : i32} : (index) -> (f32, none) loc(#loc62)
    %46 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_20, %addressResult) {id = 5 : i32} : (f32, index) -> none loc(#loc62)
    %47 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc65)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %44#1 : none loc(#loc65)
    %48 = handshake.constant %0 {value = 0 : index} : index loc(#loc65)
    %49 = handshake.constant %0 {value = 1 : index} : index loc(#loc65)
    %50 = arith.select %4, %49, %48 : index loc(#loc65)
    %51 = handshake.mux %50 [%falseResult_22, %trueResult] : index, none loc(#loc65)
    %52 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc65)
    %trueResult_23, %falseResult_24 = handshake.cond_br %15, %52 : none loc(#loc72)
    %53 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_27 : i1, none, none -> none loc(#loc72)
    %trueResult_25, %falseResult_26 = handshake.cond_br %23, %53 : none loc(#loc76)
    %54 = handshake.constant %53 {value = 0 : index} : index loc(#loc76)
    %55 = handshake.constant %53 {value = 1 : index} : index loc(#loc76)
    %56 = arith.select %23, %55, %54 : index loc(#loc76)
    %57 = handshake.mux %56 [%41#1, %trueResult_25] : index, none loc(#loc76)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_3, %57 : none loc(#loc72)
    %58 = handshake.constant %52 {value = 0 : index} : index loc(#loc72)
    %59 = handshake.constant %52 {value = 1 : index} : index loc(#loc72)
    %60 = arith.select %15, %59, %58 : index loc(#loc72)
    %61 = handshake.mux %60 [%falseResult_28, %trueResult_23] : index, none loc(#loc72)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %61 : none loc(#loc65)
    %62 = handshake.mux %50 [%falseResult_30, %trueResult] : index, none loc(#loc65)
    %63 = dataflow.carry %willContinue, %falseResult, %trueResult_39 : i1, none, none -> none loc(#loc65)
    %trueResult_31, %falseResult_32 = handshake.cond_br %15, %63 : none loc(#loc72)
    %64 = dataflow.carry %willContinue_3, %falseResult_32, %trueResult_37 : i1, none, none -> none loc(#loc72)
    %trueResult_33, %falseResult_34 = handshake.cond_br %23, %64 : none loc(#loc76)
    %65 = dataflow.carry %willContinue_15, %falseResult_34, %trueResult_35 : i1, none, none -> none loc(#loc76)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_15, %45#1 : none loc(#loc76)
    %66 = handshake.constant %64 {value = 0 : index} : index loc(#loc76)
    %67 = handshake.constant %64 {value = 1 : index} : index loc(#loc76)
    %68 = arith.select %23, %67, %66 : index loc(#loc76)
    %69 = handshake.mux %68 [%falseResult_36, %trueResult_33] : index, none loc(#loc76)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_3, %69 : none loc(#loc72)
    %70 = handshake.constant %63 {value = 0 : index} : index loc(#loc72)
    %71 = handshake.constant %63 {value = 1 : index} : index loc(#loc72)
    %72 = arith.select %15, %71, %70 : index loc(#loc72)
    %73 = handshake.mux %72 [%falseResult_38, %trueResult_31] : index, none loc(#loc72)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue, %73 : none loc(#loc65)
    %74 = handshake.mux %50 [%falseResult_40, %trueResult] : index, none loc(#loc65)
    %75 = dataflow.carry %willContinue, %falseResult, %trueResult_47 : i1, none, none -> none loc(#loc65)
    %trueResult_41, %falseResult_42 = handshake.cond_br %15, %75 : none loc(#loc72)
    %76 = dataflow.carry %willContinue_3, %falseResult_42, %trueResult_45 : i1, none, none -> none loc(#loc72)
    %trueResult_43, %falseResult_44 = handshake.cond_br %23, %76 : none loc(#loc76)
    %77 = handshake.constant %76 {value = 0 : index} : index loc(#loc76)
    %78 = handshake.constant %76 {value = 1 : index} : index loc(#loc76)
    %79 = arith.select %23, %78, %77 : index loc(#loc76)
    %80 = handshake.mux %79 [%43#1, %trueResult_43] : index, none loc(#loc76)
    %trueResult_45, %falseResult_46 = handshake.cond_br %willContinue_3, %80 : none loc(#loc72)
    %81 = handshake.constant %75 {value = 0 : index} : index loc(#loc72)
    %82 = handshake.constant %75 {value = 1 : index} : index loc(#loc72)
    %83 = arith.select %15, %82, %81 : index loc(#loc72)
    %84 = handshake.mux %83 [%falseResult_46, %trueResult_41] : index, none loc(#loc72)
    %trueResult_47, %falseResult_48 = handshake.cond_br %willContinue, %84 : none loc(#loc65)
    %85 = handshake.mux %50 [%falseResult_48, %trueResult] : index, none loc(#loc65)
    %86 = dataflow.carry %willContinue, %falseResult, %trueResult_55 : i1, none, none -> none loc(#loc65)
    %trueResult_49, %falseResult_50 = handshake.cond_br %15, %86 : none loc(#loc72)
    %87 = dataflow.carry %willContinue_3, %falseResult_50, %trueResult_53 : i1, none, none -> none loc(#loc72)
    %trueResult_51, %falseResult_52 = handshake.cond_br %23, %87 : none loc(#loc76)
    %88 = handshake.constant %87 {value = 0 : index} : index loc(#loc76)
    %89 = handshake.constant %87 {value = 1 : index} : index loc(#loc76)
    %90 = arith.select %23, %89, %88 : index loc(#loc76)
    %91 = handshake.mux %90 [%42#1, %trueResult_51] : index, none loc(#loc76)
    %trueResult_53, %falseResult_54 = handshake.cond_br %willContinue_3, %91 : none loc(#loc72)
    %92 = handshake.constant %86 {value = 0 : index} : index loc(#loc72)
    %93 = handshake.constant %86 {value = 1 : index} : index loc(#loc72)
    %94 = arith.select %15, %93, %92 : index loc(#loc72)
    %95 = handshake.mux %94 [%falseResult_54, %trueResult_49] : index, none loc(#loc72)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue, %95 : none loc(#loc65)
    %96 = handshake.mux %50 [%falseResult_56, %trueResult] : index, none loc(#loc65)
    %97 = dataflow.carry %willContinue, %falseResult, %trueResult_65 : i1, none, none -> none loc(#loc65)
    %trueResult_57, %falseResult_58 = handshake.cond_br %15, %97 : none loc(#loc72)
    %98 = dataflow.carry %willContinue_3, %falseResult_58, %trueResult_63 : i1, none, none -> none loc(#loc72)
    %trueResult_59, %falseResult_60 = handshake.cond_br %23, %98 : none loc(#loc76)
    %99 = dataflow.carry %willContinue_15, %falseResult_60, %trueResult_61 : i1, none, none -> none loc(#loc76)
    %trueResult_61, %falseResult_62 = handshake.cond_br %willContinue_15, %46 : none loc(#loc76)
    %100 = handshake.constant %98 {value = 0 : index} : index loc(#loc76)
    %101 = handshake.constant %98 {value = 1 : index} : index loc(#loc76)
    %102 = arith.select %23, %101, %100 : index loc(#loc76)
    %103 = handshake.mux %102 [%falseResult_62, %trueResult_59] : index, none loc(#loc76)
    %trueResult_63, %falseResult_64 = handshake.cond_br %willContinue_3, %103 : none loc(#loc72)
    %104 = handshake.constant %97 {value = 0 : index} : index loc(#loc72)
    %105 = handshake.constant %97 {value = 1 : index} : index loc(#loc72)
    %106 = arith.select %15, %105, %104 : index loc(#loc72)
    %107 = handshake.mux %106 [%falseResult_64, %trueResult_57] : index, none loc(#loc72)
    %trueResult_65, %falseResult_66 = handshake.cond_br %willContinue, %107 : none loc(#loc65)
    %108 = handshake.mux %50 [%falseResult_66, %trueResult] : index, none loc(#loc65)
    %109 = handshake.join %51, %62, %74, %85, %96, %108 : none, none, none, none, none, none loc(#loc62)
    handshake.return %109 : none loc(#loc63)
  } loc(#loc62)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc38)
    %false = arith.constant false loc(#loc38)
    %0 = seq.const_clock  low loc(#loc38)
    %c2_i32 = arith.constant 2 : i32 loc(#loc38)
    %1 = ub.poison : i64 loc(#loc38)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c-50_i32 = arith.constant -50 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %cst_0 = arith.constant 2.000000e+00 : f32 loc(#loc2)
    %cst_1 = arith.constant 1.000000e+00 : f32 loc(#loc2)
    %c4_i64 = arith.constant 4 : i64 loc(#loc2)
    %c4_i32 = arith.constant 4 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %cst_2 = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %cst_3 = arith.constant 9.99999974E-5 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<18xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<18xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<4xf32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<4xf32> loc(#loc2)
    %alloca_6 = memref.alloca() : memref<4xf32> loc(#loc2)
    %alloca_7 = memref.alloca() : memref<4xf32> loc(#loc2)
    %alloca_8 = memref.alloca() : memref<256xf32> loc(#loc2)
    %alloca_9 = memref.alloca() : memref<256xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc49)
      %12 = arith.remui %11, %c100_i32 : i32 loc(#loc49)
      %13 = arith.addi %12, %c-50_i32 : i32 loc(#loc49)
      %14 = arith.sitofp %13 : i32 to f32 loc(#loc49)
      %15 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      memref.store %14, %alloca[%15] : memref<256xf32> loc(#loc49)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc46)
      %17 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc50)
      scf.condition(%17) %16 : i64 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc22])):
      scf.yield %arg0 : i64 loc(#loc43)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    scf.for %arg0 = %c0 to %c4 step %c1 {
      memref.store %cst, %alloca_7[%arg0] : memref<4xf32> loc(#loc51)
    } loc(#loc51)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc52)
      %12 = arith.muli %11, %c10_i32 : i32 loc(#loc52)
      %13 = arith.uitofp %12 : i32 to f32 loc(#loc52)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      memref.store %13, %alloca_4[%14] : memref<4xf32> loc(#loc52)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc53)
      %16 = arith.trunci %15 : i64 to i32 loc(#loc53)
      %17 = arith.uitofp %16 : i32 to f32 loc(#loc53)
      %18 = arith.mulf %17, %cst_0 : f32 loc(#loc53)
      memref.store %18, %alloca_5[%14] : memref<4xf32> loc(#loc53)
      memref.store %cst_1, %alloca_6[%14] : memref<4xf32> loc(#loc54)
      %19 = arith.cmpi ne, %15, %c4_i64 : i64 loc(#loc55)
      scf.condition(%19) %15 : i64 loc(#loc44)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block16>[#loc25])):
      scf.yield %arg0 : i64 loc(#loc44)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc44)
    %cast = memref.cast %alloca : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %cast_10 = memref.cast %alloca_4 : memref<4xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %cast_11 = memref.cast %alloca_5 : memref<4xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %cast_12 = memref.cast %alloca_6 : memref<4xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %cast_13 = memref.cast %alloca_7 : memref<4xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %cast_14 = memref.cast %alloca_8 : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    call @_Z13batchnorm_cpuPKfS0_S0_S0_S0_Pfjjjf(%cast, %cast_10, %cast_11, %cast_12, %cast_13, %cast_14, %c4_i32, %c8_i32, %c8_i32, %cst_2) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32, i32, f32) -> () loc(#loc39)
    %cast_15 = memref.cast %alloca_9 : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_16, %ready_17 = esi.wrap.vr %cast_10, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_18, %ready_19 = esi.wrap.vr %cast_11, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_20, %ready_21 = esi.wrap.vr %cast_12, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_22, %ready_23 = esi.wrap.vr %cast_13, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_24, %ready_25 = esi.wrap.vr %cast_15, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_26, %ready_27 = esi.wrap.vr %c4_i32, %true : i32 loc(#loc40)
    %chanOutput_28, %ready_29 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc40)
    %chanOutput_30, %ready_31 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc40)
    %chanOutput_32, %ready_33 = esi.wrap.vr %cst_2, %true : f32 loc(#loc40)
    %chanOutput_34, %ready_35 = esi.wrap.vr %true, %true : i1 loc(#loc40)
    %6 = handshake.esi_instance @_Z13batchnorm_dsaPKfS0_S0_S0_S0_Pfjjjf_esi "_Z13batchnorm_dsaPKfS0_S0_S0_S0_Pfjjjf_inst0" clk %0 rst %false(%chanOutput, %chanOutput_16, %chanOutput_18, %chanOutput_20, %chanOutput_22, %chanOutput_24, %chanOutput_26, %chanOutput_28, %chanOutput_30, %chanOutput_32, %chanOutput_34) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<f32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc40)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc40)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %12 = memref.load %alloca_8[%11] : memref<256xf32> loc(#loc57)
      %13 = memref.load %alloca_9[%11] : memref<256xf32> loc(#loc57)
      %14 = arith.subf %12, %13 : f32 loc(#loc57)
      %15 = math.absf %14 : f32 loc(#loc57)
      %16 = arith.cmpf ule, %15, %cst_3 : f32 loc(#loc57)
      %17:3 = scf.if %16 -> (i64, i32, i32) {
        %19 = arith.addi %arg0, %c1_i64 : i64 loc(#loc48)
        %20 = arith.cmpi eq, %19, %c256_i64 : i64 loc(#loc48)
        %21 = arith.extui %20 : i1 to i32 loc(#loc45)
        %22 = arith.cmpi ne, %19, %c256_i64 : i64 loc(#loc56)
        %23 = arith.extui %22 : i1 to i32 loc(#loc45)
        scf.yield %19, %21, %23 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %18 = arith.trunci %17#2 : i32 to i1 loc(#loc45)
      scf.condition(%18) %17#0, %16, %17#1 : i64, i1, i32 loc(#loc45)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block17>[#loc31]), %arg1: i1 loc(fused<#di_lexical_block17>[#loc31]), %arg2: i32 loc(fused<#di_lexical_block17>[#loc31])):
      scf.yield %arg0 : i64 loc(#loc45)
    } loc(#loc45)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc45)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc45)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<18xi8> -> index loc(#loc58)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc58)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc58)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc58)
      scf.yield %c1_i32 : i32 loc(#loc59)
    } loc(#loc45)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<18xi8> -> index loc(#loc41)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc41)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc41)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc41)
    } loc(#loc2)
    return %10 : i32 loc(#loc42)
  } loc(#loc38)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memset.p0.i64(memref<?xi8, strided<[1], offset: ?>>, i8, i64, i1) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#di_file1 = #llvm.di_file<"/usr/include/bits/mathcalls.h" in "">
#loc = loc("tests/app/batchnorm/batchnorm.cpp":0:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/batchnorm/batchnorm.cpp":28:0)
#loc7 = loc("tests/app/batchnorm/batchnorm.cpp":32:0)
#loc8 = loc("tests/app/batchnorm/batchnorm.cpp":33:0)
#loc9 = loc("tests/app/batchnorm/batchnorm.cpp":34:0)
#loc10 = loc("tests/app/batchnorm/batchnorm.cpp":38:0)
#loc11 = loc("/usr/include/bits/mathcalls.h":143:0)
#loc13 = loc("tests/app/batchnorm/batchnorm.cpp":54:0)
#loc14 = loc("tests/app/batchnorm/batchnorm.cpp":55:0)
#loc15 = loc("tests/app/batchnorm/batchnorm.cpp":57:0)
#loc16 = loc("tests/app/batchnorm/batchnorm.cpp":58:0)
#loc17 = loc("tests/app/batchnorm/batchnorm.cpp":59:0)
#loc18 = loc("tests/app/batchnorm/batchnorm.cpp":60:0)
#loc19 = loc("tests/app/batchnorm/batchnorm.cpp":61:0)
#loc20 = loc("tests/app/batchnorm/batchnorm.cpp":65:0)
#loc21 = loc("tests/app/batchnorm/main.cpp":6:0)
#loc23 = loc("tests/app/batchnorm/main.cpp":27:0)
#loc24 = loc("tests/app/batchnorm/main.cpp":35:0)
#loc26 = loc("tests/app/batchnorm/main.cpp":32:0)
#loc27 = loc("tests/app/batchnorm/main.cpp":33:0)
#loc28 = loc("tests/app/batchnorm/main.cpp":34:0)
#loc29 = loc("tests/app/batchnorm/main.cpp":39:0)
#loc30 = loc("tests/app/batchnorm/main.cpp":42:0)
#loc32 = loc("tests/app/batchnorm/main.cpp":46:0)
#loc33 = loc("tests/app/batchnorm/main.cpp":47:0)
#loc34 = loc("tests/app/batchnorm/main.cpp":48:0)
#loc35 = loc("tests/app/batchnorm/main.cpp":52:0)
#loc36 = loc("tests/app/batchnorm/main.cpp":54:0)
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type, #di_basic_type>
#di_subprogram3 = #llvm.di_subprogram<scope = #di_file1, name = "sqrtf", file = #di_file1, line = 143, subprogramFlags = Optimized, type = #di_subroutine_type>
#loc37 = loc(fused<#di_subprogram3>[#loc11])
#loc38 = loc(fused<#di_subprogram4>[#loc21])
#loc39 = loc(fused<#di_subprogram4>[#loc29])
#loc40 = loc(fused<#di_subprogram4>[#loc30])
#loc41 = loc(fused<#di_subprogram4>[#loc35])
#loc42 = loc(fused<#di_subprogram4>[#loc36])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file2, line = 26>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file2, line = 31>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file2, line = 45>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file2, line = 26>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file2, line = 31>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file2, line = 45>
#loc46 = loc(fused<#di_lexical_block20>[#loc22])
#loc47 = loc(fused<#di_lexical_block21>[#loc25])
#loc48 = loc(fused<#di_lexical_block22>[#loc31])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file2, line = 46>
#loc49 = loc(fused<#di_lexical_block25>[#loc23])
#loc50 = loc(fused[#loc43, #loc46])
#loc51 = loc(fused<#di_lexical_block26>[#loc24])
#loc52 = loc(fused<#di_lexical_block26>[#loc26])
#loc53 = loc(fused<#di_lexical_block26>[#loc27])
#loc54 = loc(fused<#di_lexical_block26>[#loc28])
#loc55 = loc(fused[#loc44, #loc47])
#loc56 = loc(fused[#loc45, #loc48])
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file2, line = 46>
#loc57 = loc(fused<#di_lexical_block30>[#loc32])
#loc58 = loc(fused<#di_lexical_block31>[#loc33])
#loc59 = loc(fused<#di_lexical_block31>[#loc34])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 54>
#loc61 = loc(fused<#di_subprogram5>[#loc10])
#loc63 = loc(fused<#di_subprogram6>[#loc20])
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 54>
#loc65 = loc(fused<#di_lexical_block33>[#loc13])
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 54>
#loc66 = loc(fused<#di_lexical_block34>[#loc3])
#loc67 = loc(fused<#di_lexical_block35>[#loc13])
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file, line = 57>
#loc68 = loc(fused<#di_lexical_block36>[#loc4])
#loc69 = loc(fused[#loc64, #loc66])
#loc70 = loc(fused<#di_lexical_block37>[#loc14])
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file, line = 57>
#loc72 = loc(fused<#di_lexical_block39>[#loc15])
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file, line = 57>
#loc73 = loc(fused<#di_lexical_block40>[#loc5])
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_lexical_block43, file = #di_file, line = 58>
#loc74 = loc(fused[#loc71, #loc73])
#di_lexical_block46 = #llvm.di_lexical_block<scope = #di_lexical_block44, file = #di_file, line = 31>
#di_lexical_block47 = #llvm.di_lexical_block<scope = #di_lexical_block45, file = #di_file, line = 58>
#loc76 = loc(fused<#di_lexical_block45>[#loc16])
#di_lexical_block48 = #llvm.di_lexical_block<scope = #di_lexical_block46, file = #di_file, line = 31>
#di_lexical_block49 = #llvm.di_lexical_block<scope = #di_lexical_block47, file = #di_file, line = 58>
#loc77 = loc(fused<#di_lexical_block46>[#loc6])
#loc78 = loc(fused<#di_lexical_block48>[#loc7])
#loc79 = loc(fused<#di_lexical_block48>[#loc8])
#loc80 = loc(fused<#di_lexical_block48>[#loc9])
#loc81 = loc(fused[#loc75, #loc77])
#loc82 = loc(fused<#di_lexical_block49>[#loc17])
#loc83 = loc(fused<#di_lexical_block49>[#loc18])
#loc84 = loc(fused<#di_lexical_block49>[#loc19])
