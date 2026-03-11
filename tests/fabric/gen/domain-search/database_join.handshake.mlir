#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/database_join/database_join.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/database_join/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/database_join/database_join.cpp":16:0)
#loc3 = loc("tests/app/database_join/database_join.cpp":26:0)
#loc4 = loc("tests/app/database_join/database_join.cpp":27:0)
#loc12 = loc("tests/app/database_join/database_join.cpp":42:0)
#loc29 = loc("tests/app/database_join/main.cpp":41:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__int32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 26>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 54>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 41>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "int32_t", baseType = #di_derived_type1>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 26>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 54>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type4, sizeInBits = 96, elements = #llvm.di_subrange<count = 3 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type4, sizeInBits = 160, elements = #llvm.di_subrange<count = 5 : i64>>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type4>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type3>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 26>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 54>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "out_idx", file = #di_file, line = 25, type = #di_derived_type3>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 26, type = #di_derived_type3>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "out_idx", file = #di_file, line = 51, type = #di_derived_type3>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 54, type = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_count", file = #di_file1, line = 26, type = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_count", file = #di_file1, line = 30, type = #di_derived_type3>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 41, type = #di_derived_type3>
#di_derived_type10 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type6, sizeInBits = 64>
#di_derived_type11 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type7>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 27>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 55>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "size_a", file = #di_file, line = 23, arg = 8, type = #di_derived_type8>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "size_b", file = #di_file, line = 24, arg = 9, type = #di_derived_type8>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "size_a", file = #di_file, line = 49, arg = 8, type = #di_derived_type8>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "size_b", file = #di_file, line = 50, arg = 9, type = #di_derived_type8>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "size_a", file = #di_file1, line = 8, type = #di_derived_type8>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "size_b", file = #di_file1, line = 9, type = #di_derived_type8>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "max_output", file = #di_file1, line = 10, type = #di_derived_type8>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "a_ids", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "b_ids", file = #di_file1, line = 13, type = #di_composite_type>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "a_values", file = #di_file1, line = 14, type = #di_composite_type>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "b_values", file = #di_file1, line = 15, type = #di_composite_type>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_ids", file = #di_file1, line = 18, type = #di_composite_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_a_values", file = #di_file1, line = 19, type = #di_composite_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_b_values", file = #di_file1, line = 20, type = #di_composite_type1>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_ids", file = #di_file1, line = 21, type = #di_composite_type1>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_a_values", file = #di_file1, line = 22, type = #di_composite_type1>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_b_values", file = #di_file1, line = 23, type = #di_composite_type1>
#di_derived_type12 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type10>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_ids", file = #di_file, line = 20, arg = 5, type = #di_derived_type11>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_a_values", file = #di_file, line = 21, arg = 6, type = #di_derived_type11>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_b_values", file = #di_file, line = 22, arg = 7, type = #di_derived_type11>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "j", file = #di_file, line = 27, type = #di_derived_type3>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_ids", file = #di_file, line = 46, arg = 5, type = #di_derived_type11>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_a_values", file = #di_file, line = 47, arg = 6, type = #di_derived_type11>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_b_values", file = #di_file, line = 48, arg = 7, type = #di_derived_type11>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file, line = 55, type = #di_derived_type3>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable11, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable22, #di_local_variable23, #di_local_variable4, #di_local_variable5, #di_local_variable6>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 41>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram, name = "a_ids", file = #di_file, line = 16, arg = 1, type = #di_derived_type12>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram, name = "b_ids", file = #di_file, line = 17, arg = 2, type = #di_derived_type12>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_subprogram, name = "a_values", file = #di_file, line = 18, arg = 3, type = #di_derived_type12>
#di_local_variable35 = #llvm.di_local_variable<scope = #di_subprogram, name = "b_values", file = #di_file, line = 19, arg = 4, type = #di_derived_type12>
#di_local_variable36 = #llvm.di_local_variable<scope = #di_subprogram1, name = "a_ids", file = #di_file, line = 42, arg = 1, type = #di_derived_type12>
#di_local_variable37 = #llvm.di_local_variable<scope = #di_subprogram1, name = "b_ids", file = #di_file, line = 43, arg = 2, type = #di_derived_type12>
#di_local_variable38 = #llvm.di_local_variable<scope = #di_subprogram1, name = "a_values", file = #di_file, line = 44, arg = 3, type = #di_derived_type12>
#di_local_variable39 = #llvm.di_local_variable<scope = #di_subprogram1, name = "b_values", file = #di_file, line = 45, arg = 4, type = #di_derived_type12>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_derived_type3, #di_derived_type12, #di_derived_type12, #di_derived_type12, #di_derived_type12, #di_derived_type11, #di_derived_type11, #di_derived_type11, #di_derived_type8, #di_derived_type8>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "database_join_cpu", linkageName = "_Z17database_join_cpuPKiS0_S0_S0_PiS1_S1_jj", file = #di_file, line = 16, scopeLine = 24, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable32, #di_local_variable33, #di_local_variable34, #di_local_variable35, #di_local_variable24, #di_local_variable25, #di_local_variable26, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1, #di_local_variable27>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "database_join_dsa", linkageName = "_Z17database_join_dsaPKiS0_S0_S0_PiS1_S1_jj", file = #di_file, line = 42, scopeLine = 50, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable36, #di_local_variable37, #di_local_variable38, #di_local_variable39, #di_local_variable28, #di_local_variable29, #di_local_variable30, #di_local_variable9, #di_local_variable10, #di_local_variable2, #di_local_variable3, #di_local_variable31>
#loc51 = loc(fused<#di_lexical_block10>[#loc29])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 26>
#loc52 = loc(fused<#di_subprogram5>[#loc1])
#loc54 = loc(fused<#di_subprogram6>[#loc12])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 26>
#loc59 = loc(fused<#di_lexical_block13>[#loc3])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 26>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 27>
#loc70 = loc(fused<#di_lexical_block22>[#loc4])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<42xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 100, 97, 116, 97, 98, 97, 115, 101, 95, 106, 111, 105, 110, 47, 100, 97, 116, 97, 98, 97, 115, 101, 95, 106, 111, 105, 110, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @__const.main.a_ids : memref<3xi32> = dense<[1, 2, 3]> {alignment = 4 : i64} loc(#loc)
  memref.global constant @__const.main.b_ids : memref<3xi32> = dense<[2, 3, 4]> {alignment = 4 : i64} loc(#loc)
  memref.global constant @__const.main.a_values : memref<3xi32> = dense<[10, 20, 30]> {alignment = 4 : i64} loc(#loc)
  memref.global constant @__const.main.b_values : memref<3xi32> = dense<[200, 300, 400]> {alignment = 4 : i64} loc(#loc)
  memref.global constant @".str.2" : memref<61xi8> = dense<[100, 97, 116, 97, 98, 97, 115, 101, 95, 106, 111, 105, 110, 58, 32, 70, 65, 73, 76, 69, 68, 32, 40, 99, 111, 117, 110, 116, 32, 109, 105, 115, 109, 97, 116, 99, 104, 58, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 37, 117, 44, 32, 103, 111, 116, 32, 37, 117, 41, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.1.3" : memref<46xi8> = dense<[100, 97, 116, 97, 98, 97, 115, 101, 95, 106, 111, 105, 110, 58, 32, 70, 65, 73, 76, 69, 68, 32, 40, 109, 105, 115, 109, 97, 116, 99, 104, 32, 97, 116, 32, 105, 110, 100, 101, 120, 32, 37, 117, 41, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str : memref<22xi8> = dense<[100, 97, 116, 97, 98, 97, 115, 101, 95, 106, 111, 105, 110, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z17database_join_cpuPKiS0_S0_S0_PiS1_S1_jj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg3: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg4: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg5: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg6: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc1]), %arg7: i32 loc(fused<#di_subprogram5>[#loc1]), %arg8: i32 loc(fused<#di_subprogram5>[#loc1])) -> i32 {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc62)
    %1 = scf.if %0 -> (i32) {
      scf.yield %c0_i32 : i32 loc(#loc59)
    } else {
      %2 = arith.cmpi eq, %arg8, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg7 : i32 to i64 loc(#loc62)
      %4 = arith.extui %arg8 : i32 to i64 loc(#loc2)
      %5:2 = scf.while (%arg9 = %c0_i64, %arg10 = %c0_i32) : (i64, i32) -> (i64, i32) {
        %6 = scf.if %2 -> (i32) {
          scf.yield %arg10 : i32 loc(#loc70)
        } else {
          %9 = arith.index_cast %arg9 : i64 to index loc(#loc2)
          %10 = memref.load %arg0[%9] : memref<?xi32, strided<[1], offset: ?>> loc(#loc2)
          %11:2 = scf.while (%arg11 = %c0_i64, %arg12 = %arg10) : (i64, i32) -> (i64, i32) {
            %12 = arith.index_cast %arg11 : i64 to index loc(#loc74)
            %13 = memref.load %arg1[%12] : memref<?xi32, strided<[1], offset: ?>> loc(#loc74)
            %14 = arith.cmpi eq, %10, %13 : i32 loc(#loc74)
            %15 = scf.if %14 -> (i32) {
              %18 = arith.extui %arg12 : i32 to i64 loc(#loc76)
              %19 = arith.index_cast %18 : i64 to index loc(#loc76)
              memref.store %10, %arg4[%19] : memref<?xi32, strided<[1], offset: ?>> loc(#loc76)
              %20 = memref.load %arg2[%9] : memref<?xi32, strided<[1], offset: ?>> loc(#loc77)
              memref.store %20, %arg5[%19] : memref<?xi32, strided<[1], offset: ?>> loc(#loc77)
              %21 = memref.load %arg3[%12] : memref<?xi32, strided<[1], offset: ?>> loc(#loc78)
              memref.store %21, %arg6[%19] : memref<?xi32, strided<[1], offset: ?>> loc(#loc78)
              %22 = arith.addi %arg12, %c1_i32 : i32 loc(#loc79)
              scf.yield %22 : i32 loc(#loc80)
            } else {
              scf.yield %arg12 : i32 loc(#loc74)
            } loc(#loc74)
            %16 = arith.addi %arg11, %c1_i64 : i64 loc(#loc72)
            %17 = arith.cmpi ne, %16, %4 : i64 loc(#loc73)
            scf.condition(%17) %16, %15 : i64, i32 loc(#loc70)
          } do {
          ^bb0(%arg11: i64 loc(fused<#di_lexical_block22>[#loc4]), %arg12: i32 loc(fused<#di_lexical_block22>[#loc4])):
            scf.yield %arg11, %arg12 : i64, i32 loc(#loc70)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc70)
          scf.yield %11#1 : i32 loc(#loc70)
        } loc(#loc70)
        %7 = arith.addi %arg9, %c1_i64 : i64 loc(#loc62)
        %8 = arith.cmpi ne, %7, %3 : i64 loc(#loc67)
        scf.condition(%8) %7, %6 : i64, i32 loc(#loc59)
      } do {
      ^bb0(%arg9: i64 loc(fused<#di_lexical_block13>[#loc3]), %arg10: i32 loc(fused<#di_lexical_block13>[#loc3])):
        scf.yield %arg9, %arg10 : i64, i32 loc(#loc59)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc59)
      scf.yield %5#1 : i32 loc(#loc59)
    } loc(#loc59)
    return %1 : i32 loc(#loc53)
  } loc(#loc52)
  handshake.func @_Z17database_join_dsaPKiS0_S0_S0_PiS1_S1_jj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg6: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg7: i32 loc(fused<#di_subprogram6>[#loc12]), %arg8: i32 loc(fused<#di_subprogram6>[#loc12]), %arg9: i1 loc(fused<#di_subprogram6>[#loc12]), ...) -> (i32, i1) attributes {argNames = ["a_ids", "b_ids", "a_values", "b_values", "output_ids", "output_a_values", "output_b_values", "size_a", "size_b", "start_token"], loom.annotations = ["loom.accel"], resNames = ["out_idx", "done_token"]} {
    %0 = handshake.join %arg9 : i1 loc(#loc54)
    %1 = handshake.join %0 : none loc(#loc54)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg7, %3 : i32 loc(#loc63)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc60)
    %6 = arith.cmpi eq, %arg8, %3 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc60)
    %8 = arith.index_cast %4 : i64 to index loc(#loc60)
    %9 = arith.index_cast %arg7 : i32 to index loc(#loc60)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc60)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc60)
    %10 = dataflow.carry %willContinue, %3, %30 : i1, i32, i32 -> i32 loc(#loc60)
    %afterValue_0, %afterCond_1 = dataflow.gate %10, %willContinue : i32, i1 -> i32, i1 loc(#loc60)
    handshake.sink %afterCond_1 : i1 loc(#loc60)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %10 : i32 loc(#loc60)
    %11 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc60)
    %12 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc71)
    %trueResult_4, %falseResult_5 = handshake.cond_br %12, %11 : none loc(#loc71)
    %dataResult, %addressResults = handshake.load [%afterValue] %37#0, %falseResult_26 : index, i32 loc(#loc2)
    %13 = handshake.constant %falseResult_5 {value = 1 : index} : index loc(#loc71)
    %14 = arith.index_cast %arg8 : i32 to index loc(#loc71)
    %index_6, %willContinue_7 = dataflow.stream %8, %13, %14 {step_op = "+=", stop_cond = "!="} loc(#loc71)
    %afterValue_8, %afterCond_9 = dataflow.gate %index_6, %willContinue_7 : index, i1 -> index, i1 loc(#loc71)
    %15 = dataflow.carry %willContinue_7, %afterValue_0, %26 : i1, i32, i32 -> i32 loc(#loc71)
    %afterValue_10, %afterCond_11 = dataflow.gate %15, %willContinue_7 : i32, i1 -> i32, i1 loc(#loc71)
    handshake.sink %afterCond_11 : i1 loc(#loc71)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_7, %15 : i32 loc(#loc71)
    %16 = dataflow.invariant %afterCond_9, %falseResult_5 : i1, none -> none loc(#loc71)
    %dataResult_14, %addressResults_15 = handshake.load [%afterValue_8] %35#0, %82 : index, i32 loc(#loc75)
    %17 = dataflow.invariant %afterCond_9, %dataResult : i1, i32 -> i32 loc(#loc75)
    %18 = arith.cmpi eq, %17, %dataResult_14 : i32 loc(#loc75)
    %19 = arith.extui %afterValue_10 : i32 to i64 loc(#loc81)
    %20 = arith.index_cast %19 : i64 to index loc(#loc81)
    %dataResult_16, %addressResult = handshake.store [%20] %17, %trueResult_31 : index, i32 loc(#loc81)
    %21 = dataflow.invariant %afterCond_9, %afterValue : i1, index -> index loc(#loc82)
    %dataResult_17, %addressResults_18 = handshake.load [%21] %36#0, %trueResult_61 : index, i32 loc(#loc82)
    %dataResult_19, %addressResult_20 = handshake.store [%20] %dataResult_17, %trueResult_39 : index, i32 loc(#loc82)
    %dataResult_21, %addressResults_22 = handshake.load [%afterValue_8] %38#0, %trueResult_69 : index, i32 loc(#loc83)
    %dataResult_23, %addressResult_24 = handshake.store [%20] %dataResult_21, %trueResult_47 : index, i32 loc(#loc83)
    %22 = arith.addi %afterValue_10, %2 : i32 loc(#loc84)
    %23 = handshake.constant %16 {value = 0 : index} : index loc(#loc75)
    %24 = handshake.constant %16 {value = 1 : index} : index loc(#loc75)
    %25 = arith.select %18, %24, %23 : index loc(#loc75)
    %26 = handshake.mux %25 [%afterValue_10, %22] : index, i32 loc(#loc75)
    %27 = handshake.constant %11 {value = 0 : index} : index loc(#loc71)
    %28 = handshake.constant %11 {value = 1 : index} : index loc(#loc71)
    %29 = arith.select %12, %28, %27 : index loc(#loc71)
    %30 = handshake.mux %29 [%falseResult_13, %afterValue_0] : index, i32 loc(#loc71)
    %31 = handshake.constant %1 {value = 0 : index} : index loc(#loc60)
    %32 = handshake.constant %1 {value = 1 : index} : index loc(#loc60)
    %33 = arith.select %5, %32, %31 : index loc(#loc60)
    %34 = handshake.mux %33 [%falseResult_3, %3] : index, i32 loc(#loc60)
    %35:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_15) {id = 0 : i32} : (index) -> (i32, none) loc(#loc54)
    %36:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_18) {id = 1 : i32} : (index) -> (i32, none) loc(#loc54)
    %37:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 2 : i32} : (index) -> (i32, none) loc(#loc54)
    %38:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_22) {id = 3 : i32} : (index) -> (i32, none) loc(#loc54)
    %39 = handshake.extmemory[ld = 0, st = 1] (%arg6 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_23, %addressResult_24) {id = 4 : i32} : (i32, index) -> none loc(#loc54)
    %40 = handshake.extmemory[ld = 0, st = 1] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_16, %addressResult) {id = 5 : i32} : (i32, index) -> none loc(#loc54)
    %41 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_19, %addressResult_20) {id = 6 : i32} : (i32, index) -> none loc(#loc54)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc60)
    %trueResult_25, %falseResult_26 = handshake.cond_br %12, %42 : none loc(#loc71)
    %43 = handshake.constant %42 {value = 0 : index} : index loc(#loc71)
    %44 = handshake.constant %42 {value = 1 : index} : index loc(#loc71)
    %45 = arith.select %12, %44, %43 : index loc(#loc71)
    %46 = handshake.mux %45 [%37#1, %trueResult_25] : index, none loc(#loc71)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %46 : none loc(#loc60)
    %47 = handshake.mux %33 [%falseResult_28, %trueResult] : index, none loc(#loc60)
    %48 = dataflow.carry %willContinue, %falseResult, %trueResult_35 : i1, none, none -> none loc(#loc60)
    %trueResult_29, %falseResult_30 = handshake.cond_br %12, %48 : none loc(#loc71)
    %49 = dataflow.carry %willContinue_7, %falseResult_30, %trueResult_33 : i1, none, none -> none loc(#loc71)
    %trueResult_31, %falseResult_32 = handshake.cond_br %18, %49 : none loc(#loc75)
    %50 = handshake.constant %49 {value = 0 : index} : index loc(#loc75)
    %51 = handshake.constant %49 {value = 1 : index} : index loc(#loc75)
    %52 = arith.select %18, %51, %50 : index loc(#loc75)
    %53 = handshake.mux %52 [%falseResult_32, %40] : index, none loc(#loc75)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_7, %53 : none loc(#loc71)
    %54 = handshake.constant %48 {value = 0 : index} : index loc(#loc71)
    %55 = handshake.constant %48 {value = 1 : index} : index loc(#loc71)
    %56 = arith.select %12, %55, %54 : index loc(#loc71)
    %57 = handshake.mux %56 [%falseResult_34, %trueResult_29] : index, none loc(#loc71)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue, %57 : none loc(#loc60)
    %58 = handshake.mux %33 [%falseResult_36, %trueResult] : index, none loc(#loc60)
    %59 = dataflow.carry %willContinue, %falseResult, %trueResult_43 : i1, none, none -> none loc(#loc60)
    %trueResult_37, %falseResult_38 = handshake.cond_br %12, %59 : none loc(#loc71)
    %60 = dataflow.carry %willContinue_7, %falseResult_38, %trueResult_41 : i1, none, none -> none loc(#loc71)
    %trueResult_39, %falseResult_40 = handshake.cond_br %18, %60 : none loc(#loc75)
    %61 = handshake.constant %60 {value = 0 : index} : index loc(#loc75)
    %62 = handshake.constant %60 {value = 1 : index} : index loc(#loc75)
    %63 = arith.select %18, %62, %61 : index loc(#loc75)
    %64 = handshake.mux %63 [%falseResult_40, %41] : index, none loc(#loc75)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue_7, %64 : none loc(#loc71)
    %65 = handshake.constant %59 {value = 0 : index} : index loc(#loc71)
    %66 = handshake.constant %59 {value = 1 : index} : index loc(#loc71)
    %67 = arith.select %12, %66, %65 : index loc(#loc71)
    %68 = handshake.mux %67 [%falseResult_42, %trueResult_37] : index, none loc(#loc71)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue, %68 : none loc(#loc60)
    %69 = handshake.mux %33 [%falseResult_44, %trueResult] : index, none loc(#loc60)
    %70 = dataflow.carry %willContinue, %falseResult, %trueResult_51 : i1, none, none -> none loc(#loc60)
    %trueResult_45, %falseResult_46 = handshake.cond_br %12, %70 : none loc(#loc71)
    %71 = dataflow.carry %willContinue_7, %falseResult_46, %trueResult_49 : i1, none, none -> none loc(#loc71)
    %trueResult_47, %falseResult_48 = handshake.cond_br %18, %71 : none loc(#loc75)
    %72 = handshake.constant %71 {value = 0 : index} : index loc(#loc75)
    %73 = handshake.constant %71 {value = 1 : index} : index loc(#loc75)
    %74 = arith.select %18, %73, %72 : index loc(#loc75)
    %75 = handshake.mux %74 [%falseResult_48, %39] : index, none loc(#loc75)
    %trueResult_49, %falseResult_50 = handshake.cond_br %willContinue_7, %75 : none loc(#loc71)
    %76 = handshake.constant %70 {value = 0 : index} : index loc(#loc71)
    %77 = handshake.constant %70 {value = 1 : index} : index loc(#loc71)
    %78 = arith.select %12, %77, %76 : index loc(#loc71)
    %79 = handshake.mux %78 [%falseResult_50, %trueResult_45] : index, none loc(#loc71)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue, %79 : none loc(#loc60)
    %80 = handshake.mux %33 [%falseResult_52, %trueResult] : index, none loc(#loc60)
    %81 = dataflow.carry %willContinue, %falseResult, %trueResult_57 : i1, none, none -> none loc(#loc60)
    %trueResult_53, %falseResult_54 = handshake.cond_br %12, %81 : none loc(#loc71)
    %82 = dataflow.carry %willContinue_7, %falseResult_54, %trueResult_55 : i1, none, none -> none loc(#loc71)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue_7, %35#1 : none loc(#loc71)
    %83 = handshake.constant %81 {value = 0 : index} : index loc(#loc71)
    %84 = handshake.constant %81 {value = 1 : index} : index loc(#loc71)
    %85 = arith.select %12, %84, %83 : index loc(#loc71)
    %86 = handshake.mux %85 [%falseResult_56, %trueResult_53] : index, none loc(#loc71)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue, %86 : none loc(#loc60)
    %87 = handshake.mux %33 [%falseResult_58, %trueResult] : index, none loc(#loc60)
    %88 = dataflow.carry %willContinue, %falseResult, %trueResult_65 : i1, none, none -> none loc(#loc60)
    %trueResult_59, %falseResult_60 = handshake.cond_br %12, %88 : none loc(#loc71)
    %89 = dataflow.carry %willContinue_7, %falseResult_60, %trueResult_63 : i1, none, none -> none loc(#loc71)
    %trueResult_61, %falseResult_62 = handshake.cond_br %18, %89 : none loc(#loc75)
    %90 = handshake.constant %89 {value = 0 : index} : index loc(#loc75)
    %91 = handshake.constant %89 {value = 1 : index} : index loc(#loc75)
    %92 = arith.select %18, %91, %90 : index loc(#loc75)
    %93 = handshake.mux %92 [%falseResult_62, %36#1] : index, none loc(#loc75)
    %trueResult_63, %falseResult_64 = handshake.cond_br %willContinue_7, %93 : none loc(#loc71)
    %94 = handshake.constant %88 {value = 0 : index} : index loc(#loc71)
    %95 = handshake.constant %88 {value = 1 : index} : index loc(#loc71)
    %96 = arith.select %12, %95, %94 : index loc(#loc71)
    %97 = handshake.mux %96 [%falseResult_64, %trueResult_59] : index, none loc(#loc71)
    %trueResult_65, %falseResult_66 = handshake.cond_br %willContinue, %97 : none loc(#loc60)
    %98 = handshake.mux %33 [%falseResult_66, %trueResult] : index, none loc(#loc60)
    %99 = dataflow.carry %willContinue, %falseResult, %trueResult_73 : i1, none, none -> none loc(#loc60)
    %trueResult_67, %falseResult_68 = handshake.cond_br %12, %99 : none loc(#loc71)
    %100 = dataflow.carry %willContinue_7, %falseResult_68, %trueResult_71 : i1, none, none -> none loc(#loc71)
    %trueResult_69, %falseResult_70 = handshake.cond_br %18, %100 : none loc(#loc75)
    %101 = handshake.constant %100 {value = 0 : index} : index loc(#loc75)
    %102 = handshake.constant %100 {value = 1 : index} : index loc(#loc75)
    %103 = arith.select %18, %102, %101 : index loc(#loc75)
    %104 = handshake.mux %103 [%falseResult_70, %38#1] : index, none loc(#loc75)
    %trueResult_71, %falseResult_72 = handshake.cond_br %willContinue_7, %104 : none loc(#loc71)
    %105 = handshake.constant %99 {value = 0 : index} : index loc(#loc71)
    %106 = handshake.constant %99 {value = 1 : index} : index loc(#loc71)
    %107 = arith.select %12, %106, %105 : index loc(#loc71)
    %108 = handshake.mux %107 [%falseResult_72, %trueResult_67] : index, none loc(#loc71)
    %trueResult_73, %falseResult_74 = handshake.cond_br %willContinue, %108 : none loc(#loc60)
    %109 = handshake.mux %33 [%falseResult_74, %trueResult] : index, none loc(#loc60)
    %110 = handshake.join %47, %58, %69, %80, %87, %98, %109 : none, none, none, none, none, none, none loc(#loc54)
    %111 = handshake.constant %110 {value = true} : i1 loc(#loc54)
    handshake.return %34, %111 : i32, i1 loc(#loc54)
  } loc(#loc54)
  handshake.func @_Z17database_join_dsaPKiS0_S0_S0_PiS1_S1_jj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg3: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg6: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram6>[#loc12]), %arg7: i32 loc(fused<#di_subprogram6>[#loc12]), %arg8: i32 loc(fused<#di_subprogram6>[#loc12]), %arg9: none loc(fused<#di_subprogram6>[#loc12]), ...) -> (i32, none) attributes {argNames = ["a_ids", "b_ids", "a_values", "b_values", "output_ids", "output_a_values", "output_b_values", "size_a", "size_b", "start_token"], loom.annotations = ["loom.accel"], resNames = ["out_idx", "done_token"]} {
    %0 = handshake.join %arg9 : none loc(#loc54)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg7, %2 : i32 loc(#loc63)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc60)
    %5 = arith.cmpi eq, %arg8, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc60)
    %7 = arith.index_cast %3 : i64 to index loc(#loc60)
    %8 = arith.index_cast %arg7 : i32 to index loc(#loc60)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.no_parallel", "loom.loop.no_unroll"], step_op = "+=", stop_cond = "!="} loc(#loc60)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc60)
    %9 = dataflow.carry %willContinue, %2, %29 : i1, i32, i32 -> i32 loc(#loc60)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : i32, i1 -> i32, i1 loc(#loc60)
    handshake.sink %afterCond_1 : i1 loc(#loc60)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %9 : i32 loc(#loc60)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc60)
    %11 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc71)
    %trueResult_4, %falseResult_5 = handshake.cond_br %11, %10 : none loc(#loc71)
    %dataResult, %addressResults = handshake.load [%afterValue] %36#0, %falseResult_26 : index, i32 loc(#loc2)
    %12 = handshake.constant %falseResult_5 {value = 1 : index} : index loc(#loc71)
    %13 = arith.index_cast %arg8 : i32 to index loc(#loc71)
    %index_6, %willContinue_7 = dataflow.stream %7, %12, %13 {step_op = "+=", stop_cond = "!="} loc(#loc71)
    %afterValue_8, %afterCond_9 = dataflow.gate %index_6, %willContinue_7 : index, i1 -> index, i1 loc(#loc71)
    %14 = dataflow.carry %willContinue_7, %afterValue_0, %25 : i1, i32, i32 -> i32 loc(#loc71)
    %afterValue_10, %afterCond_11 = dataflow.gate %14, %willContinue_7 : i32, i1 -> i32, i1 loc(#loc71)
    handshake.sink %afterCond_11 : i1 loc(#loc71)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_7, %14 : i32 loc(#loc71)
    %15 = dataflow.invariant %afterCond_9, %falseResult_5 : i1, none -> none loc(#loc71)
    %dataResult_14, %addressResults_15 = handshake.load [%afterValue_8] %34#0, %81 : index, i32 loc(#loc75)
    %16 = dataflow.invariant %afterCond_9, %dataResult : i1, i32 -> i32 loc(#loc75)
    %17 = arith.cmpi eq, %16, %dataResult_14 : i32 loc(#loc75)
    %18 = arith.extui %afterValue_10 : i32 to i64 loc(#loc81)
    %19 = arith.index_cast %18 : i64 to index loc(#loc81)
    %dataResult_16, %addressResult = handshake.store [%19] %16, %trueResult_31 : index, i32 loc(#loc81)
    %20 = dataflow.invariant %afterCond_9, %afterValue : i1, index -> index loc(#loc82)
    %dataResult_17, %addressResults_18 = handshake.load [%20] %35#0, %trueResult_61 : index, i32 loc(#loc82)
    %dataResult_19, %addressResult_20 = handshake.store [%19] %dataResult_17, %trueResult_39 : index, i32 loc(#loc82)
    %dataResult_21, %addressResults_22 = handshake.load [%afterValue_8] %37#0, %trueResult_69 : index, i32 loc(#loc83)
    %dataResult_23, %addressResult_24 = handshake.store [%19] %dataResult_21, %trueResult_47 : index, i32 loc(#loc83)
    %21 = arith.addi %afterValue_10, %1 : i32 loc(#loc84)
    %22 = handshake.constant %15 {value = 0 : index} : index loc(#loc75)
    %23 = handshake.constant %15 {value = 1 : index} : index loc(#loc75)
    %24 = arith.select %17, %23, %22 : index loc(#loc75)
    %25 = handshake.mux %24 [%afterValue_10, %21] : index, i32 loc(#loc75)
    %26 = handshake.constant %10 {value = 0 : index} : index loc(#loc71)
    %27 = handshake.constant %10 {value = 1 : index} : index loc(#loc71)
    %28 = arith.select %11, %27, %26 : index loc(#loc71)
    %29 = handshake.mux %28 [%falseResult_13, %afterValue_0] : index, i32 loc(#loc71)
    %30 = handshake.constant %0 {value = 0 : index} : index loc(#loc60)
    %31 = handshake.constant %0 {value = 1 : index} : index loc(#loc60)
    %32 = arith.select %4, %31, %30 : index loc(#loc60)
    %33 = handshake.mux %32 [%falseResult_3, %2] : index, i32 loc(#loc60)
    %34:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_15) {id = 0 : i32} : (index) -> (i32, none) loc(#loc54)
    %35:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_18) {id = 1 : i32} : (index) -> (i32, none) loc(#loc54)
    %36:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 2 : i32} : (index) -> (i32, none) loc(#loc54)
    %37:2 = handshake.extmemory[ld = 1, st = 0] (%arg3 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_22) {id = 3 : i32} : (index) -> (i32, none) loc(#loc54)
    %38 = handshake.extmemory[ld = 0, st = 1] (%arg6 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_23, %addressResult_24) {id = 4 : i32} : (i32, index) -> none loc(#loc54)
    %39 = handshake.extmemory[ld = 0, st = 1] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_16, %addressResult) {id = 5 : i32} : (i32, index) -> none loc(#loc54)
    %40 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_19, %addressResult_20) {id = 6 : i32} : (i32, index) -> none loc(#loc54)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc60)
    %trueResult_25, %falseResult_26 = handshake.cond_br %11, %41 : none loc(#loc71)
    %42 = handshake.constant %41 {value = 0 : index} : index loc(#loc71)
    %43 = handshake.constant %41 {value = 1 : index} : index loc(#loc71)
    %44 = arith.select %11, %43, %42 : index loc(#loc71)
    %45 = handshake.mux %44 [%36#1, %trueResult_25] : index, none loc(#loc71)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %45 : none loc(#loc60)
    %46 = handshake.mux %32 [%falseResult_28, %trueResult] : index, none loc(#loc60)
    %47 = dataflow.carry %willContinue, %falseResult, %trueResult_35 : i1, none, none -> none loc(#loc60)
    %trueResult_29, %falseResult_30 = handshake.cond_br %11, %47 : none loc(#loc71)
    %48 = dataflow.carry %willContinue_7, %falseResult_30, %trueResult_33 : i1, none, none -> none loc(#loc71)
    %trueResult_31, %falseResult_32 = handshake.cond_br %17, %48 : none loc(#loc75)
    %49 = handshake.constant %48 {value = 0 : index} : index loc(#loc75)
    %50 = handshake.constant %48 {value = 1 : index} : index loc(#loc75)
    %51 = arith.select %17, %50, %49 : index loc(#loc75)
    %52 = handshake.mux %51 [%falseResult_32, %39] : index, none loc(#loc75)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_7, %52 : none loc(#loc71)
    %53 = handshake.constant %47 {value = 0 : index} : index loc(#loc71)
    %54 = handshake.constant %47 {value = 1 : index} : index loc(#loc71)
    %55 = arith.select %11, %54, %53 : index loc(#loc71)
    %56 = handshake.mux %55 [%falseResult_34, %trueResult_29] : index, none loc(#loc71)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue, %56 : none loc(#loc60)
    %57 = handshake.mux %32 [%falseResult_36, %trueResult] : index, none loc(#loc60)
    %58 = dataflow.carry %willContinue, %falseResult, %trueResult_43 : i1, none, none -> none loc(#loc60)
    %trueResult_37, %falseResult_38 = handshake.cond_br %11, %58 : none loc(#loc71)
    %59 = dataflow.carry %willContinue_7, %falseResult_38, %trueResult_41 : i1, none, none -> none loc(#loc71)
    %trueResult_39, %falseResult_40 = handshake.cond_br %17, %59 : none loc(#loc75)
    %60 = handshake.constant %59 {value = 0 : index} : index loc(#loc75)
    %61 = handshake.constant %59 {value = 1 : index} : index loc(#loc75)
    %62 = arith.select %17, %61, %60 : index loc(#loc75)
    %63 = handshake.mux %62 [%falseResult_40, %40] : index, none loc(#loc75)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue_7, %63 : none loc(#loc71)
    %64 = handshake.constant %58 {value = 0 : index} : index loc(#loc71)
    %65 = handshake.constant %58 {value = 1 : index} : index loc(#loc71)
    %66 = arith.select %11, %65, %64 : index loc(#loc71)
    %67 = handshake.mux %66 [%falseResult_42, %trueResult_37] : index, none loc(#loc71)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue, %67 : none loc(#loc60)
    %68 = handshake.mux %32 [%falseResult_44, %trueResult] : index, none loc(#loc60)
    %69 = dataflow.carry %willContinue, %falseResult, %trueResult_51 : i1, none, none -> none loc(#loc60)
    %trueResult_45, %falseResult_46 = handshake.cond_br %11, %69 : none loc(#loc71)
    %70 = dataflow.carry %willContinue_7, %falseResult_46, %trueResult_49 : i1, none, none -> none loc(#loc71)
    %trueResult_47, %falseResult_48 = handshake.cond_br %17, %70 : none loc(#loc75)
    %71 = handshake.constant %70 {value = 0 : index} : index loc(#loc75)
    %72 = handshake.constant %70 {value = 1 : index} : index loc(#loc75)
    %73 = arith.select %17, %72, %71 : index loc(#loc75)
    %74 = handshake.mux %73 [%falseResult_48, %38] : index, none loc(#loc75)
    %trueResult_49, %falseResult_50 = handshake.cond_br %willContinue_7, %74 : none loc(#loc71)
    %75 = handshake.constant %69 {value = 0 : index} : index loc(#loc71)
    %76 = handshake.constant %69 {value = 1 : index} : index loc(#loc71)
    %77 = arith.select %11, %76, %75 : index loc(#loc71)
    %78 = handshake.mux %77 [%falseResult_50, %trueResult_45] : index, none loc(#loc71)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue, %78 : none loc(#loc60)
    %79 = handshake.mux %32 [%falseResult_52, %trueResult] : index, none loc(#loc60)
    %80 = dataflow.carry %willContinue, %falseResult, %trueResult_57 : i1, none, none -> none loc(#loc60)
    %trueResult_53, %falseResult_54 = handshake.cond_br %11, %80 : none loc(#loc71)
    %81 = dataflow.carry %willContinue_7, %falseResult_54, %trueResult_55 : i1, none, none -> none loc(#loc71)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue_7, %34#1 : none loc(#loc71)
    %82 = handshake.constant %80 {value = 0 : index} : index loc(#loc71)
    %83 = handshake.constant %80 {value = 1 : index} : index loc(#loc71)
    %84 = arith.select %11, %83, %82 : index loc(#loc71)
    %85 = handshake.mux %84 [%falseResult_56, %trueResult_53] : index, none loc(#loc71)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue, %85 : none loc(#loc60)
    %86 = handshake.mux %32 [%falseResult_58, %trueResult] : index, none loc(#loc60)
    %87 = dataflow.carry %willContinue, %falseResult, %trueResult_65 : i1, none, none -> none loc(#loc60)
    %trueResult_59, %falseResult_60 = handshake.cond_br %11, %87 : none loc(#loc71)
    %88 = dataflow.carry %willContinue_7, %falseResult_60, %trueResult_63 : i1, none, none -> none loc(#loc71)
    %trueResult_61, %falseResult_62 = handshake.cond_br %17, %88 : none loc(#loc75)
    %89 = handshake.constant %88 {value = 0 : index} : index loc(#loc75)
    %90 = handshake.constant %88 {value = 1 : index} : index loc(#loc75)
    %91 = arith.select %17, %90, %89 : index loc(#loc75)
    %92 = handshake.mux %91 [%falseResult_62, %35#1] : index, none loc(#loc75)
    %trueResult_63, %falseResult_64 = handshake.cond_br %willContinue_7, %92 : none loc(#loc71)
    %93 = handshake.constant %87 {value = 0 : index} : index loc(#loc71)
    %94 = handshake.constant %87 {value = 1 : index} : index loc(#loc71)
    %95 = arith.select %11, %94, %93 : index loc(#loc71)
    %96 = handshake.mux %95 [%falseResult_64, %trueResult_59] : index, none loc(#loc71)
    %trueResult_65, %falseResult_66 = handshake.cond_br %willContinue, %96 : none loc(#loc60)
    %97 = handshake.mux %32 [%falseResult_66, %trueResult] : index, none loc(#loc60)
    %98 = dataflow.carry %willContinue, %falseResult, %trueResult_73 : i1, none, none -> none loc(#loc60)
    %trueResult_67, %falseResult_68 = handshake.cond_br %11, %98 : none loc(#loc71)
    %99 = dataflow.carry %willContinue_7, %falseResult_68, %trueResult_71 : i1, none, none -> none loc(#loc71)
    %trueResult_69, %falseResult_70 = handshake.cond_br %17, %99 : none loc(#loc75)
    %100 = handshake.constant %99 {value = 0 : index} : index loc(#loc75)
    %101 = handshake.constant %99 {value = 1 : index} : index loc(#loc75)
    %102 = arith.select %17, %101, %100 : index loc(#loc75)
    %103 = handshake.mux %102 [%falseResult_70, %37#1] : index, none loc(#loc75)
    %trueResult_71, %falseResult_72 = handshake.cond_br %willContinue_7, %103 : none loc(#loc71)
    %104 = handshake.constant %98 {value = 0 : index} : index loc(#loc71)
    %105 = handshake.constant %98 {value = 1 : index} : index loc(#loc71)
    %106 = arith.select %11, %105, %104 : index loc(#loc71)
    %107 = handshake.mux %106 [%falseResult_72, %trueResult_67] : index, none loc(#loc71)
    %trueResult_73, %falseResult_74 = handshake.cond_br %willContinue, %107 : none loc(#loc60)
    %108 = handshake.mux %32 [%falseResult_74, %trueResult] : index, none loc(#loc60)
    %109 = handshake.join %46, %57, %68, %79, %86, %97, %108 : none, none, none, none, none, none, none loc(#loc54)
    handshake.return %33, %109 : i32, none loc(#loc55)
  } loc(#loc54)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc40)
    %c2_i32 = arith.constant 2 : i32 loc(#loc40)
    %1 = ub.poison : i32 loc(#loc40)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c3 = arith.constant 3 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.a_ids : memref<3xi32> loc(#loc2)
    %3 = memref.get_global @__const.main.b_ids : memref<3xi32> loc(#loc2)
    %4 = memref.get_global @__const.main.a_values : memref<3xi32> loc(#loc2)
    %5 = memref.get_global @__const.main.b_values : memref<3xi32> loc(#loc2)
    %6 = memref.get_global @".str.2" : memref<61xi8> loc(#loc2)
    %7 = memref.get_global @".str.1.3" : memref<46xi8> loc(#loc2)
    %8 = memref.get_global @str : memref<22xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<3xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<3xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<3xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<3xi32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_6 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_7 = memref.alloca() : memref<5xi32> loc(#loc2)
    %alloca_8 = memref.alloca() : memref<5xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %13 = memref.load %2[%arg0] : memref<3xi32> loc(#loc41)
      memref.store %13, %alloca[%arg0] : memref<3xi32> loc(#loc41)
    } loc(#loc41)
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %13 = memref.load %3[%arg0] : memref<3xi32> loc(#loc42)
      memref.store %13, %alloca_0[%arg0] : memref<3xi32> loc(#loc42)
    } loc(#loc42)
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %13 = memref.load %4[%arg0] : memref<3xi32> loc(#loc43)
      memref.store %13, %alloca_1[%arg0] : memref<3xi32> loc(#loc43)
    } loc(#loc43)
    scf.for %arg0 = %c0 to %c3 step %c1 {
      %13 = memref.load %5[%arg0] : memref<3xi32> loc(#loc44)
      memref.store %13, %alloca_2[%arg0] : memref<3xi32> loc(#loc44)
    } loc(#loc44)
    %cast = memref.cast %alloca : memref<3xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_9 = memref.cast %alloca_0 : memref<3xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_10 = memref.cast %alloca_1 : memref<3xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_11 = memref.cast %alloca_2 : memref<3xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_12 = memref.cast %alloca_3 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_13 = memref.cast %alloca_4 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_14 = memref.cast %alloca_5 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
    %9 = call @_Z17database_join_cpuPKiS0_S0_S0_PiS1_S1_jj(%cast, %cast_9, %cast_10, %cast_11, %cast_12, %cast_13, %cast_14, %c3_i32, %c3_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> i32 loc(#loc45)
    %cast_15 = memref.cast %alloca_6 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %cast_16 = memref.cast %alloca_7 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %cast_17 = memref.cast %alloca_8 : memref<5xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_18, %ready_19 = esi.wrap.vr %cast_9, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_20, %ready_21 = esi.wrap.vr %cast_10, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_22, %ready_23 = esi.wrap.vr %cast_11, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_24, %ready_25 = esi.wrap.vr %cast_15, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_26, %ready_27 = esi.wrap.vr %cast_16, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_28, %ready_29 = esi.wrap.vr %cast_17, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc46)
    %chanOutput_30, %ready_31 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc46)
    %chanOutput_32, %ready_33 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc46)
    %chanOutput_34, %ready_35 = esi.wrap.vr %true, %true : i1 loc(#loc46)
    %10:2 = handshake.esi_instance @_Z17database_join_dsaPKiS0_S0_S0_PiS1_S1_jj_esi "_Z17database_join_dsaPKiS0_S0_S0_PiS1_S1_jj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_18, %chanOutput_20, %chanOutput_22, %chanOutput_24, %chanOutput_26, %chanOutput_28, %chanOutput_30, %chanOutput_32, %chanOutput_34) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>) loc(#loc46)
    %rawOutput, %valid = esi.unwrap.vr %10#0, %true : i32 loc(#loc46)
    %rawOutput_36, %valid_37 = esi.unwrap.vr %10#1, %true : i1 loc(#loc46)
    %11 = arith.cmpi eq, %9, %rawOutput : i32 loc(#loc50)
    %12 = scf.if %11 -> (i32) {
      %13 = arith.cmpi eq, %9, %c0_i32 : i32 loc(#loc56)
      %14:2 = scf.if %13 -> (i1, i32) {
        scf.yield %false, %c0_i32 : i1, i32 loc(#loc51)
      } else {
        %16:3 = scf.while (%arg0 = %c0_i32) : (i32) -> (i32, i32, i32) {
          %19 = arith.extui %arg0 : i32 to i64 loc(#loc64)
          %20 = arith.index_cast %19 : i64 to index loc(#loc64)
          %21 = memref.load %alloca_3[%20] : memref<5xi32> loc(#loc64)
          %22 = memref.load %alloca_6[%20] : memref<5xi32> loc(#loc64)
          %23 = arith.cmpi eq, %21, %22 : i32 loc(#loc64)
          %24:3 = scf.if %23 -> (i32, i32, i32) {
            %26 = memref.load %alloca_4[%20] : memref<5xi32> loc(#loc65)
            %27 = memref.load %alloca_7[%20] : memref<5xi32> loc(#loc65)
            %28 = arith.cmpi eq, %26, %27 : i32 loc(#loc65)
            %29:3 = scf.if %28 -> (i32, i32, i32) {
              %30 = memref.load %alloca_5[%20] : memref<5xi32> loc(#loc66)
              %31 = memref.load %alloca_8[%20] : memref<5xi32> loc(#loc66)
              %32 = arith.cmpi eq, %30, %31 : i32 loc(#loc66)
              %33:3 = scf.if %32 -> (i32, i32, i32) {
                %34 = arith.addi %arg0, %c1_i32 : i32 loc(#loc56)
                %35 = arith.cmpi eq, %34, %9 : i32 loc(#loc56)
                %36 = arith.extui %35 : i1 to i32 loc(#loc51)
                %37 = arith.cmpi ne, %34, %9 : i32 loc(#loc61)
                %38 = arith.extui %37 : i1 to i32 loc(#loc51)
                scf.yield %34, %36, %38 : i32, i32, i32 loc(#loc65)
              } else {
                scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc65)
              } loc(#loc65)
              scf.yield %33#0, %33#1, %33#2 : i32, i32, i32 loc(#loc65)
            } else {
              scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc65)
            } loc(#loc65)
            scf.yield %29#0, %29#1, %29#2 : i32, i32, i32 loc(#loc64)
          } else {
            scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc64)
          } loc(#loc64)
          %25 = arith.trunci %24#2 : i32 to i1 loc(#loc51)
          scf.condition(%25) %24#0, %arg0, %24#1 : i32, i32, i32 loc(#loc51)
        } do {
        ^bb0(%arg0: i32 loc(fused<#di_lexical_block10>[#loc29]), %arg1: i32 loc(fused<#di_lexical_block10>[#loc29]), %arg2: i32 loc(fused<#di_lexical_block10>[#loc29])):
          scf.yield %arg0 : i32 loc(#loc51)
        } loc(#loc51)
        %17 = arith.index_castui %16#2 : i32 to index loc(#loc51)
        %18:2 = scf.index_switch %17 -> i1, i32 
        case 1 {
          scf.yield %false, %c0_i32 : i1, i32 loc(#loc51)
        }
        default {
          %intptr = memref.extract_aligned_pointer_as_index %7 : memref<46xi8> -> index loc(#loc68)
          %19 = arith.index_cast %intptr : index to i64 loc(#loc68)
          %20 = llvm.inttoptr %19 : i64 to !llvm.ptr loc(#loc68)
          %21 = llvm.call @printf(%20, %16#1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32 loc(#loc68)
          scf.yield %true, %c1_i32 : i1, i32 loc(#loc69)
        } loc(#loc51)
        scf.yield %18#0, %18#1 : i1, i32 loc(#loc51)
      } loc(#loc51)
      %15 = arith.select %14#0, %14#1, %c0_i32 : i32 loc(#loc2)
      scf.if %14#0 {
      } else {
        %intptr = memref.extract_aligned_pointer_as_index %8 : memref<22xi8> -> index loc(#loc47)
        %16 = arith.index_cast %intptr : index to i64 loc(#loc47)
        %17 = llvm.inttoptr %16 : i64 to !llvm.ptr loc(#loc47)
        %18 = llvm.call @puts(%17) : (!llvm.ptr) -> i32 loc(#loc47)
      } loc(#loc2)
      scf.yield %15 : i32 loc(#loc50)
    } else {
      %intptr = memref.extract_aligned_pointer_as_index %6 : memref<61xi8> -> index loc(#loc57)
      %13 = arith.index_cast %intptr : index to i64 loc(#loc57)
      %14 = llvm.inttoptr %13 : i64 to !llvm.ptr loc(#loc57)
      %15 = llvm.call @printf(%14, %9, %rawOutput) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32 loc(#loc57)
      scf.yield %c1_i32 : i32 loc(#loc58)
    } loc(#loc50)
    return %12 : i32 loc(#loc48)
  } loc(#loc40)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc49)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file2 = #llvm.di_file<"/usr/include/stdio.h" in "">
#di_null_type = #llvm.di_null_type
#loc = loc("tests/app/database_join/database_join.cpp":0:0)
#loc2 = loc(unknown)
#loc5 = loc("tests/app/database_join/database_join.cpp":28:0)
#loc6 = loc("tests/app/database_join/database_join.cpp":29:0)
#loc7 = loc("tests/app/database_join/database_join.cpp":30:0)
#loc8 = loc("tests/app/database_join/database_join.cpp":31:0)
#loc9 = loc("tests/app/database_join/database_join.cpp":32:0)
#loc10 = loc("tests/app/database_join/database_join.cpp":33:0)
#loc11 = loc("tests/app/database_join/database_join.cpp":36:0)
#loc13 = loc("tests/app/database_join/database_join.cpp":54:0)
#loc14 = loc("tests/app/database_join/database_join.cpp":55:0)
#loc15 = loc("tests/app/database_join/database_join.cpp":56:0)
#loc16 = loc("tests/app/database_join/database_join.cpp":57:0)
#loc17 = loc("tests/app/database_join/database_join.cpp":58:0)
#loc18 = loc("tests/app/database_join/database_join.cpp":59:0)
#loc19 = loc("tests/app/database_join/database_join.cpp":60:0)
#loc20 = loc("tests/app/database_join/database_join.cpp":64:0)
#loc21 = loc("tests/app/database_join/main.cpp":5:0)
#loc22 = loc("tests/app/database_join/main.cpp":12:0)
#loc23 = loc("tests/app/database_join/main.cpp":13:0)
#loc24 = loc("tests/app/database_join/main.cpp":14:0)
#loc25 = loc("tests/app/database_join/main.cpp":15:0)
#loc26 = loc("tests/app/database_join/main.cpp":26:0)
#loc27 = loc("tests/app/database_join/main.cpp":30:0)
#loc28 = loc("tests/app/database_join/main.cpp":34:0)
#loc30 = loc("tests/app/database_join/main.cpp":42:0)
#loc31 = loc("tests/app/database_join/main.cpp":43:0)
#loc32 = loc("tests/app/database_join/main.cpp":44:0)
#loc33 = loc("tests/app/database_join/main.cpp":45:0)
#loc34 = loc("tests/app/database_join/main.cpp":46:0)
#loc35 = loc("tests/app/database_join/main.cpp":50:0)
#loc36 = loc("tests/app/database_join/main.cpp":35:0)
#loc37 = loc("tests/app/database_join/main.cpp":37:0)
#loc38 = loc("tests/app/database_join/main.cpp":52:0)
#loc39 = loc("/usr/include/stdio.h":363:0)
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type1, #di_derived_type9, #di_null_type>
#di_subprogram4 = #llvm.di_subprogram<scope = #di_file2, name = "printf", file = #di_file2, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type1>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 34>
#loc40 = loc(fused<#di_subprogram3>[#loc21])
#loc41 = loc(fused<#di_subprogram3>[#loc22])
#loc42 = loc(fused<#di_subprogram3>[#loc23])
#loc43 = loc(fused<#di_subprogram3>[#loc24])
#loc44 = loc(fused<#di_subprogram3>[#loc25])
#loc45 = loc(fused<#di_subprogram3>[#loc26])
#loc46 = loc(fused<#di_subprogram3>[#loc27])
#loc47 = loc(fused<#di_subprogram3>[#loc35])
#loc48 = loc(fused<#di_subprogram3>[#loc38])
#loc49 = loc(fused<#di_subprogram4>[#loc39])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 41>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 34>
#loc50 = loc(fused<#di_lexical_block9>[#loc28])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 54>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 41>
#loc53 = loc(fused<#di_subprogram5>[#loc11])
#loc55 = loc(fused<#di_subprogram6>[#loc20])
#loc56 = loc(fused<#di_lexical_block11>[#loc29])
#loc57 = loc(fused<#di_lexical_block12>[#loc36])
#loc58 = loc(fused<#di_lexical_block12>[#loc37])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 54>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 42>
#loc60 = loc(fused<#di_lexical_block14>[#loc13])
#loc61 = loc(fused[#loc51, #loc56])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 54>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 44>
#loc62 = loc(fused<#di_lexical_block16>[#loc3])
#loc63 = loc(fused<#di_lexical_block17>[#loc13])
#loc64 = loc(fused<#di_lexical_block18>[#loc30])
#loc65 = loc(fused<#di_lexical_block18>[#loc31])
#loc66 = loc(fused<#di_lexical_block18>[#loc32])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 55>
#loc67 = loc(fused[#loc59, #loc62])
#loc68 = loc(fused<#di_lexical_block21>[#loc33])
#loc69 = loc(fused<#di_lexical_block21>[#loc34])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 27>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 55>
#loc71 = loc(fused<#di_lexical_block23>[#loc14])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 27>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 55>
#loc72 = loc(fused<#di_lexical_block24>[#loc4])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 28>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 56>
#loc73 = loc(fused[#loc70, #loc72])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 28>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 56>
#loc74 = loc(fused<#di_lexical_block28>[#loc5])
#loc75 = loc(fused<#di_lexical_block29>[#loc15])
#loc76 = loc(fused<#di_lexical_block30>[#loc6])
#loc77 = loc(fused<#di_lexical_block30>[#loc7])
#loc78 = loc(fused<#di_lexical_block30>[#loc8])
#loc79 = loc(fused<#di_lexical_block30>[#loc9])
#loc80 = loc(fused<#di_lexical_block30>[#loc10])
#loc81 = loc(fused<#di_lexical_block31>[#loc16])
#loc82 = loc(fused<#di_lexical_block31>[#loc17])
#loc83 = loc(fused<#di_lexical_block31>[#loc18])
#loc84 = loc(fused<#di_lexical_block31>[#loc19])
