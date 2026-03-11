#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/gemv/gemv.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/gemv/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/gemv/gemv.cpp":18:0)
#loc3 = loc("tests/app/gemv/gemv.cpp":26:0)
#loc4 = loc("tests/app/gemv/gemv.cpp":28:0)
#loc8 = loc("tests/app/gemv/gemv.cpp":38:0)
#loc15 = loc("tests/app/gemv/main.cpp":21:0)
#loc17 = loc("tests/app/gemv/main.cpp":24:0)
#loc19 = loc("tests/app/gemv/main.cpp":27:0)
#loc23 = loc("tests/app/gemv/main.cpp":38:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 26>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 46>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 21>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 24>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 27>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 38>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 26>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 46>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 49152, elements = #llvm.di_subrange<count = 1536 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1536, elements = #llvm.di_subrange<count = 48 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1024, elements = #llvm.di_subrange<count = 32 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 26>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 46>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 46, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 21, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 24, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 27, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "i", file = #di_file1, line = 38, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 28>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 48>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "alpha", file = #di_file, line = 18, arg = 1, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "beta", file = #di_file, line = 21, arg = 4, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 24, arg = 7, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 25, arg = 8, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "sum", file = #di_file, line = 27, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "alpha", file = #di_file, line = 38, arg = 1, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "beta", file = #di_file, line = 41, arg = 4, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file, line = 44, arg = 7, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 45, arg = 8, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "sum", file = #di_file, line = 47, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 6, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 7, type = #di_derived_type2>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "alpha", file = #di_file1, line = 8, type = #di_derived_type2>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "beta", file = #di_file1, line = 9, type = #di_derived_type2>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "x", file = #di_file1, line = 13, type = #di_composite_type1>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_y", file = #di_file1, line = 14, type = #di_composite_type2>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_y", file = #di_file1, line = 17, type = #di_composite_type2>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_y", file = #di_file1, line = 18, type = #di_composite_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_y", file = #di_file, line = 23, arg = 6, type = #di_derived_type5>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "j", file = #di_file, line = 28, type = #di_derived_type1>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_y", file = #di_file, line = 43, arg = 6, type = #di_derived_type5>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block11, name = "j", file = #di_file, line = 48, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable16, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable22, #di_local_variable23, #di_local_variable24, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_local_variable5>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 21>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 24>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 27>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 38>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 19, arg = 2, type = #di_derived_type6>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram, name = "x", file = #di_file, line = 20, arg = 3, type = #di_derived_type6>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_y", file = #di_file, line = 22, arg = 5, type = #di_derived_type6>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file, line = 39, arg = 2, type = #di_derived_type6>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram1, name = "x", file = #di_file, line = 40, arg = 3, type = #di_derived_type6>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_y", file = #di_file, line = 42, arg = 5, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type2, #di_derived_type6, #di_derived_type6, #di_derived_type2, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gemv_cpu", linkageName = "_Z8gemv_cpujPKjS0_jS0_Pjjj", file = #di_file, line = 18, scopeLine = 25, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable6, #di_local_variable29, #di_local_variable30, #di_local_variable7, #di_local_variable31, #di_local_variable25, #di_local_variable8, #di_local_variable9, #di_local_variable, #di_local_variable10, #di_local_variable26>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "gemv_dsa", linkageName = "_Z8gemv_dsajPKjS0_jS0_Pjjj", file = #di_file, line = 38, scopeLine = 45, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable11, #di_local_variable32, #di_local_variable33, #di_local_variable12, #di_local_variable34, #di_local_variable27, #di_local_variable13, #di_local_variable14, #di_local_variable1, #di_local_variable15, #di_local_variable28>
#loc34 = loc(fused<#di_lexical_block12>[#loc15])
#loc35 = loc(fused<#di_lexical_block13>[#loc17])
#loc36 = loc(fused<#di_lexical_block14>[#loc19])
#loc37 = loc(fused<#di_lexical_block15>[#loc23])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 26>
#loc38 = loc(fused<#di_subprogram4>[#loc1])
#loc40 = loc(fused<#di_subprogram5>[#loc8])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 26>
#loc46 = loc(fused<#di_lexical_block20>[#loc3])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 26>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 28>
#loc63 = loc(fused<#di_lexical_block32>[#loc4])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 52, 44, 98, 108, 111, 99, 107, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<24xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 103, 101, 109, 118, 47, 103, 101, 109, 118, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<13xi8> = dense<[103, 101, 109, 118, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<13xi8> = dense<[103, 101, 109, 118, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z8gemv_cpujPKjS0_jS0_Pjjj(%arg0: i32 loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg5: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg6: i32 loc(fused<#di_subprogram4>[#loc1]), %arg7: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg6, %c0_i32 : i32 loc(#loc55)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg6 : i32 to i64 loc(#loc55)
      %3 = arith.extui %arg7 : i32 to i64 loc(#loc2)
      %4 = scf.while (%arg8 = %c0_i64) : (i64) -> i64 {
        %5 = scf.if %1 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc63)
        } else {
          %12 = arith.trunci %arg8 : i64 to i32 loc(#loc2)
          %13 = arith.muli %arg7, %12 : i32 loc(#loc2)
          %14:2 = scf.while (%arg9 = %c0_i64, %arg10 = %c0_i32) : (i64, i32) -> (i64, i32) {
            %16 = arith.trunci %arg9 : i64 to i32 loc(#loc66)
            %17 = arith.addi %13, %16 : i32 loc(#loc66)
            %18 = arith.extui %17 : i32 to i64 loc(#loc66)
            %19 = arith.index_cast %18 : i64 to index loc(#loc66)
            %20 = memref.load %arg1[%19] : memref<?xi32, strided<[1], offset: ?>> loc(#loc66)
            %21 = arith.index_cast %arg9 : i64 to index loc(#loc66)
            %22 = memref.load %arg2[%21] : memref<?xi32, strided<[1], offset: ?>> loc(#loc66)
            %23 = arith.muli %22, %20 : i32 loc(#loc66)
            %24 = arith.addi %23, %arg10 : i32 loc(#loc66)
            %25 = arith.addi %arg9, %c1_i64 : i64 loc(#loc65)
            %26 = arith.cmpi ne, %25, %3 : i64 loc(#loc67)
            scf.condition(%26) %25, %24 : i64, i32 loc(#loc63)
          } do {
          ^bb0(%arg9: i64 loc(fused<#di_lexical_block32>[#loc4]), %arg10: i32 loc(fused<#di_lexical_block32>[#loc4])):
            scf.yield %arg9, %arg10 : i64, i32 loc(#loc63)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc63)
          %15 = arith.muli %14#1, %arg0 : i32 loc(#loc58)
          scf.yield %15 : i32 loc(#loc58)
        } loc(#loc63)
        %6 = arith.index_cast %arg8 : i64 to index loc(#loc58)
        %7 = memref.load %arg4[%6] : memref<?xi32, strided<[1], offset: ?>> loc(#loc58)
        %8 = arith.muli %7, %arg3 : i32 loc(#loc58)
        %9 = arith.addi %8, %5 : i32 loc(#loc58)
        memref.store %9, %arg5[%6] : memref<?xi32, strided<[1], offset: ?>> loc(#loc58)
        %10 = arith.addi %arg8, %c1_i64 : i64 loc(#loc55)
        %11 = arith.cmpi ne, %10, %2 : i64 loc(#loc59)
        scf.condition(%11) %10 : i64 loc(#loc46)
      } do {
      ^bb0(%arg8: i64 loc(fused<#di_lexical_block20>[#loc3])):
        scf.yield %arg8 : i64 loc(#loc46)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    } loc(#loc46)
    return loc(#loc39)
  } loc(#loc38)
  handshake.func @_Z8gemv_dsajPKjS0_jS0_Pjjj_esi(%arg0: i32 loc(fused<#di_subprogram5>[#loc8]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg3: i32 loc(fused<#di_subprogram5>[#loc8]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg6: i32 loc(fused<#di_subprogram5>[#loc8]), %arg7: i32 loc(fused<#di_subprogram5>[#loc8]), %arg8: i1 loc(fused<#di_subprogram5>[#loc8]), ...) -> i1 attributes {argNames = ["alpha", "A", "x", "beta", "input_y", "output_y", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg8 : i1 loc(#loc40)
    %1 = handshake.join %0 : none loc(#loc40)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg6, %2 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc47)
    %5 = arith.cmpi eq, %arg7, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %7 = arith.index_cast %3 : i64 to index loc(#loc47)
    %8 = arith.index_cast %arg6 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc47)
    %10 = arith.index_cast %afterValue : index to i64 loc(#loc47)
    %11 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc64)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %9 : none loc(#loc64)
    %12 = arith.trunci %10 : i64 to i32 loc(#loc2)
    %13 = arith.muli %arg7, %12 : i32 loc(#loc2)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc64)
    %15 = arith.index_cast %arg7 : i32 to index loc(#loc64)
    %index_2, %willContinue_3 = dataflow.stream %7, %14, %15 {step_op = "+=", stop_cond = "!="} loc(#loc64)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc64)
    %16 = dataflow.carry %willContinue_3, %2, %24 : i1, i32, i32 -> i32 loc(#loc64)
    %afterValue_6, %afterCond_7 = dataflow.gate %16, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc64)
    handshake.sink %afterCond_7 : i1 loc(#loc64)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %16 : i32 loc(#loc64)
    %17 = arith.index_cast %afterValue_4 : index to i64 loc(#loc64)
    %18 = arith.trunci %17 : i64 to i32 loc(#loc68)
    %19 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc68)
    %20 = arith.addi %19, %18 : i32 loc(#loc68)
    %21 = arith.extui %20 : i32 to i64 loc(#loc68)
    %22 = arith.index_cast %21 : i64 to index loc(#loc68)
    %dataResult, %addressResults = handshake.load [%22] %32#0, %37 : index, i32 loc(#loc68)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %34#0, %49 : index, i32 loc(#loc68)
    %23 = arith.muli %dataResult_10, %dataResult : i32 loc(#loc68)
    %24 = arith.addi %23, %afterValue_6 : i32 loc(#loc68)
    %25 = arith.muli %falseResult_9, %arg0 : i32 loc(#loc60)
    %26 = handshake.constant %9 {value = 0 : index} : index loc(#loc64)
    %27 = handshake.constant %9 {value = 1 : index} : index loc(#loc64)
    %28 = arith.select %11, %27, %26 : index loc(#loc64)
    %29 = handshake.mux %28 [%25, %2] : index, i32 loc(#loc64)
    %dataResult_12, %addressResults_13 = handshake.load [%afterValue] %33#0, %46 : index, i32 loc(#loc60)
    %30 = arith.muli %dataResult_12, %arg3 : i32 loc(#loc60)
    %31 = arith.addi %30, %29 : i32 loc(#loc60)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %31, %55 : index, i32 loc(#loc60)
    %32:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc40)
    %33:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (i32, none) loc(#loc40)
    %34:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_11) {id = 2 : i32} : (index) -> (i32, none) loc(#loc40)
    %35 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 3 : i32} : (i32, index) -> none loc(#loc40)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc47)
    %trueResult_15, %falseResult_16 = handshake.cond_br %11, %36 : none loc(#loc64)
    %37 = dataflow.carry %willContinue_3, %falseResult_16, %trueResult_17 : i1, none, none -> none loc(#loc64)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_3, %32#1 : none loc(#loc64)
    %38 = handshake.constant %36 {value = 0 : index} : index loc(#loc64)
    %39 = handshake.constant %36 {value = 1 : index} : index loc(#loc64)
    %40 = arith.select %11, %39, %38 : index loc(#loc64)
    %41 = handshake.mux %40 [%falseResult_18, %trueResult_15] : index, none loc(#loc64)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %41 : none loc(#loc47)
    %42 = handshake.constant %1 {value = 0 : index} : index loc(#loc47)
    %43 = handshake.constant %1 {value = 1 : index} : index loc(#loc47)
    %44 = arith.select %4, %43, %42 : index loc(#loc47)
    %45 = handshake.mux %44 [%falseResult_20, %trueResult] : index, none loc(#loc47)
    %46 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc47)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %33#1 : none loc(#loc47)
    %47 = handshake.mux %44 [%falseResult_22, %trueResult] : index, none loc(#loc47)
    %48 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc47)
    %trueResult_23, %falseResult_24 = handshake.cond_br %11, %48 : none loc(#loc64)
    %49 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_25 : i1, none, none -> none loc(#loc64)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue_3, %34#1 : none loc(#loc64)
    %50 = handshake.constant %48 {value = 0 : index} : index loc(#loc64)
    %51 = handshake.constant %48 {value = 1 : index} : index loc(#loc64)
    %52 = arith.select %11, %51, %50 : index loc(#loc64)
    %53 = handshake.mux %52 [%falseResult_26, %trueResult_23] : index, none loc(#loc64)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %53 : none loc(#loc47)
    %54 = handshake.mux %44 [%falseResult_28, %trueResult] : index, none loc(#loc47)
    %55 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc47)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %35 : none loc(#loc47)
    %56 = handshake.mux %44 [%falseResult_30, %trueResult] : index, none loc(#loc47)
    %57 = handshake.join %45, %47, %54, %56 : none, none, none, none loc(#loc40)
    %58 = handshake.constant %57 {value = true} : i1 loc(#loc40)
    handshake.return %58 : i1 loc(#loc40)
  } loc(#loc40)
  handshake.func @_Z8gemv_dsajPKjS0_jS0_Pjjj(%arg0: i32 loc(fused<#di_subprogram5>[#loc8]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg3: i32 loc(fused<#di_subprogram5>[#loc8]), %arg4: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg5: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc8]), %arg6: i32 loc(fused<#di_subprogram5>[#loc8]), %arg7: i32 loc(fused<#di_subprogram5>[#loc8]), %arg8: none loc(fused<#di_subprogram5>[#loc8]), ...) -> none attributes {argNames = ["alpha", "A", "x", "beta", "input_y", "output_y", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg8 : none loc(#loc40)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg6, %1 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc47)
    %4 = arith.cmpi eq, %arg7, %1 : i32 loc(#loc2)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %6 = arith.index_cast %2 : i64 to index loc(#loc47)
    %7 = arith.index_cast %arg6 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc47)
    %9 = arith.index_cast %afterValue : index to i64 loc(#loc47)
    %10 = dataflow.invariant %afterCond, %4 : i1, i1 -> i1 loc(#loc64)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %8 : none loc(#loc64)
    %11 = arith.trunci %9 : i64 to i32 loc(#loc2)
    %12 = arith.muli %arg7, %11 : i32 loc(#loc2)
    %13 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc64)
    %14 = arith.index_cast %arg7 : i32 to index loc(#loc64)
    %index_2, %willContinue_3 = dataflow.stream %6, %13, %14 {step_op = "+=", stop_cond = "!="} loc(#loc64)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc64)
    %15 = dataflow.carry %willContinue_3, %1, %23 : i1, i32, i32 -> i32 loc(#loc64)
    %afterValue_6, %afterCond_7 = dataflow.gate %15, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc64)
    handshake.sink %afterCond_7 : i1 loc(#loc64)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %15 : i32 loc(#loc64)
    %16 = arith.index_cast %afterValue_4 : index to i64 loc(#loc64)
    %17 = arith.trunci %16 : i64 to i32 loc(#loc68)
    %18 = dataflow.invariant %afterCond_5, %12 : i1, i32 -> i32 loc(#loc68)
    %19 = arith.addi %18, %17 : i32 loc(#loc68)
    %20 = arith.extui %19 : i32 to i64 loc(#loc68)
    %21 = arith.index_cast %20 : i64 to index loc(#loc68)
    %dataResult, %addressResults = handshake.load [%21] %31#0, %36 : index, i32 loc(#loc68)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %33#0, %48 : index, i32 loc(#loc68)
    %22 = arith.muli %dataResult_10, %dataResult : i32 loc(#loc68)
    %23 = arith.addi %22, %afterValue_6 : i32 loc(#loc68)
    %24 = arith.muli %falseResult_9, %arg0 : i32 loc(#loc60)
    %25 = handshake.constant %8 {value = 0 : index} : index loc(#loc64)
    %26 = handshake.constant %8 {value = 1 : index} : index loc(#loc64)
    %27 = arith.select %10, %26, %25 : index loc(#loc64)
    %28 = handshake.mux %27 [%24, %1] : index, i32 loc(#loc64)
    %dataResult_12, %addressResults_13 = handshake.load [%afterValue] %32#0, %45 : index, i32 loc(#loc60)
    %29 = arith.muli %dataResult_12, %arg3 : i32 loc(#loc60)
    %30 = arith.addi %29, %28 : i32 loc(#loc60)
    %dataResult_14, %addressResult = handshake.store [%afterValue] %30, %54 : index, i32 loc(#loc60)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc40)
    %32:2 = handshake.extmemory[ld = 1, st = 0] (%arg4 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_13) {id = 1 : i32} : (index) -> (i32, none) loc(#loc40)
    %33:2 = handshake.extmemory[ld = 1, st = 0] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_11) {id = 2 : i32} : (index) -> (i32, none) loc(#loc40)
    %34 = handshake.extmemory[ld = 0, st = 1] (%arg5 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 3 : i32} : (i32, index) -> none loc(#loc40)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc47)
    %trueResult_15, %falseResult_16 = handshake.cond_br %10, %35 : none loc(#loc64)
    %36 = dataflow.carry %willContinue_3, %falseResult_16, %trueResult_17 : i1, none, none -> none loc(#loc64)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue_3, %31#1 : none loc(#loc64)
    %37 = handshake.constant %35 {value = 0 : index} : index loc(#loc64)
    %38 = handshake.constant %35 {value = 1 : index} : index loc(#loc64)
    %39 = arith.select %10, %38, %37 : index loc(#loc64)
    %40 = handshake.mux %39 [%falseResult_18, %trueResult_15] : index, none loc(#loc64)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %40 : none loc(#loc47)
    %41 = handshake.constant %0 {value = 0 : index} : index loc(#loc47)
    %42 = handshake.constant %0 {value = 1 : index} : index loc(#loc47)
    %43 = arith.select %3, %42, %41 : index loc(#loc47)
    %44 = handshake.mux %43 [%falseResult_20, %trueResult] : index, none loc(#loc47)
    %45 = dataflow.carry %willContinue, %falseResult, %trueResult_21 : i1, none, none -> none loc(#loc47)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue, %32#1 : none loc(#loc47)
    %46 = handshake.mux %43 [%falseResult_22, %trueResult] : index, none loc(#loc47)
    %47 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc47)
    %trueResult_23, %falseResult_24 = handshake.cond_br %10, %47 : none loc(#loc64)
    %48 = dataflow.carry %willContinue_3, %falseResult_24, %trueResult_25 : i1, none, none -> none loc(#loc64)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue_3, %33#1 : none loc(#loc64)
    %49 = handshake.constant %47 {value = 0 : index} : index loc(#loc64)
    %50 = handshake.constant %47 {value = 1 : index} : index loc(#loc64)
    %51 = arith.select %10, %50, %49 : index loc(#loc64)
    %52 = handshake.mux %51 [%falseResult_26, %trueResult_23] : index, none loc(#loc64)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %52 : none loc(#loc47)
    %53 = handshake.mux %43 [%falseResult_28, %trueResult] : index, none loc(#loc47)
    %54 = dataflow.carry %willContinue, %falseResult, %trueResult_29 : i1, none, none -> none loc(#loc47)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue, %34 : none loc(#loc47)
    %55 = handshake.mux %43 [%falseResult_30, %trueResult] : index, none loc(#loc47)
    %56 = handshake.join %44, %46, %53, %55 : none, none, none, none loc(#loc40)
    handshake.return %56 : none loc(#loc41)
  } loc(#loc40)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc29)
    %false = arith.constant false loc(#loc29)
    %0 = seq.const_clock  low loc(#loc29)
    %c2_i32 = arith.constant 2 : i32 loc(#loc2)
    %1 = ub.poison : i64 loc(#loc29)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1536_i64 = arith.constant 1536 : i64 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c48_i64 = arith.constant 48 : i64 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c48_i32 = arith.constant 48 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<13xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<13xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1536xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<48xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<32xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<32xi32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<32xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %12 = arith.trunci %arg0 : i64 to i32 loc(#loc48)
      %13 = arith.remui %12, %c10_i32 : i32 loc(#loc48)
      %14 = arith.addi %13, %c1_i32 : i32 loc(#loc48)
      %15 = arith.index_cast %arg0 : i64 to index loc(#loc48)
      memref.store %14, %alloca[%15] : memref<1536xi32> loc(#loc48)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc42)
      %17 = arith.cmpi ne, %16, %c1536_i64 : i64 loc(#loc49)
      scf.condition(%17) %16 : i64 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc15])):
      scf.yield %arg0 : i64 loc(#loc34)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc34)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %12 = arith.trunci %arg0 : i64 to i32 loc(#loc50)
      %13 = arith.remui %12, %c7_i32 : i32 loc(#loc50)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc50)
      memref.store %13, %alloca_0[%14] : memref<48xi32> loc(#loc50)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc43)
      %16 = arith.cmpi ne, %15, %c48_i64 : i64 loc(#loc51)
      scf.condition(%16) %15 : i64 loc(#loc35)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc17])):
      scf.yield %arg0 : i64 loc(#loc35)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc35)
    %6 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %12 = arith.trunci %arg0 : i64 to i32 loc(#loc52)
      %13 = arith.remui %12, %c5_i32 : i32 loc(#loc52)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      memref.store %13, %alloca_1[%14] : memref<32xi32> loc(#loc52)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc44)
      %16 = arith.cmpi ne, %15, %c32_i64 : i64 loc(#loc53)
      scf.condition(%16) %15 : i64 loc(#loc36)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block14>[#loc19])):
      scf.yield %arg0 : i64 loc(#loc36)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc36)
    %cast = memref.cast %alloca : memref<1536xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_4 = memref.cast %alloca_0 : memref<48xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_5 = memref.cast %alloca_1 : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_6 = memref.cast %alloca_2 : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    call @_Z8gemv_cpujPKjS0_jS0_Pjjj(%c2_i32, %cast, %cast_4, %c3_i32, %cast_5, %cast_6, %c32_i32, %c48_i32) : (i32, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc30)
    %cast_7 = memref.cast %alloca_3 : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput, %ready = esi.wrap.vr %c2_i32, %true : i32 loc(#loc31)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_10, %ready_11 = esi.wrap.vr %cast_4, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc31)
    %chanOutput_14, %ready_15 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_16, %ready_17 = esi.wrap.vr %cast_7, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_18, %ready_19 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc31)
    %chanOutput_20, %ready_21 = esi.wrap.vr %c48_i32, %true : i32 loc(#loc31)
    %chanOutput_22, %ready_23 = esi.wrap.vr %true, %true : i1 loc(#loc31)
    %7 = handshake.esi_instance @_Z8gemv_dsajPKjS0_jS0_Pjjj_esi "_Z8gemv_dsajPKjS0_jS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18, %chanOutput_20, %chanOutput_22) : (!esi.channel<i32>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc31)
    %rawOutput, %valid = esi.unwrap.vr %7, %true : i1 loc(#loc31)
    %8:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %13 = memref.load %alloca_2[%12] : memref<32xi32> loc(#loc57)
      %14 = memref.load %alloca_3[%12] : memref<32xi32> loc(#loc57)
      %15 = arith.cmpi eq, %13, %14 : i32 loc(#loc57)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
        %19 = arith.cmpi eq, %18, %c32_i64 : i64 loc(#loc45)
        %20 = arith.extui %19 : i1 to i32 loc(#loc37)
        %21 = arith.cmpi ne, %18, %c32_i64 : i64 loc(#loc54)
        %22 = arith.extui %21 : i1 to i32 loc(#loc37)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc37)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc37)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc23]), %arg1: i1 loc(fused<#di_lexical_block15>[#loc23]), %arg2: i32 loc(fused<#di_lexical_block15>[#loc23])):
      scf.yield %arg0 : i64 loc(#loc37)
    } loc(#loc37)
    %9 = arith.index_castui %8#2 : i32 to index loc(#loc37)
    %10 = scf.index_switch %9 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc37)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<13xi8> -> index loc(#loc61)
      %12 = arith.index_cast %intptr : index to i64 loc(#loc61)
      %13 = llvm.inttoptr %12 : i64 to !llvm.ptr loc(#loc61)
      %14 = llvm.call @puts(%13) : (!llvm.ptr) -> i32 loc(#loc61)
      scf.yield %c1_i32 : i32 loc(#loc62)
    } loc(#loc37)
    %11 = arith.select %8#1, %c0_i32, %10 : i32 loc(#loc2)
    scf.if %8#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<13xi8> -> index loc(#loc32)
      %12 = arith.index_cast %intptr : index to i64 loc(#loc32)
      %13 = llvm.inttoptr %12 : i64 to !llvm.ptr loc(#loc32)
      %14 = llvm.call @puts(%13) : (!llvm.ptr) -> i32 loc(#loc32)
    } loc(#loc2)
    return %11 : i32 loc(#loc33)
  } loc(#loc29)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/gemv/gemv.cpp":0:0)
#loc2 = loc(unknown)
#loc5 = loc("tests/app/gemv/gemv.cpp":29:0)
#loc6 = loc("tests/app/gemv/gemv.cpp":31:0)
#loc7 = loc("tests/app/gemv/gemv.cpp":33:0)
#loc9 = loc("tests/app/gemv/gemv.cpp":46:0)
#loc10 = loc("tests/app/gemv/gemv.cpp":48:0)
#loc11 = loc("tests/app/gemv/gemv.cpp":49:0)
#loc12 = loc("tests/app/gemv/gemv.cpp":51:0)
#loc13 = loc("tests/app/gemv/gemv.cpp":53:0)
#loc14 = loc("tests/app/gemv/main.cpp":5:0)
#loc16 = loc("tests/app/gemv/main.cpp":22:0)
#loc18 = loc("tests/app/gemv/main.cpp":25:0)
#loc20 = loc("tests/app/gemv/main.cpp":28:0)
#loc21 = loc("tests/app/gemv/main.cpp":32:0)
#loc22 = loc("tests/app/gemv/main.cpp":35:0)
#loc24 = loc("tests/app/gemv/main.cpp":39:0)
#loc25 = loc("tests/app/gemv/main.cpp":40:0)
#loc26 = loc("tests/app/gemv/main.cpp":41:0)
#loc27 = loc("tests/app/gemv/main.cpp":45:0)
#loc28 = loc("tests/app/gemv/main.cpp":47:0)
#loc29 = loc(fused<#di_subprogram3>[#loc14])
#loc30 = loc(fused<#di_subprogram3>[#loc21])
#loc31 = loc(fused<#di_subprogram3>[#loc22])
#loc32 = loc(fused<#di_subprogram3>[#loc27])
#loc33 = loc(fused<#di_subprogram3>[#loc28])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 21>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 24>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 27>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 38>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 46>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 21>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 24>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 27>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 38>
#loc39 = loc(fused<#di_subprogram4>[#loc7])
#loc41 = loc(fused<#di_subprogram5>[#loc13])
#loc42 = loc(fused<#di_lexical_block16>[#loc15])
#loc43 = loc(fused<#di_lexical_block17>[#loc17])
#loc44 = loc(fused<#di_lexical_block18>[#loc19])
#loc45 = loc(fused<#di_lexical_block19>[#loc23])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 46>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 39>
#loc47 = loc(fused<#di_lexical_block21>[#loc9])
#loc48 = loc(fused<#di_lexical_block22>[#loc16])
#loc49 = loc(fused[#loc34, #loc42])
#loc50 = loc(fused<#di_lexical_block23>[#loc18])
#loc51 = loc(fused[#loc35, #loc43])
#loc52 = loc(fused<#di_lexical_block24>[#loc20])
#loc53 = loc(fused[#loc36, #loc44])
#loc54 = loc(fused[#loc37, #loc45])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 46>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 39>
#loc55 = loc(fused<#di_lexical_block26>[#loc3])
#loc56 = loc(fused<#di_lexical_block27>[#loc9])
#loc57 = loc(fused<#di_lexical_block28>[#loc24])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 48>
#loc58 = loc(fused<#di_lexical_block29>[#loc6])
#loc59 = loc(fused[#loc46, #loc55])
#loc60 = loc(fused<#di_lexical_block30>[#loc12])
#loc61 = loc(fused<#di_lexical_block31>[#loc25])
#loc62 = loc(fused<#di_lexical_block31>[#loc26])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 28>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 48>
#loc64 = loc(fused<#di_lexical_block33>[#loc10])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 28>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 48>
#loc65 = loc(fused<#di_lexical_block34>[#loc4])
#loc66 = loc(fused<#di_lexical_block36>[#loc5])
#loc67 = loc(fused[#loc63, #loc65])
#loc68 = loc(fused<#di_lexical_block37>[#loc11])
