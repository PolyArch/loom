#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_file = #llvm.di_file<"tests/app/mat3x3_mult/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/mat3x3_mult/mat3x3_mult.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/mat3x3_mult/main.cpp":13:0)
#loc8 = loc("tests/app/mat3x3_mult/main.cpp":29:0)
#loc14 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":38:0)
#loc23 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":14:0)
#loc24 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":18:0)
#loc26 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":24:0)
#loc27 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":25:0)
#loc28 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":27:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type2, sizeInBits = 9216, elements = #llvm.di_subrange<count = 288 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type2>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type2, sizeInBits = 64>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 13>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 44>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 18>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type2>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 44>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 18>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "mat_a", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "mat_b", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_result", file = #di_file, line = 19, type = #di_composite_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_result", file = #di_file, line = 20, type = #di_composite_type>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 44>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 18>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 13, type = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 29, type = #di_derived_type3>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_mat_c", file = #di_file1, line = 40, arg = 3, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "n", file = #di_file1, line = 44, type = #di_derived_type3>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_mat_c", file = #di_file1, line = 16, arg = 3, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "n", file = #di_file1, line = 18, type = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 50>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 24>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type6>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_mat_a", file = #di_file1, line = 38, arg = 1, type = #di_derived_type7>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_mat_b", file = #di_file1, line = 39, arg = 2, type = #di_derived_type7>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 41, arg = 4, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "A", file = #di_file1, line = 45, type = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "B", file = #di_file1, line = 46, type = #di_derived_type4>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "C", file = #di_file1, line = 47, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_mat_a", file = #di_file1, line = 14, arg = 1, type = #di_derived_type7>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_mat_b", file = #di_file1, line = 15, arg = 2, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 17, arg = 4, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "A", file = #di_file1, line = 19, type = #di_derived_type4>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "B", file = #di_file1, line = 20, type = #di_derived_type4>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "C", file = #di_file1, line = 21, type = #di_derived_type2>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type7, #di_derived_type7, #di_derived_type5, #di_derived_type6>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 50>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 24>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "i", file = #di_file1, line = 50, type = #di_derived_type3>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "i", file = #di_file1, line = 24, type = #di_derived_type3>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable, #di_local_variable1, #di_local_variable4, #di_local_variable2, #di_local_variable3, #di_local_variable5>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 13>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 29>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 50>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 24>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 51>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 25>
#loc37 = loc(fused<#di_lexical_block12>[#loc3])
#loc38 = loc(fused<#di_lexical_block13>[#loc8])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 51>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 25>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_lexical_block18, name = "j", file = #di_file1, line = 51, type = #di_derived_type3>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block19, name = "j", file = #di_file1, line = 25, type = #di_derived_type3>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 51>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 25>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 53>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 27>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block25, name = "sum", file = #di_file1, line = 52, type = #di_basic_type2>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block26, name = "sum", file = #di_file1, line = 26, type = #di_basic_type2>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_lexical_block28, name = "k", file = #di_file1, line = 53, type = #di_derived_type3>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_lexical_block29, name = "k", file = #di_file1, line = 27, type = #di_derived_type3>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "mat3x3_mult_dsa", linkageName = "_Z15mat3x3_mult_dsaPKfS0_Pfj", file = #di_file1, line = 38, scopeLine = 41, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable11, #di_local_variable12, #di_local_variable6, #di_local_variable13, #di_local_variable7, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable23, #di_local_variable25, #di_local_variable27, #di_local_variable29>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "mat3x3_mult_cpu", linkageName = "_Z15mat3x3_mult_cpuPKfS0_Pfj", file = #di_file1, line = 14, scopeLine = 17, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable8, #di_local_variable19, #di_local_variable9, #di_local_variable20, #di_local_variable21, #di_local_variable22, #di_local_variable24, #di_local_variable26, #di_local_variable28, #di_local_variable30>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 18>
#loc48 = loc(fused<#di_subprogram4>[#loc14])
#loc50 = loc(fused<#di_subprogram5>[#loc23])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file1, line = 18>
#loc53 = loc(fused<#di_lexical_block31>[#loc24])
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file1, line = 18>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file1, line = 24>
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file1, line = 24>
#loc60 = loc(fused<#di_lexical_block37>[#loc26])
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file1, line = 24>
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file1, line = 25>
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_lexical_block43, file = #di_file1, line = 25>
#loc64 = loc(fused<#di_lexical_block43>[#loc27])
#di_lexical_block47 = #llvm.di_lexical_block<scope = #di_lexical_block45, file = #di_file1, line = 25>
#di_lexical_block49 = #llvm.di_lexical_block<scope = #di_lexical_block47, file = #di_file1, line = 27>
#loc70 = loc(fused<#di_lexical_block49>[#loc28])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[109, 97, 116, 51, 120, 51, 95, 109, 117, 108, 116, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[109, 97, 116, 51, 120, 51, 95, 109, 117, 108, 116, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 109, 97, 116, 51, 120, 51, 95, 109, 117, 108, 116, 47, 109, 97, 116, 51, 120, 51, 95, 109, 117, 108, 116, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc32)
    %false = arith.constant false loc(#loc32)
    %0 = seq.const_clock  low loc(#loc32)
    %c2_i32 = arith.constant 2 : i32 loc(#loc32)
    %1 = ub.poison : i64 loc(#loc32)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 1.000000e-01 : f32 loc(#loc2)
    %cst_0 = arith.constant 1.000000e+00 : f32 loc(#loc2)
    %cst_1 = arith.constant 5.000000e-02 : f32 loc(#loc2)
    %cst_2 = arith.constant 5.000000e-01 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c288_i64 = arith.constant 288 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %cst_3 = arith.constant 9.99999974E-5 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<288xf32> loc(#loc2)
    %alloca_4 = memref.alloca() : memref<288xf32> loc(#loc2)
    %alloca_5 = memref.alloca() : memref<288xf32> loc(#loc2)
    %alloca_6 = memref.alloca() : memref<288xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc41)
      %11 = arith.uitofp %10 : i32 to f32 loc(#loc41)
      %12 = math.fma %11, %cst, %cst_0 : f32 loc(#loc41)
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      memref.store %12, %alloca[%13] : memref<288xf32> loc(#loc41)
      %14 = math.fma %11, %cst_1, %cst_2 : f32 loc(#loc42)
      memref.store %14, %alloca_4[%13] : memref<288xf32> loc(#loc42)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
      %16 = arith.cmpi ne, %15, %c288_i64 : i64 loc(#loc43)
      scf.condition(%16) %15 : i64 loc(#loc37)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc37)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc37)
    %cast = memref.cast %alloca : memref<288xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %cast_7 = memref.cast %alloca_4 : memref<288xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    %cast_8 = memref.cast %alloca_5 : memref<288xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc33)
    call @_Z15mat3x3_mult_cpuPKfS0_Pfj(%cast, %cast_7, %cast_8, %c32_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc33)
    %cast_9 = memref.cast %alloca_6 : memref<288xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput_10, %ready_11 = esi.wrap.vr %cast_7, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput_12, %ready_13 = esi.wrap.vr %cast_9, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput_14, %ready_15 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc34)
    %chanOutput_16, %ready_17 = esi.wrap.vr %true, %true : i1 loc(#loc34)
    %5 = handshake.esi_instance @_Z15mat3x3_mult_dsaPKfS0_Pfj_esi "_Z15mat3x3_mult_dsaPKfS0_Pfj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc34)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc34)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      %11 = memref.load %alloca_5[%10] : memref<288xf32> loc(#loc45)
      %12 = memref.load %alloca_6[%10] : memref<288xf32> loc(#loc45)
      %13 = arith.subf %11, %12 : f32 loc(#loc45)
      %14 = math.absf %13 : f32 loc(#loc45)
      %15 = arith.cmpf ule, %14, %cst_3 : f32 loc(#loc45)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc40)
        %19 = arith.cmpi eq, %18, %c288_i64 : i64 loc(#loc40)
        %20 = arith.extui %19 : i1 to i32 loc(#loc38)
        %21 = arith.cmpi ne, %18, %c288_i64 : i64 loc(#loc44)
        %22 = arith.extui %21 : i1 to i32 loc(#loc38)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc45)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc45)
      } loc(#loc45)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc38)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc38)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc8]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc8]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc38)
    } loc(#loc38)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc38)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc38)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc46)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc46)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc46)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc46)
      scf.yield %c1_i32 : i32 loc(#loc47)
    } loc(#loc38)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc35)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc35)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc35)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc35)
    } loc(#loc2)
    return %9 : i32 loc(#loc36)
  } loc(#loc32)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z15mat3x3_mult_dsaPKfS0_Pfj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i1 loc(fused<#di_subprogram4>[#loc14]), ...) -> i1 attributes {argNames = ["input_mat_a", "input_mat_b", "output_mat_c", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc48)
    %1 = handshake.join %0 : none loc(#loc48)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 4 : index} : index loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 9 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %1 {value = 4294967295 : i64} : i64 loc(#loc2)
    %7 = handshake.constant %1 {value = 12 : i64} : i64 loc(#loc2)
    %8 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %9 = handshake.constant %1 {value = 3 : i64} : i64 loc(#loc2)
    %10 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %10, %1 : none loc(#loc52)
    %11 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc52)
    %12 = arith.index_cast %4 : i64 to index loc(#loc52)
    %13 = arith.index_cast %arg3 : i32 to index loc(#loc52)
    %index, %willContinue = dataflow.stream %12, %11, %13 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc52)
    %14 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc52)
    %15 = arith.index_cast %afterValue : index to i64 loc(#loc52)
    %16 = arith.muli %15, %5 : i64 loc(#loc56)
    %17 = arith.andi %16, %6 : i64 loc(#loc56)
    %18 = arith.index_cast %17 : i64 to index loc(#loc56)
    %19 = handshake.constant %14 {value = 1 : index} : index loc(#loc59)
    %20 = arith.index_cast %9 : i64 to index loc(#loc59)
    %index_0, %willContinue_1 = dataflow.stream %12, %19, %20 {step_op = "+=", stop_cond = "!="} loc(#loc59)
    %afterValue_2, %afterCond_3 = dataflow.gate %index_0, %willContinue_1 : index, i1 -> index, i1 loc(#loc59)
    %21 = dataflow.invariant %afterCond_3, %14 : i1, none -> none loc(#loc59)
    %22 = arith.index_cast %afterValue_2 : index to i64 loc(#loc59)
    %23 = arith.muli %22, %7 : i64 loc(#loc2)
    %24 = arith.index_cast %23 : i64 to index loc(#loc63)
    %25 = arith.divui %24, %3 : index loc(#loc63)
    %26 = dataflow.invariant %afterCond_3, %18 : i1, index -> index loc(#loc63)
    %27 = arith.addi %26, %25 : index loc(#loc63)
    %28 = handshake.constant %21 {value = 1 : index} : index loc(#loc63)
    %index_4, %willContinue_5 = dataflow.stream %12, %28, %20 {step_op = "+=", stop_cond = "!="} loc(#loc63)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc63)
    %29 = dataflow.invariant %afterCond_7, %21 : i1, none -> none loc(#loc63)
    %30 = arith.addi %26, %afterValue_6 : index loc(#loc69)
    %31 = handshake.constant %29 {value = 1 : index} : index loc(#loc69)
    %index_8, %willContinue_9 = dataflow.stream %12, %31, %20 {step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc69)
    %32 = dataflow.carry %willContinue_9, %8, %41 : i1, f32, f32 -> f32 loc(#loc69)
    %afterValue_12, %afterCond_13 = dataflow.gate %32, %willContinue_9 : f32, i1 -> f32, i1 loc(#loc69)
    handshake.sink %afterCond_13 : i1 loc(#loc69)
    %trueResult_14, %falseResult_15 = handshake.cond_br %willContinue_9, %32 : f32 loc(#loc69)
    %33 = arith.index_cast %afterValue_10 : index to i64 loc(#loc69)
    %34 = dataflow.invariant %afterCond_7, %27 : i1, index -> index loc(#loc72)
    %35 = arith.addi %34, %afterValue_10 : index loc(#loc72)
    %dataResult, %addressResults = handshake.load [%35] %44#0, %49 : index, f32 loc(#loc72)
    %36 = arith.muli %33, %7 : i64 loc(#loc72)
    %37 = arith.index_cast %36 : i64 to index loc(#loc72)
    %38 = arith.divui %37, %3 : index loc(#loc72)
    %39 = dataflow.invariant %afterCond_11, %30 : i1, index -> index loc(#loc72)
    %40 = arith.addi %39, %38 : index loc(#loc72)
    %dataResult_16, %addressResults_17 = handshake.load [%40] %43#0, %61 : index, f32 loc(#loc72)
    %41 = math.fma %dataResult, %dataResult_16, %afterValue_12 : f32 loc(#loc72)
    %42 = arith.addi %34, %afterValue_6 : index loc(#loc66)
    %dataResult_18, %addressResult = handshake.store [%42] %falseResult_15, %56 : index, f32 loc(#loc66)
    %43:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_17) {id = 0 : i32} : (index) -> (f32, none) loc(#loc48)
    %44:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc48)
    %45 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_18, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc48)
    %46 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc52)
    %47 = dataflow.carry %willContinue_1, %46, %trueResult_23 : i1, none, none -> none loc(#loc59)
    %48 = dataflow.carry %willContinue_5, %47, %trueResult_21 : i1, none, none -> none loc(#loc63)
    %49 = dataflow.carry %willContinue_9, %48, %trueResult_19 : i1, none, none -> none loc(#loc69)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue_9, %44#1 : none loc(#loc69)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_5, %falseResult_20 : none loc(#loc63)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_1, %falseResult_22 : none loc(#loc59)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %falseResult_24 : none loc(#loc52)
    %50 = handshake.constant %1 {value = 0 : index} : index loc(#loc52)
    %51 = handshake.constant %1 {value = 1 : index} : index loc(#loc52)
    %52 = arith.select %10, %51, %50 : index loc(#loc52)
    %53 = handshake.mux %52 [%falseResult_26, %trueResult] : index, none loc(#loc52)
    %54 = dataflow.carry %willContinue, %falseResult, %trueResult_31 : i1, none, none -> none loc(#loc52)
    %55 = dataflow.carry %willContinue_1, %54, %trueResult_29 : i1, none, none -> none loc(#loc59)
    %56 = dataflow.carry %willContinue_5, %55, %trueResult_27 : i1, none, none -> none loc(#loc63)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_5, %45 : none loc(#loc63)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue_1, %falseResult_28 : none loc(#loc59)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue, %falseResult_30 : none loc(#loc52)
    %57 = handshake.mux %52 [%falseResult_32, %trueResult] : index, none loc(#loc52)
    %58 = dataflow.carry %willContinue, %falseResult, %trueResult_39 : i1, none, none -> none loc(#loc52)
    %59 = dataflow.carry %willContinue_1, %58, %trueResult_37 : i1, none, none -> none loc(#loc59)
    %60 = dataflow.carry %willContinue_5, %59, %trueResult_35 : i1, none, none -> none loc(#loc63)
    %61 = dataflow.carry %willContinue_9, %60, %trueResult_33 : i1, none, none -> none loc(#loc69)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_9, %43#1 : none loc(#loc69)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_5, %falseResult_34 : none loc(#loc63)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_1, %falseResult_36 : none loc(#loc59)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue, %falseResult_38 : none loc(#loc52)
    %62 = handshake.mux %52 [%falseResult_40, %trueResult] : index, none loc(#loc52)
    %63 = handshake.join %53, %57, %62 : none, none, none loc(#loc48)
    %64 = handshake.constant %63 {value = true} : i1 loc(#loc48)
    handshake.return %64 : i1 loc(#loc48)
  } loc(#loc48)
  handshake.func @_Z15mat3x3_mult_dsaPKfS0_Pfj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: none loc(fused<#di_subprogram4>[#loc14]), ...) -> none attributes {argNames = ["input_mat_a", "input_mat_b", "output_mat_c", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc48)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 4 : index} : index loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 9 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %0 {value = 4294967295 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %0 {value = 12 : i64} : i64 loc(#loc2)
    %7 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %8 = handshake.constant %0 {value = 3 : i64} : i64 loc(#loc2)
    %9 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %9, %0 : none loc(#loc52)
    %10 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc52)
    %11 = arith.index_cast %3 : i64 to index loc(#loc52)
    %12 = arith.index_cast %arg3 : i32 to index loc(#loc52)
    %index, %willContinue = dataflow.stream %11, %10, %12 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc52)
    %13 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc52)
    %14 = arith.index_cast %afterValue : index to i64 loc(#loc52)
    %15 = arith.muli %14, %4 : i64 loc(#loc56)
    %16 = arith.andi %15, %5 : i64 loc(#loc56)
    %17 = arith.index_cast %16 : i64 to index loc(#loc56)
    %18 = handshake.constant %13 {value = 1 : index} : index loc(#loc59)
    %19 = arith.index_cast %8 : i64 to index loc(#loc59)
    %index_0, %willContinue_1 = dataflow.stream %11, %18, %19 {step_op = "+=", stop_cond = "!="} loc(#loc59)
    %afterValue_2, %afterCond_3 = dataflow.gate %index_0, %willContinue_1 : index, i1 -> index, i1 loc(#loc59)
    %20 = dataflow.invariant %afterCond_3, %13 : i1, none -> none loc(#loc59)
    %21 = arith.index_cast %afterValue_2 : index to i64 loc(#loc59)
    %22 = arith.muli %21, %6 : i64 loc(#loc2)
    %23 = arith.index_cast %22 : i64 to index loc(#loc63)
    %24 = arith.divui %23, %2 : index loc(#loc63)
    %25 = dataflow.invariant %afterCond_3, %17 : i1, index -> index loc(#loc63)
    %26 = arith.addi %25, %24 : index loc(#loc63)
    %27 = handshake.constant %20 {value = 1 : index} : index loc(#loc63)
    %index_4, %willContinue_5 = dataflow.stream %11, %27, %19 {step_op = "+=", stop_cond = "!="} loc(#loc63)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc63)
    %28 = dataflow.invariant %afterCond_7, %20 : i1, none -> none loc(#loc63)
    %29 = arith.addi %25, %afterValue_6 : index loc(#loc69)
    %30 = handshake.constant %28 {value = 1 : index} : index loc(#loc69)
    %index_8, %willContinue_9 = dataflow.stream %11, %30, %19 {step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc69)
    %31 = dataflow.carry %willContinue_9, %7, %40 : i1, f32, f32 -> f32 loc(#loc69)
    %afterValue_12, %afterCond_13 = dataflow.gate %31, %willContinue_9 : f32, i1 -> f32, i1 loc(#loc69)
    handshake.sink %afterCond_13 : i1 loc(#loc69)
    %trueResult_14, %falseResult_15 = handshake.cond_br %willContinue_9, %31 : f32 loc(#loc69)
    %32 = arith.index_cast %afterValue_10 : index to i64 loc(#loc69)
    %33 = dataflow.invariant %afterCond_7, %26 : i1, index -> index loc(#loc72)
    %34 = arith.addi %33, %afterValue_10 : index loc(#loc72)
    %dataResult, %addressResults = handshake.load [%34] %43#0, %48 : index, f32 loc(#loc72)
    %35 = arith.muli %32, %6 : i64 loc(#loc72)
    %36 = arith.index_cast %35 : i64 to index loc(#loc72)
    %37 = arith.divui %36, %2 : index loc(#loc72)
    %38 = dataflow.invariant %afterCond_11, %29 : i1, index -> index loc(#loc72)
    %39 = arith.addi %38, %37 : index loc(#loc72)
    %dataResult_16, %addressResults_17 = handshake.load [%39] %42#0, %60 : index, f32 loc(#loc72)
    %40 = math.fma %dataResult, %dataResult_16, %afterValue_12 : f32 loc(#loc72)
    %41 = arith.addi %33, %afterValue_6 : index loc(#loc66)
    %dataResult_18, %addressResult = handshake.store [%41] %falseResult_15, %55 : index, f32 loc(#loc66)
    %42:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_17) {id = 0 : i32} : (index) -> (f32, none) loc(#loc48)
    %43:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (f32, none) loc(#loc48)
    %44 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_18, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc48)
    %45 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc52)
    %46 = dataflow.carry %willContinue_1, %45, %trueResult_23 : i1, none, none -> none loc(#loc59)
    %47 = dataflow.carry %willContinue_5, %46, %trueResult_21 : i1, none, none -> none loc(#loc63)
    %48 = dataflow.carry %willContinue_9, %47, %trueResult_19 : i1, none, none -> none loc(#loc69)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue_9, %43#1 : none loc(#loc69)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_5, %falseResult_20 : none loc(#loc63)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_1, %falseResult_22 : none loc(#loc59)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %falseResult_24 : none loc(#loc52)
    %49 = handshake.constant %0 {value = 0 : index} : index loc(#loc52)
    %50 = handshake.constant %0 {value = 1 : index} : index loc(#loc52)
    %51 = arith.select %9, %50, %49 : index loc(#loc52)
    %52 = handshake.mux %51 [%falseResult_26, %trueResult] : index, none loc(#loc52)
    %53 = dataflow.carry %willContinue, %falseResult, %trueResult_31 : i1, none, none -> none loc(#loc52)
    %54 = dataflow.carry %willContinue_1, %53, %trueResult_29 : i1, none, none -> none loc(#loc59)
    %55 = dataflow.carry %willContinue_5, %54, %trueResult_27 : i1, none, none -> none loc(#loc63)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue_5, %44 : none loc(#loc63)
    %trueResult_29, %falseResult_30 = handshake.cond_br %willContinue_1, %falseResult_28 : none loc(#loc59)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue, %falseResult_30 : none loc(#loc52)
    %56 = handshake.mux %51 [%falseResult_32, %trueResult] : index, none loc(#loc52)
    %57 = dataflow.carry %willContinue, %falseResult, %trueResult_39 : i1, none, none -> none loc(#loc52)
    %58 = dataflow.carry %willContinue_1, %57, %trueResult_37 : i1, none, none -> none loc(#loc59)
    %59 = dataflow.carry %willContinue_5, %58, %trueResult_35 : i1, none, none -> none loc(#loc63)
    %60 = dataflow.carry %willContinue_9, %59, %trueResult_33 : i1, none, none -> none loc(#loc69)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_9, %42#1 : none loc(#loc69)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_5, %falseResult_34 : none loc(#loc63)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_1, %falseResult_36 : none loc(#loc59)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue, %falseResult_38 : none loc(#loc52)
    %61 = handshake.mux %51 [%falseResult_40, %trueResult] : index, none loc(#loc52)
    %62 = handshake.join %52, %56, %61 : none, none, none loc(#loc48)
    handshake.return %62 : none loc(#loc49)
  } loc(#loc48)
  func.func @_Z15mat3x3_mult_cpuPKfS0_Pfj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc23]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc23]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc23]), %arg3: i32 loc(fused<#di_subprogram5>[#loc23])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c9_i64 = arith.constant 9 : i64 loc(#loc2)
    %c4294967295_i64 = arith.constant 4294967295 : i64 loc(#loc2)
    %c12_i64 = arith.constant 12 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c3_i64 = arith.constant 3 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc55)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc55)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.muli %arg4, %c9_i64 : i64 loc(#loc57)
        %4 = arith.andi %3, %c4294967295_i64 : i64 loc(#loc57)
        %5 = arith.index_cast %4 : i64 to index loc(#loc57)
        %6 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
          %9 = arith.muli %arg5, %c12_i64 : i64 loc(#loc2)
          %10 = arith.index_cast %9 : i64 to index loc(#loc64)
          %11 = arith.divui %10, %c4 : index loc(#loc64)
          %12 = arith.addi %5, %11 : index loc(#loc64)
          %13 = scf.while (%arg6 = %c0_i64) : (i64) -> i64 {
            %16 = arith.index_cast %arg6 : i64 to index loc(#loc70)
            %17 = arith.addi %5, %16 : index loc(#loc70)
            %18:2 = scf.while (%arg7 = %c0_i64, %arg8 = %cst) : (i64, f32) -> (i64, f32) {
              %22 = arith.index_cast %arg7 : i64 to index loc(#loc73)
              %23 = arith.addi %12, %22 : index loc(#loc73)
              %24 = memref.load %arg0[%23] : memref<?xf32, strided<[1], offset: ?>> loc(#loc73)
              %25 = arith.muli %arg7, %c12_i64 : i64 loc(#loc73)
              %26 = arith.index_cast %25 : i64 to index loc(#loc73)
              %27 = arith.divui %26, %c4 : index loc(#loc73)
              %28 = arith.addi %17, %27 : index loc(#loc73)
              %29 = memref.load %arg1[%28] : memref<?xf32, strided<[1], offset: ?>> loc(#loc73)
              %30 = math.fma %24, %29, %arg8 : f32 loc(#loc73)
              %31 = arith.addi %arg7, %c1_i64 : i64 loc(#loc71)
              %32 = arith.cmpi ne, %31, %c3_i64 : i64 loc(#loc74)
              scf.condition(%32) %31, %30 : i64, f32 loc(#loc70)
            } do {
            ^bb0(%arg7: i64 loc(fused<#di_lexical_block49>[#loc28]), %arg8: f32 loc(fused<#di_lexical_block49>[#loc28])):
              scf.yield %arg7, %arg8 : i64, f32 loc(#loc70)
            } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc70)
            %19 = arith.addi %12, %16 : index loc(#loc67)
            memref.store %18#1, %arg2[%19] : memref<?xf32, strided<[1], offset: ?>> loc(#loc67)
            %20 = arith.addi %arg6, %c1_i64 : i64 loc(#loc65)
            %21 = arith.cmpi ne, %20, %c3_i64 : i64 loc(#loc68)
            scf.condition(%21) %20 : i64 loc(#loc64)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block43>[#loc27])):
            scf.yield %arg6 : i64 loc(#loc64)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc64)
          %14 = arith.addi %arg5, %c1_i64 : i64 loc(#loc61)
          %15 = arith.cmpi ne, %14, %c3_i64 : i64 loc(#loc62)
          scf.condition(%15) %14 : i64 loc(#loc60)
        } do {
        ^bb0(%arg5: i64 loc(fused<#di_lexical_block37>[#loc26])):
          scf.yield %arg5 : i64 loc(#loc60)
        } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc60)
        %7 = arith.addi %arg4, %c1_i64 : i64 loc(#loc55)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc58)
        scf.condition(%8) %7 : i64 loc(#loc53)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block31>[#loc24])):
        scf.yield %arg4 : i64 loc(#loc53)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc53)
    } loc(#loc53)
    return loc(#loc51)
  } loc(#loc50)
} loc(#loc)
#loc = loc("tests/app/mat3x3_mult/main.cpp":0:0)
#loc1 = loc("tests/app/mat3x3_mult/main.cpp":6:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/mat3x3_mult/main.cpp":14:0)
#loc5 = loc("tests/app/mat3x3_mult/main.cpp":15:0)
#loc6 = loc("tests/app/mat3x3_mult/main.cpp":23:0)
#loc7 = loc("tests/app/mat3x3_mult/main.cpp":26:0)
#loc9 = loc("tests/app/mat3x3_mult/main.cpp":30:0)
#loc10 = loc("tests/app/mat3x3_mult/main.cpp":31:0)
#loc11 = loc("tests/app/mat3x3_mult/main.cpp":32:0)
#loc12 = loc("tests/app/mat3x3_mult/main.cpp":36:0)
#loc13 = loc("tests/app/mat3x3_mult/main.cpp":38:0)
#loc15 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":44:0)
#loc16 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":45:0)
#loc17 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":50:0)
#loc18 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":51:0)
#loc19 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":53:0)
#loc20 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":54:0)
#loc21 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":56:0)
#loc22 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":60:0)
#loc25 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":19:0)
#loc29 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":28:0)
#loc30 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":30:0)
#loc31 = loc("tests/app/mat3x3_mult/mat3x3_mult.cpp":34:0)
#loc32 = loc(fused<#di_subprogram3>[#loc1])
#loc33 = loc(fused<#di_subprogram3>[#loc6])
#loc34 = loc(fused<#di_subprogram3>[#loc7])
#loc35 = loc(fused<#di_subprogram3>[#loc12])
#loc36 = loc(fused<#di_subprogram3>[#loc13])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 13>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 29>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 13>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 29>
#loc39 = loc(fused<#di_lexical_block16>[#loc3])
#loc40 = loc(fused<#di_lexical_block17>[#loc8])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 30>
#loc41 = loc(fused<#di_lexical_block20>[#loc4])
#loc42 = loc(fused<#di_lexical_block20>[#loc5])
#loc43 = loc(fused[#loc37, #loc39])
#loc44 = loc(fused[#loc38, #loc40])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 30>
#loc45 = loc(fused<#di_lexical_block24>[#loc9])
#loc46 = loc(fused<#di_lexical_block27>[#loc10])
#loc47 = loc(fused<#di_lexical_block27>[#loc11])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 44>
#loc49 = loc(fused<#di_subprogram4>[#loc22])
#loc51 = loc(fused<#di_subprogram5>[#loc31])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 44>
#loc52 = loc(fused<#di_lexical_block30>[#loc15])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file1, line = 44>
#loc54 = loc(fused<#di_lexical_block32>[#loc15])
#loc55 = loc(fused<#di_lexical_block33>[#loc24])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file1, line = 50>
#loc56 = loc(fused<#di_lexical_block34>[#loc16])
#loc57 = loc(fused<#di_lexical_block35>[#loc25])
#loc58 = loc(fused[#loc53, #loc55])
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file1, line = 50>
#loc59 = loc(fused<#di_lexical_block36>[#loc17])
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file1, line = 50>
#loc61 = loc(fused<#di_lexical_block39>[#loc26])
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file1, line = 51>
#loc62 = loc(fused[#loc60, #loc61])
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_lexical_block42, file = #di_file1, line = 51>
#loc63 = loc(fused<#di_lexical_block42>[#loc18])
#di_lexical_block46 = #llvm.di_lexical_block<scope = #di_lexical_block44, file = #di_file1, line = 51>
#loc65 = loc(fused<#di_lexical_block45>[#loc27])
#di_lexical_block48 = #llvm.di_lexical_block<scope = #di_lexical_block46, file = #di_file1, line = 53>
#loc66 = loc(fused<#di_lexical_block46>[#loc21])
#loc67 = loc(fused<#di_lexical_block47>[#loc30])
#loc68 = loc(fused[#loc64, #loc65])
#di_lexical_block50 = #llvm.di_lexical_block<scope = #di_lexical_block48, file = #di_file1, line = 53>
#di_lexical_block51 = #llvm.di_lexical_block<scope = #di_lexical_block49, file = #di_file1, line = 27>
#loc69 = loc(fused<#di_lexical_block48>[#loc19])
#di_lexical_block52 = #llvm.di_lexical_block<scope = #di_lexical_block50, file = #di_file1, line = 53>
#di_lexical_block53 = #llvm.di_lexical_block<scope = #di_lexical_block51, file = #di_file1, line = 27>
#loc71 = loc(fused<#di_lexical_block51>[#loc28])
#loc72 = loc(fused<#di_lexical_block52>[#loc20])
#loc73 = loc(fused<#di_lexical_block53>[#loc29])
#loc74 = loc(fused[#loc70, #loc71])
