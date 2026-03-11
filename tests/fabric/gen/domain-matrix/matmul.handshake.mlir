#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/matmul/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/matmul/matmul.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/matmul/main.cpp":19:0)
#loc5 = loc("tests/app/matmul/main.cpp":22:0)
#loc9 = loc("tests/app/matmul/main.cpp":33:0)
#loc15 = loc("tests/app/matmul/matmul.cpp":35:0)
#loc22 = loc("tests/app/matmul/matmul.cpp":15:0)
#loc23 = loc("tests/app/matmul/matmul.cpp":21:0)
#loc24 = loc("tests/app/matmul/matmul.cpp":22:0)
#loc25 = loc("tests/app/matmul/matmul.cpp":24:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 19>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 22>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 33>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 42>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 21>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 42>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 21>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 16384, elements = #llvm.di_subrange<count = 512 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 24576, elements = #llvm.di_subrange<count = 768 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 12288, elements = #llvm.di_subrange<count = 384 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 42>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 21>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 19, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 22, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 33, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 42, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 21, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 44>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 22>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "K", file = #di_file, line = 8, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "B", file = #di_file, line = 12, type = #di_composite_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_C", file = #di_file, line = 15, type = #di_composite_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_C", file = #di_file, line = 16, type = #di_composite_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file1, line = 38, arg = 4, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 39, arg = 5, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "K", file = #di_file1, line = 40, arg = 6, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 18, arg = 4, type = #di_derived_type2>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 19, arg = 5, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "K", file = #di_file1, line = 20, arg = 6, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 44>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 22>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "C", file = #di_file1, line = 37, arg = 3, type = #di_derived_type5>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file1, line = 44, type = #di_derived_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "C", file = #di_file1, line = 17, arg = 3, type = #di_derived_type5>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "j", file = #di_file1, line = 22, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 19>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 22>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 33>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 44>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file1, line = 22>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file1, line = 35, arg = 1, type = #di_derived_type6>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "B", file = #di_file1, line = 36, arg = 2, type = #di_derived_type6>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 15, arg = 1, type = #di_derived_type6>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram2, name = "B", file = #di_file1, line = 16, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2, #di_derived_type2>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 47>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 24>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_lexical_block16, name = "sum", file = #di_file1, line = 45, type = #di_derived_type1>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block17, name = "sum", file = #di_file1, line = 23, type = #di_derived_type1>
#loc34 = loc(fused<#di_lexical_block13>[#loc3])
#loc35 = loc(fused<#di_lexical_block14>[#loc5])
#loc36 = loc(fused<#di_lexical_block15>[#loc9])
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block21, name = "k", file = #di_file1, line = 47, type = #di_derived_type1>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_lexical_block22, name = "k", file = #di_file1, line = 24, type = #di_derived_type1>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "matmul_dsa", linkageName = "_Z10matmul_dsaPKjS0_Pjjjj", file = #di_file1, line = 35, scopeLine = 40, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable22, #di_local_variable23, #di_local_variable18, #di_local_variable12, #di_local_variable13, #di_local_variable14, #di_local_variable3, #di_local_variable19, #di_local_variable26, #di_local_variable28>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "matmul_cpu", linkageName = "_Z10matmul_cpuPKjS0_Pjjjj", file = #di_file1, line = 15, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable20, #di_local_variable15, #di_local_variable16, #di_local_variable17, #di_local_variable4, #di_local_variable21, #di_local_variable27, #di_local_variable29>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 21>
#loc46 = loc(fused<#di_subprogram4>[#loc15])
#loc48 = loc(fused<#di_subprogram5>[#loc22])
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file1, line = 21>
#loc53 = loc(fused<#di_lexical_block29>[#loc23])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file1, line = 21>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file1, line = 22>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file1, line = 22>
#loc58 = loc(fused<#di_lexical_block35>[#loc24])
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file1, line = 22>
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file1, line = 24>
#loc64 = loc(fused<#di_lexical_block41>[#loc25])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[109, 97, 116, 109, 117, 108, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[109, 97, 116, 109, 117, 108, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 52, 44, 98, 108, 111, 99, 107, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 109, 97, 116, 109, 117, 108, 47, 109, 97, 116, 109, 117, 108, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc29)
    %false = arith.constant false loc(#loc29)
    %0 = seq.const_clock  low loc(#loc29)
    %c2_i32 = arith.constant 2 : i32 loc(#loc29)
    %1 = ub.poison : i64 loc(#loc29)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c384_i64 = arith.constant 384 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c512_i64 = arith.constant 512 : i64 loc(#loc2)
    %c768_i64 = arith.constant 768 : i64 loc(#loc2)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c24_i32 = arith.constant 24 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<512xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<768xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<384xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<384xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc40)
      %12 = arith.remui %11, %c10_i32 : i32 loc(#loc40)
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc40)
      memref.store %12, %alloca[%13] : memref<512xi32> loc(#loc40)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc37)
      %15 = arith.cmpi ne, %14, %c512_i64 : i64 loc(#loc41)
      scf.condition(%15) %14 : i64 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc34)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc34)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.addi %arg0, %c1_i64 : i64 loc(#loc42)
      %12 = arith.trunci %11 : i64 to i32 loc(#loc42)
      %13 = arith.remui %12, %c10_i32 : i32 loc(#loc42)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      memref.store %13, %alloca_0[%14] : memref<768xi32> loc(#loc42)
      %15 = arith.cmpi ne, %11, %c768_i64 : i64 loc(#loc43)
      scf.condition(%15) %11 : i64 loc(#loc35)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block14>[#loc5])):
      scf.yield %arg0 : i64 loc(#loc35)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc35)
    %cast = memref.cast %alloca : memref<512xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_3 = memref.cast %alloca_0 : memref<768xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_4 = memref.cast %alloca_1 : memref<384xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    call @_Z10matmul_cpuPKjS0_Pjjjj(%cast, %cast_3, %cast_4, %c16_i32, %c32_i32, %c24_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32, i32) -> () loc(#loc30)
    %cast_5 = memref.cast %alloca_2 : memref<384xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc31)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c16_i32, %true : i32 loc(#loc31)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc31)
    %chanOutput_14, %ready_15 = esi.wrap.vr %c24_i32, %true : i32 loc(#loc31)
    %chanOutput_16, %ready_17 = esi.wrap.vr %true, %true : i1 loc(#loc31)
    %6 = handshake.esi_instance @_Z10matmul_dsaPKjS0_Pjjjj_esi "_Z10matmul_dsaPKjS0_Pjjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc31)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc31)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      %12 = memref.load %alloca_1[%11] : memref<384xi32> loc(#loc45)
      %13 = memref.load %alloca_2[%11] : memref<384xi32> loc(#loc45)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc45)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
        %18 = arith.cmpi eq, %17, %c384_i64 : i64 loc(#loc39)
        %19 = arith.extui %18 : i1 to i32 loc(#loc36)
        %20 = arith.cmpi ne, %17, %c384_i64 : i64 loc(#loc44)
        %21 = arith.extui %20 : i1 to i32 loc(#loc36)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc45)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc45)
      } loc(#loc45)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc36)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc36)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc9]), %arg1: i1 loc(fused<#di_lexical_block15>[#loc9]), %arg2: i32 loc(fused<#di_lexical_block15>[#loc9])):
      scf.yield %arg0 : i64 loc(#loc36)
    } loc(#loc36)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc36)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc36)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc50)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc50)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc36)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc32)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc32)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc32)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc32)
    } loc(#loc2)
    return %10 : i32 loc(#loc33)
  } loc(#loc29)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10matmul_dsaPKjS0_Pjjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: i32 loc(fused<#di_subprogram4>[#loc15]), %arg6: i1 loc(fused<#di_subprogram4>[#loc15]), ...) -> i1 attributes {argNames = ["A", "B", "C", "M", "N", "K", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg6 : i1 loc(#loc46)
    %1 = handshake.join %0 : none loc(#loc46)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc52)
    %5 = arith.cmpi eq, %arg5, %2 : i32 loc(#loc2)
    %6 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc52)
    %8 = arith.index_cast %2 : i32 to index loc(#loc52)
    %9 = arith.index_cast %arg3 : i32 to index loc(#loc52)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc52)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc52)
    %11 = arith.index_cast %afterValue : index to i32 loc(#loc52)
    %12 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc57)
    %trueResult_0, %falseResult_1 = handshake.cond_br %12, %10 : none loc(#loc57)
    %13 = arith.muli %11, %arg4 : i32 loc(#loc2)
    %14 = arith.muli %11, %arg5 : i32 loc(#loc2)
    %15 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc57)
    %16 = arith.index_cast %3 : i64 to index loc(#loc57)
    %17 = arith.index_cast %arg5 : i32 to index loc(#loc57)
    %index_2, %willContinue_3 = dataflow.stream %16, %15, %17 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc57)
    %18 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc57)
    %19 = arith.index_cast %afterValue_4 : index to i64 loc(#loc57)
    %20 = dataflow.invariant %afterCond, %6 : i1, i1 -> i1 loc(#loc63)
    %trueResult_6, %falseResult_7 = handshake.cond_br %20, %18 : none loc(#loc63)
    %21 = arith.trunci %19 : i64 to i32 loc(#loc60)
    %22 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc63)
    %23 = arith.index_cast %arg4 : i32 to index loc(#loc63)
    %index_8, %willContinue_9 = dataflow.stream %16, %22, %23 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "!="} loc(#loc63)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc63)
    %24 = dataflow.carry %willContinue_9, %2, %37 : i1, i32, i32 -> i32 loc(#loc63)
    %afterValue_12, %afterCond_13 = dataflow.gate %24, %willContinue_9 : i32, i1 -> i32, i1 loc(#loc63)
    handshake.sink %afterCond_13 : i1 loc(#loc63)
    %trueResult_14, %falseResult_15 = handshake.cond_br %willContinue_9, %24 : i32 loc(#loc63)
    %25 = arith.index_cast %afterValue_10 : index to i64 loc(#loc63)
    %26 = arith.trunci %25 : i64 to i32 loc(#loc66)
    %27 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc66)
    %28 = arith.addi %27, %26 : i32 loc(#loc66)
    %29 = arith.extui %28 : i32 to i64 loc(#loc66)
    %30 = arith.index_cast %29 : i64 to index loc(#loc66)
    %dataResult, %addressResults = handshake.load [%30] %47#0, %51 : index, i32 loc(#loc66)
    %31 = arith.muli %arg5, %26 : i32 loc(#loc66)
    %32 = dataflow.invariant %afterCond_11, %21 : i1, i32 -> i32 loc(#loc66)
    %33 = arith.addi %31, %32 : i32 loc(#loc66)
    %34 = arith.extui %33 : i32 to i64 loc(#loc66)
    %35 = arith.index_cast %34 : i64 to index loc(#loc66)
    %dataResult_16, %addressResults_17 = handshake.load [%35] %46#0, %73 : index, i32 loc(#loc66)
    %36 = arith.muli %dataResult_16, %dataResult : i32 loc(#loc66)
    %37 = arith.addi %36, %afterValue_12 : i32 loc(#loc66)
    %38 = handshake.constant %18 {value = 0 : index} : index loc(#loc63)
    %39 = handshake.constant %18 {value = 1 : index} : index loc(#loc63)
    %40 = arith.select %20, %39, %38 : index loc(#loc63)
    %41 = handshake.mux %40 [%falseResult_15, %2] : index, i32 loc(#loc63)
    %42 = dataflow.invariant %afterCond_5, %14 : i1, i32 -> i32 loc(#loc60)
    %43 = arith.addi %42, %21 : i32 loc(#loc60)
    %44 = arith.extui %43 : i32 to i64 loc(#loc60)
    %45 = arith.index_cast %44 : i64 to index loc(#loc60)
    %dataResult_18, %addressResult = handshake.store [%45] %41, %65 : index, i32 loc(#loc60)
    %46:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_17) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %47:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %48 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_18, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc46)
    %49 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc52)
    %trueResult_19, %falseResult_20 = handshake.cond_br %12, %49 : none loc(#loc57)
    %50 = dataflow.carry %willContinue_3, %falseResult_20, %trueResult_25 : i1, none, none -> none loc(#loc57)
    %trueResult_21, %falseResult_22 = handshake.cond_br %20, %50 : none loc(#loc63)
    %51 = dataflow.carry %willContinue_9, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc63)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_9, %47#1 : none loc(#loc63)
    %52 = handshake.constant %50 {value = 0 : index} : index loc(#loc63)
    %53 = handshake.constant %50 {value = 1 : index} : index loc(#loc63)
    %54 = arith.select %20, %53, %52 : index loc(#loc63)
    %55 = handshake.mux %54 [%falseResult_24, %trueResult_21] : index, none loc(#loc63)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue_3, %55 : none loc(#loc57)
    %56 = handshake.constant %49 {value = 0 : index} : index loc(#loc57)
    %57 = handshake.constant %49 {value = 1 : index} : index loc(#loc57)
    %58 = arith.select %12, %57, %56 : index loc(#loc57)
    %59 = handshake.mux %58 [%falseResult_26, %trueResult_19] : index, none loc(#loc57)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %59 : none loc(#loc52)
    %60 = handshake.constant %1 {value = 0 : index} : index loc(#loc52)
    %61 = handshake.constant %1 {value = 1 : index} : index loc(#loc52)
    %62 = arith.select %4, %61, %60 : index loc(#loc52)
    %63 = handshake.mux %62 [%falseResult_28, %trueResult] : index, none loc(#loc52)
    %64 = dataflow.carry %willContinue, %falseResult, %trueResult_33 : i1, none, none -> none loc(#loc52)
    %trueResult_29, %falseResult_30 = handshake.cond_br %12, %64 : none loc(#loc57)
    %65 = dataflow.carry %willContinue_3, %falseResult_30, %trueResult_31 : i1, none, none -> none loc(#loc57)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue_3, %48 : none loc(#loc57)
    %66 = handshake.constant %64 {value = 0 : index} : index loc(#loc57)
    %67 = handshake.constant %64 {value = 1 : index} : index loc(#loc57)
    %68 = arith.select %12, %67, %66 : index loc(#loc57)
    %69 = handshake.mux %68 [%falseResult_32, %trueResult_29] : index, none loc(#loc57)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue, %69 : none loc(#loc52)
    %70 = handshake.mux %62 [%falseResult_34, %trueResult] : index, none loc(#loc52)
    %71 = dataflow.carry %willContinue, %falseResult, %trueResult_43 : i1, none, none -> none loc(#loc52)
    %trueResult_35, %falseResult_36 = handshake.cond_br %12, %71 : none loc(#loc57)
    %72 = dataflow.carry %willContinue_3, %falseResult_36, %trueResult_41 : i1, none, none -> none loc(#loc57)
    %trueResult_37, %falseResult_38 = handshake.cond_br %20, %72 : none loc(#loc63)
    %73 = dataflow.carry %willContinue_9, %falseResult_38, %trueResult_39 : i1, none, none -> none loc(#loc63)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue_9, %46#1 : none loc(#loc63)
    %74 = handshake.constant %72 {value = 0 : index} : index loc(#loc63)
    %75 = handshake.constant %72 {value = 1 : index} : index loc(#loc63)
    %76 = arith.select %20, %75, %74 : index loc(#loc63)
    %77 = handshake.mux %76 [%falseResult_40, %trueResult_37] : index, none loc(#loc63)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue_3, %77 : none loc(#loc57)
    %78 = handshake.constant %71 {value = 0 : index} : index loc(#loc57)
    %79 = handshake.constant %71 {value = 1 : index} : index loc(#loc57)
    %80 = arith.select %12, %79, %78 : index loc(#loc57)
    %81 = handshake.mux %80 [%falseResult_42, %trueResult_35] : index, none loc(#loc57)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue, %81 : none loc(#loc52)
    %82 = handshake.mux %62 [%falseResult_44, %trueResult] : index, none loc(#loc52)
    %83 = handshake.join %63, %70, %82 : none, none, none loc(#loc46)
    %84 = handshake.constant %83 {value = true} : i1 loc(#loc46)
    handshake.return %84 : i1 loc(#loc46)
  } loc(#loc46)
  handshake.func @_Z10matmul_dsaPKjS0_Pjjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: i32 loc(fused<#di_subprogram4>[#loc15]), %arg6: none loc(fused<#di_subprogram4>[#loc15]), ...) -> none attributes {argNames = ["A", "B", "C", "M", "N", "K", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg6 : none loc(#loc46)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc52)
    %4 = arith.cmpi eq, %arg5, %1 : i32 loc(#loc2)
    %5 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc52)
    %7 = arith.index_cast %1 : i32 to index loc(#loc52)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc52)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc52)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc52)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc52)
    %10 = arith.index_cast %afterValue : index to i32 loc(#loc52)
    %11 = dataflow.invariant %afterCond, %4 : i1, i1 -> i1 loc(#loc57)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %9 : none loc(#loc57)
    %12 = arith.muli %10, %arg4 : i32 loc(#loc2)
    %13 = arith.muli %10, %arg5 : i32 loc(#loc2)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc57)
    %15 = arith.index_cast %2 : i64 to index loc(#loc57)
    %16 = arith.index_cast %arg5 : i32 to index loc(#loc57)
    %index_2, %willContinue_3 = dataflow.stream %15, %14, %16 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc57)
    %17 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc57)
    %18 = arith.index_cast %afterValue_4 : index to i64 loc(#loc57)
    %19 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc63)
    %trueResult_6, %falseResult_7 = handshake.cond_br %19, %17 : none loc(#loc63)
    %20 = arith.trunci %18 : i64 to i32 loc(#loc60)
    %21 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc63)
    %22 = arith.index_cast %arg4 : i32 to index loc(#loc63)
    %index_8, %willContinue_9 = dataflow.stream %15, %21, %22 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "!="} loc(#loc63)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc63)
    %23 = dataflow.carry %willContinue_9, %1, %36 : i1, i32, i32 -> i32 loc(#loc63)
    %afterValue_12, %afterCond_13 = dataflow.gate %23, %willContinue_9 : i32, i1 -> i32, i1 loc(#loc63)
    handshake.sink %afterCond_13 : i1 loc(#loc63)
    %trueResult_14, %falseResult_15 = handshake.cond_br %willContinue_9, %23 : i32 loc(#loc63)
    %24 = arith.index_cast %afterValue_10 : index to i64 loc(#loc63)
    %25 = arith.trunci %24 : i64 to i32 loc(#loc66)
    %26 = dataflow.invariant %afterCond_5, %12 : i1, i32 -> i32 loc(#loc66)
    %27 = arith.addi %26, %25 : i32 loc(#loc66)
    %28 = arith.extui %27 : i32 to i64 loc(#loc66)
    %29 = arith.index_cast %28 : i64 to index loc(#loc66)
    %dataResult, %addressResults = handshake.load [%29] %46#0, %50 : index, i32 loc(#loc66)
    %30 = arith.muli %arg5, %25 : i32 loc(#loc66)
    %31 = dataflow.invariant %afterCond_11, %20 : i1, i32 -> i32 loc(#loc66)
    %32 = arith.addi %30, %31 : i32 loc(#loc66)
    %33 = arith.extui %32 : i32 to i64 loc(#loc66)
    %34 = arith.index_cast %33 : i64 to index loc(#loc66)
    %dataResult_16, %addressResults_17 = handshake.load [%34] %45#0, %72 : index, i32 loc(#loc66)
    %35 = arith.muli %dataResult_16, %dataResult : i32 loc(#loc66)
    %36 = arith.addi %35, %afterValue_12 : i32 loc(#loc66)
    %37 = handshake.constant %17 {value = 0 : index} : index loc(#loc63)
    %38 = handshake.constant %17 {value = 1 : index} : index loc(#loc63)
    %39 = arith.select %19, %38, %37 : index loc(#loc63)
    %40 = handshake.mux %39 [%falseResult_15, %1] : index, i32 loc(#loc63)
    %41 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc60)
    %42 = arith.addi %41, %20 : i32 loc(#loc60)
    %43 = arith.extui %42 : i32 to i64 loc(#loc60)
    %44 = arith.index_cast %43 : i64 to index loc(#loc60)
    %dataResult_18, %addressResult = handshake.store [%44] %40, %64 : index, i32 loc(#loc60)
    %45:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_17) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %46:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %47 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_18, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc46)
    %48 = dataflow.carry %willContinue, %falseResult, %trueResult_27 : i1, none, none -> none loc(#loc52)
    %trueResult_19, %falseResult_20 = handshake.cond_br %11, %48 : none loc(#loc57)
    %49 = dataflow.carry %willContinue_3, %falseResult_20, %trueResult_25 : i1, none, none -> none loc(#loc57)
    %trueResult_21, %falseResult_22 = handshake.cond_br %19, %49 : none loc(#loc63)
    %50 = dataflow.carry %willContinue_9, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc63)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_9, %46#1 : none loc(#loc63)
    %51 = handshake.constant %49 {value = 0 : index} : index loc(#loc63)
    %52 = handshake.constant %49 {value = 1 : index} : index loc(#loc63)
    %53 = arith.select %19, %52, %51 : index loc(#loc63)
    %54 = handshake.mux %53 [%falseResult_24, %trueResult_21] : index, none loc(#loc63)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue_3, %54 : none loc(#loc57)
    %55 = handshake.constant %48 {value = 0 : index} : index loc(#loc57)
    %56 = handshake.constant %48 {value = 1 : index} : index loc(#loc57)
    %57 = arith.select %11, %56, %55 : index loc(#loc57)
    %58 = handshake.mux %57 [%falseResult_26, %trueResult_19] : index, none loc(#loc57)
    %trueResult_27, %falseResult_28 = handshake.cond_br %willContinue, %58 : none loc(#loc52)
    %59 = handshake.constant %0 {value = 0 : index} : index loc(#loc52)
    %60 = handshake.constant %0 {value = 1 : index} : index loc(#loc52)
    %61 = arith.select %3, %60, %59 : index loc(#loc52)
    %62 = handshake.mux %61 [%falseResult_28, %trueResult] : index, none loc(#loc52)
    %63 = dataflow.carry %willContinue, %falseResult, %trueResult_33 : i1, none, none -> none loc(#loc52)
    %trueResult_29, %falseResult_30 = handshake.cond_br %11, %63 : none loc(#loc57)
    %64 = dataflow.carry %willContinue_3, %falseResult_30, %trueResult_31 : i1, none, none -> none loc(#loc57)
    %trueResult_31, %falseResult_32 = handshake.cond_br %willContinue_3, %47 : none loc(#loc57)
    %65 = handshake.constant %63 {value = 0 : index} : index loc(#loc57)
    %66 = handshake.constant %63 {value = 1 : index} : index loc(#loc57)
    %67 = arith.select %11, %66, %65 : index loc(#loc57)
    %68 = handshake.mux %67 [%falseResult_32, %trueResult_29] : index, none loc(#loc57)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue, %68 : none loc(#loc52)
    %69 = handshake.mux %61 [%falseResult_34, %trueResult] : index, none loc(#loc52)
    %70 = dataflow.carry %willContinue, %falseResult, %trueResult_43 : i1, none, none -> none loc(#loc52)
    %trueResult_35, %falseResult_36 = handshake.cond_br %11, %70 : none loc(#loc57)
    %71 = dataflow.carry %willContinue_3, %falseResult_36, %trueResult_41 : i1, none, none -> none loc(#loc57)
    %trueResult_37, %falseResult_38 = handshake.cond_br %19, %71 : none loc(#loc63)
    %72 = dataflow.carry %willContinue_9, %falseResult_38, %trueResult_39 : i1, none, none -> none loc(#loc63)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue_9, %45#1 : none loc(#loc63)
    %73 = handshake.constant %71 {value = 0 : index} : index loc(#loc63)
    %74 = handshake.constant %71 {value = 1 : index} : index loc(#loc63)
    %75 = arith.select %19, %74, %73 : index loc(#loc63)
    %76 = handshake.mux %75 [%falseResult_40, %trueResult_37] : index, none loc(#loc63)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue_3, %76 : none loc(#loc57)
    %77 = handshake.constant %70 {value = 0 : index} : index loc(#loc57)
    %78 = handshake.constant %70 {value = 1 : index} : index loc(#loc57)
    %79 = arith.select %11, %78, %77 : index loc(#loc57)
    %80 = handshake.mux %79 [%falseResult_42, %trueResult_35] : index, none loc(#loc57)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue, %80 : none loc(#loc52)
    %81 = handshake.mux %61 [%falseResult_44, %trueResult] : index, none loc(#loc52)
    %82 = handshake.join %62, %69, %81 : none, none, none loc(#loc46)
    handshake.return %82 : none loc(#loc47)
  } loc(#loc46)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z10matmul_cpuPKjS0_Pjjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg3: i32 loc(fused<#di_subprogram5>[#loc22]), %arg4: i32 loc(fused<#di_subprogram5>[#loc22]), %arg5: i32 loc(fused<#di_subprogram5>[#loc22])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc55)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg5, %c0_i32 : i32 loc(#loc2)
      %2 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %3 = arith.extui %arg5 : i32 to i64 loc(#loc2)
      %4 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %5 = scf.while (%arg6 = %c0_i32) : (i32) -> i32 {
        scf.if %1 {
        } else {
          %8 = arith.muli %arg6, %arg4 : i32 loc(#loc2)
          %9 = arith.muli %arg6, %arg5 : i32 loc(#loc2)
          %10 = scf.while (%arg7 = %c0_i64) : (i64) -> i64 {
            %11 = scf.if %2 -> (i32) {
              scf.yield %c0_i32 : i32 loc(#loc64)
            } else {
              %18 = arith.trunci %arg7 : i64 to i32 loc(#loc2)
              %19:2 = scf.while (%arg8 = %c0_i64, %arg9 = %c0_i32) : (i64, i32) -> (i64, i32) {
                %20 = arith.trunci %arg8 : i64 to i32 loc(#loc67)
                %21 = arith.addi %8, %20 : i32 loc(#loc67)
                %22 = arith.extui %21 : i32 to i64 loc(#loc67)
                %23 = arith.index_cast %22 : i64 to index loc(#loc67)
                %24 = memref.load %arg0[%23] : memref<?xi32, strided<[1], offset: ?>> loc(#loc67)
                %25 = arith.muli %arg5, %20 : i32 loc(#loc67)
                %26 = arith.addi %25, %18 : i32 loc(#loc67)
                %27 = arith.extui %26 : i32 to i64 loc(#loc67)
                %28 = arith.index_cast %27 : i64 to index loc(#loc67)
                %29 = memref.load %arg1[%28] : memref<?xi32, strided<[1], offset: ?>> loc(#loc67)
                %30 = arith.muli %29, %24 : i32 loc(#loc67)
                %31 = arith.addi %30, %arg9 : i32 loc(#loc67)
                %32 = arith.addi %arg8, %c1_i64 : i64 loc(#loc65)
                %33 = arith.cmpi ne, %32, %4 : i64 loc(#loc68)
                scf.condition(%33) %32, %31 : i64, i32 loc(#loc64)
              } do {
              ^bb0(%arg8: i64 loc(fused<#di_lexical_block41>[#loc25]), %arg9: i32 loc(fused<#di_lexical_block41>[#loc25])):
                scf.yield %arg8, %arg9 : i64, i32 loc(#loc64)
              } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc64)
              scf.yield %19#1 : i32 loc(#loc64)
            } loc(#loc64)
            %12 = arith.trunci %arg7 : i64 to i32 loc(#loc61)
            %13 = arith.addi %9, %12 : i32 loc(#loc61)
            %14 = arith.extui %13 : i32 to i64 loc(#loc61)
            %15 = arith.index_cast %14 : i64 to index loc(#loc61)
            memref.store %11, %arg2[%15] : memref<?xi32, strided<[1], offset: ?>> loc(#loc61)
            %16 = arith.addi %arg7, %c1_i64 : i64 loc(#loc59)
            %17 = arith.cmpi ne, %16, %3 : i64 loc(#loc62)
            scf.condition(%17) %16 : i64 loc(#loc58)
          } do {
          ^bb0(%arg7: i64 loc(fused<#di_lexical_block35>[#loc24])):
            scf.yield %arg7 : i64 loc(#loc58)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc58)
        } loc(#loc58)
        %6 = arith.addi %arg6, %c1_i32 : i32 loc(#loc55)
        %7 = arith.cmpi ne, %6, %arg3 : i32 loc(#loc56)
        scf.condition(%7) %6 : i32 loc(#loc53)
      } do {
      ^bb0(%arg6: i32 loc(fused<#di_lexical_block29>[#loc23])):
        scf.yield %arg6 : i32 loc(#loc53)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc53)
    } loc(#loc53)
    return loc(#loc49)
  } loc(#loc48)
} loc(#loc)
#loc = loc("tests/app/matmul/main.cpp":0:0)
#loc1 = loc("tests/app/matmul/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/matmul/main.cpp":20:0)
#loc6 = loc("tests/app/matmul/main.cpp":23:0)
#loc7 = loc("tests/app/matmul/main.cpp":27:0)
#loc8 = loc("tests/app/matmul/main.cpp":30:0)
#loc10 = loc("tests/app/matmul/main.cpp":34:0)
#loc11 = loc("tests/app/matmul/main.cpp":35:0)
#loc12 = loc("tests/app/matmul/main.cpp":36:0)
#loc13 = loc("tests/app/matmul/main.cpp":40:0)
#loc14 = loc("tests/app/matmul/main.cpp":42:0)
#loc16 = loc("tests/app/matmul/matmul.cpp":42:0)
#loc17 = loc("tests/app/matmul/matmul.cpp":44:0)
#loc18 = loc("tests/app/matmul/matmul.cpp":47:0)
#loc19 = loc("tests/app/matmul/matmul.cpp":50:0)
#loc20 = loc("tests/app/matmul/matmul.cpp":48:0)
#loc21 = loc("tests/app/matmul/matmul.cpp":53:0)
#loc26 = loc("tests/app/matmul/matmul.cpp":25:0)
#loc27 = loc("tests/app/matmul/matmul.cpp":27:0)
#loc28 = loc("tests/app/matmul/matmul.cpp":30:0)
#loc29 = loc(fused<#di_subprogram3>[#loc1])
#loc30 = loc(fused<#di_subprogram3>[#loc7])
#loc31 = loc(fused<#di_subprogram3>[#loc8])
#loc32 = loc(fused<#di_subprogram3>[#loc13])
#loc33 = loc(fused<#di_subprogram3>[#loc14])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 19>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 22>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 33>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 19>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 22>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 33>
#loc37 = loc(fused<#di_lexical_block18>[#loc3])
#loc38 = loc(fused<#di_lexical_block19>[#loc5])
#loc39 = loc(fused<#di_lexical_block20>[#loc9])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 34>
#loc40 = loc(fused<#di_lexical_block23>[#loc4])
#loc41 = loc(fused[#loc34, #loc37])
#loc42 = loc(fused<#di_lexical_block24>[#loc6])
#loc43 = loc(fused[#loc35, #loc38])
#loc44 = loc(fused[#loc36, #loc39])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file, line = 34>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 42>
#loc45 = loc(fused<#di_lexical_block26>[#loc10])
#loc47 = loc(fused<#di_subprogram4>[#loc21])
#loc49 = loc(fused<#di_subprogram5>[#loc28])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 42>
#loc50 = loc(fused<#di_lexical_block27>[#loc11])
#loc51 = loc(fused<#di_lexical_block27>[#loc12])
#loc52 = loc(fused<#di_lexical_block28>[#loc16])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 42>
#loc54 = loc(fused<#di_lexical_block30>[#loc16])
#loc55 = loc(fused<#di_lexical_block31>[#loc23])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file1, line = 44>
#loc56 = loc(fused[#loc53, #loc55])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file1, line = 44>
#loc57 = loc(fused<#di_lexical_block34>[#loc17])
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file1, line = 44>
#loc59 = loc(fused<#di_lexical_block37>[#loc24])
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file1, line = 47>
#loc60 = loc(fused<#di_lexical_block38>[#loc19])
#loc61 = loc(fused<#di_lexical_block39>[#loc27])
#loc62 = loc(fused[#loc58, #loc59])
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file1, line = 47>
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file1, line = 24>
#loc63 = loc(fused<#di_lexical_block40>[#loc18])
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_lexical_block42, file = #di_file1, line = 47>
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_lexical_block43, file = #di_file1, line = 24>
#loc65 = loc(fused<#di_lexical_block43>[#loc25])
#loc66 = loc(fused<#di_lexical_block44>[#loc20])
#loc67 = loc(fused<#di_lexical_block45>[#loc26])
#loc68 = loc(fused[#loc64, #loc65])
