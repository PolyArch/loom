#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/matvec/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/matvec/matvec.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/matvec/main.cpp":18:0)
#loc5 = loc("tests/app/matvec/main.cpp":21:0)
#loc9 = loc("tests/app/matvec/main.cpp":32:0)
#loc15 = loc("tests/app/matvec/matvec.cpp":33:0)
#loc21 = loc("tests/app/matvec/matvec.cpp":16:0)
#loc22 = loc("tests/app/matvec/matvec.cpp":21:0)
#loc23 = loc("tests/app/matvec/matvec.cpp":23:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 21>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 32>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 38>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 21>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 38>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 21>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 49152, elements = #llvm.di_subrange<count = 1536 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1536, elements = #llvm.di_subrange<count = 48 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1024, elements = #llvm.di_subrange<count = 32 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 38>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 21>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 32, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 21, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 40>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 23>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "x", file = #di_file, line = 11, type = #di_composite_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_y", file = #di_file, line = 14, type = #di_composite_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_y", file = #di_file, line = 15, type = #di_composite_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file1, line = 36, arg = 4, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 37, arg = 5, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "sum", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 19, arg = 4, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 20, arg = 5, type = #di_derived_type2>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "sum", file = #di_file1, line = 22, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "y", file = #di_file1, line = 35, arg = 3, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file1, line = 40, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "y", file = #di_file1, line = 18, arg = 3, type = #di_derived_type5>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "j", file = #di_file1, line = 23, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 18>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 21>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 32>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file1, line = 33, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "x", file = #di_file1, line = 34, arg = 2, type = #di_derived_type6>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 16, arg = 1, type = #di_derived_type6>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram2, name = "x", file = #di_file1, line = 17, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "matvec_dsa", linkageName = "_Z10matvec_dsaPKjS0_Pjjj", file = #di_file1, line = 33, scopeLine = 37, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable17, #di_local_variable11, #di_local_variable12, #di_local_variable3, #di_local_variable13, #di_local_variable18>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "matvec_cpu", linkageName = "_Z10matvec_cpuPKjS0_Pjjj", file = #di_file1, line = 16, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable23, #di_local_variable24, #di_local_variable19, #di_local_variable14, #di_local_variable15, #di_local_variable4, #di_local_variable16, #di_local_variable20>
#loc32 = loc(fused<#di_lexical_block11>[#loc3])
#loc33 = loc(fused<#di_lexical_block12>[#loc5])
#loc34 = loc(fused<#di_lexical_block13>[#loc9])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 21>
#loc38 = loc(fused<#di_subprogram4>[#loc15])
#loc40 = loc(fused<#di_subprogram5>[#loc21])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 21>
#loc48 = loc(fused<#di_lexical_block21>[#loc22])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 21>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 23>
#loc58 = loc(fused<#di_lexical_block29>[#loc23])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[109, 97, 116, 118, 101, 99, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[109, 97, 116, 118, 101, 99, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 52, 44, 98, 108, 111, 99, 107, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 109, 97, 116, 118, 101, 99, 47, 109, 97, 116, 118, 101, 99, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc27)
    %false = arith.constant false loc(#loc27)
    %0 = seq.const_clock  low loc(#loc27)
    %c2_i32 = arith.constant 2 : i32 loc(#loc27)
    %1 = ub.poison : i64 loc(#loc27)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1536_i64 = arith.constant 1536 : i64 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c48_i64 = arith.constant 48 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c48_i32 = arith.constant 48 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1536xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<48xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<32xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<32xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc42)
      %12 = arith.remui %11, %c10_i32 : i32 loc(#loc42)
      %13 = arith.addi %12, %c1_i32 : i32 loc(#loc42)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      memref.store %13, %alloca[%14] : memref<1536xi32> loc(#loc42)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
      %16 = arith.cmpi ne, %15, %c1536_i64 : i64 loc(#loc43)
      scf.condition(%16) %15 : i64 loc(#loc32)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc32)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc32)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc44)
      %12 = arith.remui %11, %c7_i32 : i32 loc(#loc44)
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc44)
      memref.store %12, %alloca_0[%13] : memref<48xi32> loc(#loc44)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc36)
      %15 = arith.cmpi ne, %14, %c48_i64 : i64 loc(#loc45)
      scf.condition(%15) %14 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc5])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<1536xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %cast_3 = memref.cast %alloca_0 : memref<48xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %cast_4 = memref.cast %alloca_1 : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    call @_Z10matvec_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c32_i32, %c48_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc28)
    %cast_5 = memref.cast %alloca_2 : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc29)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c48_i32, %true : i32 loc(#loc29)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc29)
    %6 = handshake.esi_instance @_Z10matvec_dsaPKjS0_Pjjj_esi "_Z10matvec_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc29)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc29)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      %12 = memref.load %alloca_1[%11] : memref<32xi32> loc(#loc49)
      %13 = memref.load %alloca_2[%11] : memref<32xi32> loc(#loc49)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc49)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc37)
        %18 = arith.cmpi eq, %17, %c32_i64 : i64 loc(#loc37)
        %19 = arith.extui %18 : i1 to i32 loc(#loc34)
        %20 = arith.cmpi ne, %17, %c32_i64 : i64 loc(#loc46)
        %21 = arith.extui %20 : i1 to i32 loc(#loc34)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc49)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc49)
      } loc(#loc49)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc34)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc9]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc9]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc9])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc34)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc52)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc52)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc52)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc52)
      scf.yield %c1_i32 : i32 loc(#loc53)
    } loc(#loc34)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc30)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc30)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc30)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc30)
    } loc(#loc2)
    return %10 : i32 loc(#loc31)
  } loc(#loc27)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10matvec_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: i1 loc(fused<#di_subprogram4>[#loc15]), ...) -> i1 attributes {argNames = ["A", "x", "y", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc38)
    %1 = handshake.join %0 : none loc(#loc38)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc50)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc47)
    %5 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %7 = arith.index_cast %3 : i64 to index loc(#loc47)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc47)
    %10 = arith.index_cast %afterValue : index to i64 loc(#loc47)
    %11 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc57)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %9 : none loc(#loc57)
    %12 = arith.trunci %10 : i64 to i32 loc(#loc2)
    %13 = arith.muli %arg4, %12 : i32 loc(#loc2)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc57)
    %15 = arith.index_cast %arg4 : i32 to index loc(#loc57)
    %index_2, %willContinue_3 = dataflow.stream %7, %14, %15 {step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc57)
    %16 = dataflow.carry %willContinue_3, %2, %24 : i1, i32, i32 -> i32 loc(#loc57)
    %afterValue_6, %afterCond_7 = dataflow.gate %16, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc57)
    handshake.sink %afterCond_7 : i1 loc(#loc57)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %16 : i32 loc(#loc57)
    %17 = arith.index_cast %afterValue_4 : index to i64 loc(#loc57)
    %18 = arith.trunci %17 : i64 to i32 loc(#loc60)
    %19 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc60)
    %20 = arith.addi %19, %18 : i32 loc(#loc60)
    %21 = arith.extui %20 : i32 to i64 loc(#loc60)
    %22 = arith.index_cast %21 : i64 to index loc(#loc60)
    %dataResult, %addressResults = handshake.load [%22] %30#0, %33 : index, i32 loc(#loc60)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %29#0, %45 : index, i32 loc(#loc60)
    %23 = arith.muli %dataResult_10, %dataResult : i32 loc(#loc60)
    %24 = arith.addi %23, %afterValue_6 : i32 loc(#loc60)
    %25 = handshake.constant %9 {value = 0 : index} : index loc(#loc57)
    %26 = handshake.constant %9 {value = 1 : index} : index loc(#loc57)
    %27 = arith.select %11, %26, %25 : index loc(#loc57)
    %28 = handshake.mux %27 [%falseResult_9, %2] : index, i32 loc(#loc57)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %28, %42 : index, i32 loc(#loc54)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc47)
    %trueResult_13, %falseResult_14 = handshake.cond_br %11, %32 : none loc(#loc57)
    %33 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc57)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %30#1 : none loc(#loc57)
    %34 = handshake.constant %32 {value = 0 : index} : index loc(#loc57)
    %35 = handshake.constant %32 {value = 1 : index} : index loc(#loc57)
    %36 = arith.select %11, %35, %34 : index loc(#loc57)
    %37 = handshake.mux %36 [%falseResult_16, %trueResult_13] : index, none loc(#loc57)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %37 : none loc(#loc47)
    %38 = handshake.constant %1 {value = 0 : index} : index loc(#loc47)
    %39 = handshake.constant %1 {value = 1 : index} : index loc(#loc47)
    %40 = arith.select %4, %39, %38 : index loc(#loc47)
    %41 = handshake.mux %40 [%falseResult_18, %trueResult] : index, none loc(#loc47)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc47)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %31 : none loc(#loc47)
    %43 = handshake.mux %40 [%falseResult_20, %trueResult] : index, none loc(#loc47)
    %44 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc47)
    %trueResult_21, %falseResult_22 = handshake.cond_br %11, %44 : none loc(#loc57)
    %45 = dataflow.carry %willContinue_3, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc57)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_3, %29#1 : none loc(#loc57)
    %46 = handshake.constant %44 {value = 0 : index} : index loc(#loc57)
    %47 = handshake.constant %44 {value = 1 : index} : index loc(#loc57)
    %48 = arith.select %11, %47, %46 : index loc(#loc57)
    %49 = handshake.mux %48 [%falseResult_24, %trueResult_21] : index, none loc(#loc57)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %49 : none loc(#loc47)
    %50 = handshake.mux %40 [%falseResult_26, %trueResult] : index, none loc(#loc47)
    %51 = handshake.join %41, %43, %50 : none, none, none loc(#loc38)
    %52 = handshake.constant %51 {value = true} : i1 loc(#loc38)
    handshake.return %52 : i1 loc(#loc38)
  } loc(#loc38)
  handshake.func @_Z10matvec_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: none loc(fused<#di_subprogram4>[#loc15]), ...) -> none attributes {argNames = ["A", "x", "y", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc38)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc50)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc47)
    %4 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc2)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %6 = arith.index_cast %2 : i64 to index loc(#loc47)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc47)
    %9 = arith.index_cast %afterValue : index to i64 loc(#loc47)
    %10 = dataflow.invariant %afterCond, %4 : i1, i1 -> i1 loc(#loc57)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %8 : none loc(#loc57)
    %11 = arith.trunci %9 : i64 to i32 loc(#loc2)
    %12 = arith.muli %arg4, %11 : i32 loc(#loc2)
    %13 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc57)
    %14 = arith.index_cast %arg4 : i32 to index loc(#loc57)
    %index_2, %willContinue_3 = dataflow.stream %6, %13, %14 {step_op = "+=", stop_cond = "!="} loc(#loc57)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc57)
    %15 = dataflow.carry %willContinue_3, %1, %23 : i1, i32, i32 -> i32 loc(#loc57)
    %afterValue_6, %afterCond_7 = dataflow.gate %15, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc57)
    handshake.sink %afterCond_7 : i1 loc(#loc57)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %15 : i32 loc(#loc57)
    %16 = arith.index_cast %afterValue_4 : index to i64 loc(#loc57)
    %17 = arith.trunci %16 : i64 to i32 loc(#loc60)
    %18 = dataflow.invariant %afterCond_5, %12 : i1, i32 -> i32 loc(#loc60)
    %19 = arith.addi %18, %17 : i32 loc(#loc60)
    %20 = arith.extui %19 : i32 to i64 loc(#loc60)
    %21 = arith.index_cast %20 : i64 to index loc(#loc60)
    %dataResult, %addressResults = handshake.load [%21] %29#0, %32 : index, i32 loc(#loc60)
    %dataResult_10, %addressResults_11 = handshake.load [%afterValue_4] %28#0, %44 : index, i32 loc(#loc60)
    %22 = arith.muli %dataResult_10, %dataResult : i32 loc(#loc60)
    %23 = arith.addi %22, %afterValue_6 : i32 loc(#loc60)
    %24 = handshake.constant %8 {value = 0 : index} : index loc(#loc57)
    %25 = handshake.constant %8 {value = 1 : index} : index loc(#loc57)
    %26 = arith.select %10, %25, %24 : index loc(#loc57)
    %27 = handshake.mux %26 [%falseResult_9, %1] : index, i32 loc(#loc57)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %27, %41 : index, i32 loc(#loc54)
    %28:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_11) {id = 0 : i32} : (index) -> (i32, none) loc(#loc38)
    %29:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc38)
    %30 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc38)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc47)
    %trueResult_13, %falseResult_14 = handshake.cond_br %10, %31 : none loc(#loc57)
    %32 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc57)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %29#1 : none loc(#loc57)
    %33 = handshake.constant %31 {value = 0 : index} : index loc(#loc57)
    %34 = handshake.constant %31 {value = 1 : index} : index loc(#loc57)
    %35 = arith.select %10, %34, %33 : index loc(#loc57)
    %36 = handshake.mux %35 [%falseResult_16, %trueResult_13] : index, none loc(#loc57)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %36 : none loc(#loc47)
    %37 = handshake.constant %0 {value = 0 : index} : index loc(#loc47)
    %38 = handshake.constant %0 {value = 1 : index} : index loc(#loc47)
    %39 = arith.select %3, %38, %37 : index loc(#loc47)
    %40 = handshake.mux %39 [%falseResult_18, %trueResult] : index, none loc(#loc47)
    %41 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc47)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %30 : none loc(#loc47)
    %42 = handshake.mux %39 [%falseResult_20, %trueResult] : index, none loc(#loc47)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_25 : i1, none, none -> none loc(#loc47)
    %trueResult_21, %falseResult_22 = handshake.cond_br %10, %43 : none loc(#loc57)
    %44 = dataflow.carry %willContinue_3, %falseResult_22, %trueResult_23 : i1, none, none -> none loc(#loc57)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_3, %28#1 : none loc(#loc57)
    %45 = handshake.constant %43 {value = 0 : index} : index loc(#loc57)
    %46 = handshake.constant %43 {value = 1 : index} : index loc(#loc57)
    %47 = arith.select %10, %46, %45 : index loc(#loc57)
    %48 = handshake.mux %47 [%falseResult_24, %trueResult_21] : index, none loc(#loc57)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %48 : none loc(#loc47)
    %49 = handshake.mux %39 [%falseResult_26, %trueResult] : index, none loc(#loc47)
    %50 = handshake.join %40, %42, %49 : none, none, none loc(#loc38)
    handshake.return %50 : none loc(#loc39)
  } loc(#loc38)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z10matvec_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc21]), %arg3: i32 loc(fused<#di_subprogram5>[#loc21]), %arg4: i32 loc(fused<#di_subprogram5>[#loc21])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc51)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg3 : i32 to i64 loc(#loc51)
      %3 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %4 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %5 = scf.if %1 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc58)
        } else {
          %9 = arith.trunci %arg5 : i64 to i32 loc(#loc2)
          %10 = arith.muli %arg4, %9 : i32 loc(#loc2)
          %11:2 = scf.while (%arg6 = %c0_i64, %arg7 = %c0_i32) : (i64, i32) -> (i64, i32) {
            %12 = arith.trunci %arg6 : i64 to i32 loc(#loc61)
            %13 = arith.addi %10, %12 : i32 loc(#loc61)
            %14 = arith.extui %13 : i32 to i64 loc(#loc61)
            %15 = arith.index_cast %14 : i64 to index loc(#loc61)
            %16 = memref.load %arg0[%15] : memref<?xi32, strided<[1], offset: ?>> loc(#loc61)
            %17 = arith.index_cast %arg6 : i64 to index loc(#loc61)
            %18 = memref.load %arg1[%17] : memref<?xi32, strided<[1], offset: ?>> loc(#loc61)
            %19 = arith.muli %18, %16 : i32 loc(#loc61)
            %20 = arith.addi %19, %arg7 : i32 loc(#loc61)
            %21 = arith.addi %arg6, %c1_i64 : i64 loc(#loc59)
            %22 = arith.cmpi ne, %21, %3 : i64 loc(#loc62)
            scf.condition(%22) %21, %20 : i64, i32 loc(#loc58)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block29>[#loc23]), %arg7: i32 loc(fused<#di_lexical_block29>[#loc23])):
            scf.yield %arg6, %arg7 : i64, i32 loc(#loc58)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc58)
          scf.yield %11#1 : i32 loc(#loc58)
        } loc(#loc58)
        %6 = arith.index_cast %arg5 : i64 to index loc(#loc55)
        memref.store %5, %arg2[%6] : memref<?xi32, strided<[1], offset: ?>> loc(#loc55)
        %7 = arith.addi %arg5, %c1_i64 : i64 loc(#loc51)
        %8 = arith.cmpi ne, %7, %2 : i64 loc(#loc56)
        scf.condition(%8) %7 : i64 loc(#loc48)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block21>[#loc22])):
        scf.yield %arg5 : i64 loc(#loc48)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc48)
    } loc(#loc48)
    return loc(#loc41)
  } loc(#loc40)
} loc(#loc)
#loc = loc("tests/app/matvec/main.cpp":0:0)
#loc1 = loc("tests/app/matvec/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/matvec/main.cpp":19:0)
#loc6 = loc("tests/app/matvec/main.cpp":22:0)
#loc7 = loc("tests/app/matvec/main.cpp":26:0)
#loc8 = loc("tests/app/matvec/main.cpp":29:0)
#loc10 = loc("tests/app/matvec/main.cpp":33:0)
#loc11 = loc("tests/app/matvec/main.cpp":34:0)
#loc12 = loc("tests/app/matvec/main.cpp":35:0)
#loc13 = loc("tests/app/matvec/main.cpp":39:0)
#loc14 = loc("tests/app/matvec/main.cpp":41:0)
#loc16 = loc("tests/app/matvec/matvec.cpp":38:0)
#loc17 = loc("tests/app/matvec/matvec.cpp":40:0)
#loc18 = loc("tests/app/matvec/matvec.cpp":41:0)
#loc19 = loc("tests/app/matvec/matvec.cpp":43:0)
#loc20 = loc("tests/app/matvec/matvec.cpp":45:0)
#loc24 = loc("tests/app/matvec/matvec.cpp":24:0)
#loc25 = loc("tests/app/matvec/matvec.cpp":26:0)
#loc26 = loc("tests/app/matvec/matvec.cpp":28:0)
#loc27 = loc(fused<#di_subprogram3>[#loc1])
#loc28 = loc(fused<#di_subprogram3>[#loc7])
#loc29 = loc(fused<#di_subprogram3>[#loc8])
#loc30 = loc(fused<#di_subprogram3>[#loc13])
#loc31 = loc(fused<#di_subprogram3>[#loc14])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 18>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 21>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 32>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 18>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 21>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 32>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 38>
#loc35 = loc(fused<#di_lexical_block14>[#loc3])
#loc36 = loc(fused<#di_lexical_block15>[#loc5])
#loc37 = loc(fused<#di_lexical_block16>[#loc9])
#loc39 = loc(fused<#di_subprogram4>[#loc20])
#loc41 = loc(fused<#di_subprogram5>[#loc26])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 33>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 38>
#loc42 = loc(fused<#di_lexical_block17>[#loc4])
#loc43 = loc(fused[#loc32, #loc35])
#loc44 = loc(fused<#di_lexical_block18>[#loc6])
#loc45 = loc(fused[#loc33, #loc36])
#loc46 = loc(fused[#loc34, #loc37])
#loc47 = loc(fused<#di_lexical_block20>[#loc16])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 33>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 38>
#loc49 = loc(fused<#di_lexical_block22>[#loc10])
#loc50 = loc(fused<#di_lexical_block23>[#loc16])
#loc51 = loc(fused<#di_lexical_block24>[#loc22])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 40>
#loc52 = loc(fused<#di_lexical_block25>[#loc11])
#loc53 = loc(fused<#di_lexical_block25>[#loc12])
#loc54 = loc(fused<#di_lexical_block26>[#loc19])
#loc55 = loc(fused<#di_lexical_block27>[#loc25])
#loc56 = loc(fused[#loc48, #loc51])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 40>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file1, line = 23>
#loc57 = loc(fused<#di_lexical_block28>[#loc17])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 40>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file1, line = 23>
#loc59 = loc(fused<#di_lexical_block31>[#loc23])
#loc60 = loc(fused<#di_lexical_block32>[#loc18])
#loc61 = loc(fused<#di_lexical_block33>[#loc24])
#loc62 = loc(fused[#loc58, #loc59])
