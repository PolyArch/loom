#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/outer/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/outer/outer.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/outer/main.cpp":18:0)
#loc5 = loc("tests/app/outer/main.cpp":21:0)
#loc9 = loc("tests/app/outer/main.cpp":32:0)
#loc15 = loc("tests/app/outer/outer.cpp":31:0)
#loc20 = loc("tests/app/outer/outer.cpp":16:0)
#loc21 = loc("tests/app/outer/outer.cpp":21:0)
#loc22 = loc("tests/app/outer/outer.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 21>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 32>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 37>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 21>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 37>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 21>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1024, elements = #llvm.di_subrange<count = 32 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 1536, elements = #llvm.di_subrange<count = 48 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 49152, elements = #llvm.di_subrange<count = 1536 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 37>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 21>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 32, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 37, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 21, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 38>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 22>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "b", file = #di_file, line = 11, type = #di_composite_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_C", file = #di_file, line = 14, type = #di_composite_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_C", file = #di_file, line = 15, type = #di_composite_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file1, line = 34, arg = 4, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 35, arg = 5, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 19, arg = 4, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 20, arg = 5, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "C", file = #di_file1, line = 33, arg = 3, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "C", file = #di_file1, line = 18, arg = 3, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "j", file = #di_file1, line = 22, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 18>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 21>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 32>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "a", file = #di_file1, line = 31, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "b", file = #di_file1, line = 32, arg = 2, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "a", file = #di_file1, line = 16, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "b", file = #di_file1, line = 17, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "outer_dsa", linkageName = "_Z9outer_dsaPKjS0_Pjjj", file = #di_file1, line = 31, scopeLine = 35, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable15, #di_local_variable11, #di_local_variable12, #di_local_variable3, #di_local_variable16>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "outer_cpu", linkageName = "_Z9outer_cpuPKjS0_Pjjj", file = #di_file1, line = 16, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable17, #di_local_variable13, #di_local_variable14, #di_local_variable4, #di_local_variable18>
#loc30 = loc(fused<#di_lexical_block11>[#loc3])
#loc31 = loc(fused<#di_lexical_block12>[#loc5])
#loc32 = loc(fused<#di_lexical_block13>[#loc9])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 21>
#loc36 = loc(fused<#di_subprogram4>[#loc15])
#loc38 = loc(fused<#di_subprogram5>[#loc20])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 21>
#loc46 = loc(fused<#di_lexical_block21>[#loc21])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 21>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 22>
#loc54 = loc(fused<#di_lexical_block29>[#loc22])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<14xi8> = dense<[111, 117, 116, 101, 114, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<14xi8> = dense<[111, 117, 116, 101, 114, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<26xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 111, 117, 116, 101, 114, 47, 111, 117, 116, 101, 114, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc25)
    %false = arith.constant false loc(#loc25)
    %0 = seq.const_clock  low loc(#loc25)
    %c2_i32 = arith.constant 2 : i32 loc(#loc25)
    %1 = ub.poison : i64 loc(#loc25)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c1536_i64 = arith.constant 1536 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %c48_i64 = arith.constant 48 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c48_i32 = arith.constant 48 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<14xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<14xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<32xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<48xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1536xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1536xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.addi %arg0, %c1_i64 : i64 loc(#loc40)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc40)
      %13 = arith.trunci %11 : i64 to i32 loc(#loc40)
      memref.store %13, %alloca[%12] : memref<32xi32> loc(#loc40)
      %14 = arith.cmpi ne, %11, %c32_i64 : i64 loc(#loc41)
      scf.condition(%14) %11 : i64 loc(#loc30)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc30)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc30)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      %12 = arith.trunci %arg0 : i64 to i32 loc(#loc42)
      %13 = arith.shli %12, %c1_i32 : i32 loc(#loc42)
      %14 = arith.ori %13, %c1_i32 : i32 loc(#loc42)
      memref.store %14, %alloca_0[%11] : memref<48xi32> loc(#loc42)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc34)
      %16 = arith.cmpi ne, %15, %c48_i64 : i64 loc(#loc43)
      scf.condition(%16) %15 : i64 loc(#loc31)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc5])):
      scf.yield %arg0 : i64 loc(#loc31)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc31)
    %cast = memref.cast %alloca : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    %cast_3 = memref.cast %alloca_0 : memref<48xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    %cast_4 = memref.cast %alloca_1 : memref<1536xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    call @_Z9outer_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c32_i32, %c48_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc26)
    %cast_5 = memref.cast %alloca_2 : memref<1536xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc27)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c48_i32, %true : i32 loc(#loc27)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc27)
    %6 = handshake.esi_instance @_Z9outer_dsaPKjS0_Pjjj_esi "_Z9outer_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc27)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc27)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc47)
      %12 = memref.load %alloca_1[%11] : memref<1536xi32> loc(#loc47)
      %13 = memref.load %alloca_2[%11] : memref<1536xi32> loc(#loc47)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc47)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
        %18 = arith.cmpi eq, %17, %c1536_i64 : i64 loc(#loc35)
        %19 = arith.extui %18 : i1 to i32 loc(#loc32)
        %20 = arith.cmpi ne, %17, %c1536_i64 : i64 loc(#loc44)
        %21 = arith.extui %20 : i1 to i32 loc(#loc32)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc47)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc47)
      } loc(#loc47)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc32)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc32)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc9]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc9]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc9])):
      scf.yield %arg0 : i64 loc(#loc32)
    } loc(#loc32)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc32)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc32)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<14xi8> -> index loc(#loc50)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc50)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc32)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<14xi8> -> index loc(#loc28)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc28)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc28)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc28)
    } loc(#loc2)
    return %10 : i32 loc(#loc29)
  } loc(#loc25)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z9outer_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: i1 loc(fused<#di_subprogram4>[#loc15]), ...) -> i1 attributes {argNames = ["a", "b", "C", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc36)
    %1 = handshake.join %0 : none loc(#loc36)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc45)
    %5 = arith.cmpi eq, %arg4, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %7 = arith.index_cast %3 : i64 to index loc(#loc45)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc45)
    %10 = arith.index_cast %afterValue : index to i64 loc(#loc45)
    %11 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc53)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %9 : none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %25#0, %falseResult_10 : index, i32 loc(#loc2)
    %12 = arith.trunci %10 : i64 to i32 loc(#loc2)
    %13 = arith.muli %arg4, %12 : i32 loc(#loc2)
    %14 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc53)
    %15 = arith.index_cast %arg4 : i32 to index loc(#loc53)
    %index_2, %willContinue_3 = dataflow.stream %7, %14, %15 {step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc53)
    %16 = arith.index_cast %afterValue_4 : index to i64 loc(#loc53)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue_4] %24#0, %44 : index, i32 loc(#loc56)
    %17 = dataflow.invariant %afterCond_5, %dataResult : i1, i32 -> i32 loc(#loc56)
    %18 = arith.muli %dataResult_6, %17 : i32 loc(#loc56)
    %19 = arith.trunci %16 : i64 to i32 loc(#loc56)
    %20 = dataflow.invariant %afterCond_5, %13 : i1, i32 -> i32 loc(#loc56)
    %21 = arith.addi %20, %19 : i32 loc(#loc56)
    %22 = arith.extui %21 : i32 to i64 loc(#loc56)
    %23 = arith.index_cast %22 : i64 to index loc(#loc56)
    %dataResult_8, %addressResult = handshake.store [%23] %18, %37 : index, i32 loc(#loc56)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_7) {id = 0 : i32} : (index) -> (i32, none) loc(#loc36)
    %25:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc36)
    %26 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc36)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc45)
    %trueResult_9, %falseResult_10 = handshake.cond_br %11, %27 : none loc(#loc53)
    %28 = handshake.constant %27 {value = 0 : index} : index loc(#loc53)
    %29 = handshake.constant %27 {value = 1 : index} : index loc(#loc53)
    %30 = arith.select %11, %29, %28 : index loc(#loc53)
    %31 = handshake.mux %30 [%25#1, %trueResult_9] : index, none loc(#loc53)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %31 : none loc(#loc45)
    %32 = handshake.constant %1 {value = 0 : index} : index loc(#loc45)
    %33 = handshake.constant %1 {value = 1 : index} : index loc(#loc45)
    %34 = arith.select %4, %33, %32 : index loc(#loc45)
    %35 = handshake.mux %34 [%falseResult_12, %trueResult] : index, none loc(#loc45)
    %36 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc45)
    %trueResult_13, %falseResult_14 = handshake.cond_br %11, %36 : none loc(#loc53)
    %37 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %26 : none loc(#loc53)
    %38 = handshake.constant %36 {value = 0 : index} : index loc(#loc53)
    %39 = handshake.constant %36 {value = 1 : index} : index loc(#loc53)
    %40 = arith.select %11, %39, %38 : index loc(#loc53)
    %41 = handshake.mux %40 [%falseResult_16, %trueResult_13] : index, none loc(#loc53)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %41 : none loc(#loc45)
    %42 = handshake.mux %34 [%falseResult_18, %trueResult] : index, none loc(#loc45)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_23 : i1, none, none -> none loc(#loc45)
    %trueResult_19, %falseResult_20 = handshake.cond_br %11, %43 : none loc(#loc53)
    %44 = dataflow.carry %willContinue_3, %falseResult_20, %trueResult_21 : i1, none, none -> none loc(#loc53)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_3, %24#1 : none loc(#loc53)
    %45 = handshake.constant %43 {value = 0 : index} : index loc(#loc53)
    %46 = handshake.constant %43 {value = 1 : index} : index loc(#loc53)
    %47 = arith.select %11, %46, %45 : index loc(#loc53)
    %48 = handshake.mux %47 [%falseResult_22, %trueResult_19] : index, none loc(#loc53)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue, %48 : none loc(#loc45)
    %49 = handshake.mux %34 [%falseResult_24, %trueResult] : index, none loc(#loc45)
    %50 = handshake.join %35, %42, %49 : none, none, none loc(#loc36)
    %51 = handshake.constant %50 {value = true} : i1 loc(#loc36)
    handshake.return %51 : i1 loc(#loc36)
  } loc(#loc36)
  handshake.func @_Z9outer_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i32 loc(fused<#di_subprogram4>[#loc15]), %arg5: none loc(fused<#di_subprogram4>[#loc15]), ...) -> none attributes {argNames = ["a", "b", "C", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc36)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc45)
    %4 = arith.cmpi eq, %arg4, %1 : i32 loc(#loc2)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %6 = arith.index_cast %2 : i64 to index loc(#loc45)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc45)
    %9 = arith.index_cast %afterValue : index to i64 loc(#loc45)
    %10 = dataflow.invariant %afterCond, %4 : i1, i1 -> i1 loc(#loc53)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %8 : none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %24#0, %falseResult_10 : index, i32 loc(#loc2)
    %11 = arith.trunci %9 : i64 to i32 loc(#loc2)
    %12 = arith.muli %arg4, %11 : i32 loc(#loc2)
    %13 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc53)
    %14 = arith.index_cast %arg4 : i32 to index loc(#loc53)
    %index_2, %willContinue_3 = dataflow.stream %6, %13, %14 {step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc53)
    %15 = arith.index_cast %afterValue_4 : index to i64 loc(#loc53)
    %dataResult_6, %addressResults_7 = handshake.load [%afterValue_4] %23#0, %43 : index, i32 loc(#loc56)
    %16 = dataflow.invariant %afterCond_5, %dataResult : i1, i32 -> i32 loc(#loc56)
    %17 = arith.muli %dataResult_6, %16 : i32 loc(#loc56)
    %18 = arith.trunci %15 : i64 to i32 loc(#loc56)
    %19 = dataflow.invariant %afterCond_5, %12 : i1, i32 -> i32 loc(#loc56)
    %20 = arith.addi %19, %18 : i32 loc(#loc56)
    %21 = arith.extui %20 : i32 to i64 loc(#loc56)
    %22 = arith.index_cast %21 : i64 to index loc(#loc56)
    %dataResult_8, %addressResult = handshake.store [%22] %17, %36 : index, i32 loc(#loc56)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_7) {id = 0 : i32} : (index) -> (i32, none) loc(#loc36)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc36)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc36)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc45)
    %trueResult_9, %falseResult_10 = handshake.cond_br %10, %26 : none loc(#loc53)
    %27 = handshake.constant %26 {value = 0 : index} : index loc(#loc53)
    %28 = handshake.constant %26 {value = 1 : index} : index loc(#loc53)
    %29 = arith.select %10, %28, %27 : index loc(#loc53)
    %30 = handshake.mux %29 [%24#1, %trueResult_9] : index, none loc(#loc53)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %30 : none loc(#loc45)
    %31 = handshake.constant %0 {value = 0 : index} : index loc(#loc45)
    %32 = handshake.constant %0 {value = 1 : index} : index loc(#loc45)
    %33 = arith.select %3, %32, %31 : index loc(#loc45)
    %34 = handshake.mux %33 [%falseResult_12, %trueResult] : index, none loc(#loc45)
    %35 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc45)
    %trueResult_13, %falseResult_14 = handshake.cond_br %10, %35 : none loc(#loc53)
    %36 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %25 : none loc(#loc53)
    %37 = handshake.constant %35 {value = 0 : index} : index loc(#loc53)
    %38 = handshake.constant %35 {value = 1 : index} : index loc(#loc53)
    %39 = arith.select %10, %38, %37 : index loc(#loc53)
    %40 = handshake.mux %39 [%falseResult_16, %trueResult_13] : index, none loc(#loc53)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %40 : none loc(#loc45)
    %41 = handshake.mux %33 [%falseResult_18, %trueResult] : index, none loc(#loc45)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_23 : i1, none, none -> none loc(#loc45)
    %trueResult_19, %falseResult_20 = handshake.cond_br %10, %42 : none loc(#loc53)
    %43 = dataflow.carry %willContinue_3, %falseResult_20, %trueResult_21 : i1, none, none -> none loc(#loc53)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_3, %23#1 : none loc(#loc53)
    %44 = handshake.constant %42 {value = 0 : index} : index loc(#loc53)
    %45 = handshake.constant %42 {value = 1 : index} : index loc(#loc53)
    %46 = arith.select %10, %45, %44 : index loc(#loc53)
    %47 = handshake.mux %46 [%falseResult_22, %trueResult_19] : index, none loc(#loc53)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue, %47 : none loc(#loc45)
    %48 = handshake.mux %33 [%falseResult_24, %trueResult] : index, none loc(#loc45)
    %49 = handshake.join %34, %41, %48 : none, none, none loc(#loc36)
    handshake.return %49 : none loc(#loc37)
  } loc(#loc36)
  func.func @_Z9outer_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg3: i32 loc(fused<#di_subprogram5>[#loc20]), %arg4: i32 loc(fused<#di_subprogram5>[#loc20])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc49)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg4, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg3 : i32 to i64 loc(#loc49)
      %3 = arith.extui %arg4 : i32 to i64 loc(#loc2)
      %4 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        scf.if %1 {
        } else {
          %7 = arith.index_cast %arg5 : i64 to index loc(#loc2)
          %8 = memref.load %arg0[%7] : memref<?xi32, strided<[1], offset: ?>> loc(#loc2)
          %9 = arith.trunci %arg5 : i64 to i32 loc(#loc2)
          %10 = arith.muli %arg4, %9 : i32 loc(#loc2)
          %11 = scf.while (%arg6 = %c0_i64) : (i64) -> i64 {
            %12 = arith.index_cast %arg6 : i64 to index loc(#loc57)
            %13 = memref.load %arg1[%12] : memref<?xi32, strided<[1], offset: ?>> loc(#loc57)
            %14 = arith.muli %13, %8 : i32 loc(#loc57)
            %15 = arith.trunci %arg6 : i64 to i32 loc(#loc57)
            %16 = arith.addi %10, %15 : i32 loc(#loc57)
            %17 = arith.extui %16 : i32 to i64 loc(#loc57)
            %18 = arith.index_cast %17 : i64 to index loc(#loc57)
            memref.store %14, %arg2[%18] : memref<?xi32, strided<[1], offset: ?>> loc(#loc57)
            %19 = arith.addi %arg6, %c1_i64 : i64 loc(#loc55)
            %20 = arith.cmpi ne, %19, %3 : i64 loc(#loc58)
            scf.condition(%20) %19 : i64 loc(#loc54)
          } do {
          ^bb0(%arg6: i64 loc(fused<#di_lexical_block29>[#loc22])):
            scf.yield %arg6 : i64 loc(#loc54)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc54)
        } loc(#loc54)
        %5 = arith.addi %arg5, %c1_i64 : i64 loc(#loc49)
        %6 = arith.cmpi ne, %5, %2 : i64 loc(#loc52)
        scf.condition(%6) %5 : i64 loc(#loc46)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block21>[#loc21])):
        scf.yield %arg5 : i64 loc(#loc46)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    } loc(#loc46)
    return loc(#loc39)
  } loc(#loc38)
} loc(#loc)
#loc = loc("tests/app/outer/main.cpp":0:0)
#loc1 = loc("tests/app/outer/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/outer/main.cpp":19:0)
#loc6 = loc("tests/app/outer/main.cpp":22:0)
#loc7 = loc("tests/app/outer/main.cpp":26:0)
#loc8 = loc("tests/app/outer/main.cpp":29:0)
#loc10 = loc("tests/app/outer/main.cpp":33:0)
#loc11 = loc("tests/app/outer/main.cpp":34:0)
#loc12 = loc("tests/app/outer/main.cpp":35:0)
#loc13 = loc("tests/app/outer/main.cpp":39:0)
#loc14 = loc("tests/app/outer/main.cpp":41:0)
#loc16 = loc("tests/app/outer/outer.cpp":37:0)
#loc17 = loc("tests/app/outer/outer.cpp":38:0)
#loc18 = loc("tests/app/outer/outer.cpp":39:0)
#loc19 = loc("tests/app/outer/outer.cpp":42:0)
#loc23 = loc("tests/app/outer/outer.cpp":23:0)
#loc24 = loc("tests/app/outer/outer.cpp":26:0)
#loc25 = loc(fused<#di_subprogram3>[#loc1])
#loc26 = loc(fused<#di_subprogram3>[#loc7])
#loc27 = loc(fused<#di_subprogram3>[#loc8])
#loc28 = loc(fused<#di_subprogram3>[#loc13])
#loc29 = loc(fused<#di_subprogram3>[#loc14])
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 18>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 21>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 32>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 18>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 21>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 32>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 37>
#loc33 = loc(fused<#di_lexical_block14>[#loc3])
#loc34 = loc(fused<#di_lexical_block15>[#loc5])
#loc35 = loc(fused<#di_lexical_block16>[#loc9])
#loc37 = loc(fused<#di_subprogram4>[#loc19])
#loc39 = loc(fused<#di_subprogram5>[#loc24])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 33>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 37>
#loc40 = loc(fused<#di_lexical_block17>[#loc4])
#loc41 = loc(fused[#loc30, #loc33])
#loc42 = loc(fused<#di_lexical_block18>[#loc6])
#loc43 = loc(fused[#loc31, #loc34])
#loc44 = loc(fused[#loc32, #loc35])
#loc45 = loc(fused<#di_lexical_block20>[#loc16])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file, line = 33>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 37>
#loc47 = loc(fused<#di_lexical_block22>[#loc10])
#loc48 = loc(fused<#di_lexical_block23>[#loc16])
#loc49 = loc(fused<#di_lexical_block24>[#loc21])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 38>
#loc50 = loc(fused<#di_lexical_block25>[#loc11])
#loc51 = loc(fused<#di_lexical_block25>[#loc12])
#loc52 = loc(fused[#loc46, #loc49])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 38>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file1, line = 22>
#loc53 = loc(fused<#di_lexical_block28>[#loc17])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 38>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file1, line = 22>
#loc55 = loc(fused<#di_lexical_block31>[#loc22])
#loc56 = loc(fused<#di_lexical_block32>[#loc18])
#loc57 = loc(fused<#di_lexical_block33>[#loc23])
#loc58 = loc(fused[#loc54, #loc55])
