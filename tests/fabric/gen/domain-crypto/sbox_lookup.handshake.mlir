#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/sbox_lookup/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/sbox_lookup/sbox_lookup.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/sbox_lookup/main.cpp":20:0)
#loc5 = loc("tests/app/sbox_lookup/main.cpp":25:0)
#loc9 = loc("tests/app/sbox_lookup/main.cpp":36:0)
#loc15 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":25:0)
#loc20 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":13:0)
#loc21 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":17:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 20>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 25>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 36>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 31>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 31>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 17>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 31>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 17>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 20, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 25, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 36, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 31, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "SBOX_SIZE", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "sbox", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 13, type = #di_composite_type1>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 16, type = #di_composite_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 17, type = #di_composite_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 28, arg = 4, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "index", file = #di_file1, line = 32, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 16, arg = 4, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "index", file = #di_file1, line = 18, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_result", file = #di_file1, line = 27, arg = 3, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_result", file = #di_file1, line = 15, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 20>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 25>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 36>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file1, line = 25, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_sbox", file = #di_file1, line = 26, arg = 2, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_data", file = #di_file1, line = 13, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_sbox", file = #di_file1, line = 14, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "sbox_lookup_dsa", linkageName = "_Z15sbox_lookup_dsaPKjS0_Pjj", file = #di_file1, line = 25, scopeLine = 28, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable15, #di_local_variable11, #di_local_variable3, #di_local_variable12>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "sbox_lookup_cpu", linkageName = "_Z15sbox_lookup_cpuPKjS0_Pjj", file = #di_file1, line = 13, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable16, #di_local_variable13, #di_local_variable4, #di_local_variable14>
#loc30 = loc(fused<#di_lexical_block9>[#loc3])
#loc31 = loc(fused<#di_lexical_block10>[#loc5])
#loc32 = loc(fused<#di_lexical_block11>[#loc9])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 17>
#loc36 = loc(fused<#di_subprogram4>[#loc15])
#loc38 = loc(fused<#di_subprogram5>[#loc20])
#loc46 = loc(fused<#di_lexical_block19>[#loc21])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[115, 98, 111, 120, 95, 108, 111, 111, 107, 117, 112, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[115, 98, 111, 120, 95, 108, 111, 111, 107, 117, 112, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 115, 98, 111, 120, 95, 108, 111, 111, 107, 117, 112, 47, 115, 98, 111, 120, 95, 108, 111, 111, 107, 117, 112, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc25)
    %false = arith.constant false loc(#loc25)
    %0 = seq.const_clock  low loc(#loc25)
    %c2_i32 = arith.constant 2 : i32 loc(#loc25)
    %1 = ub.poison : i64 loc(#loc25)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c13_i32 = arith.constant 13 : i32 loc(#loc2)
    %c17_i32 = arith.constant 17 : i32 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc40)
      %12 = arith.muli %11, %c7_i32 : i32 loc(#loc40)
      %13 = arith.addi %12, %c31_i32 : i32 loc(#loc40)
      %14 = arith.andi %13, %c255_i32 : i32 loc(#loc40)
      %15 = arith.index_cast %arg0 : i64 to index loc(#loc40)
      memref.store %14, %alloca[%15] : memref<256xi32> loc(#loc40)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc33)
      %17 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc41)
      scf.condition(%17) %16 : i64 loc(#loc30)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc30)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc30)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc42)
      %12 = arith.muli %11, %c13_i32 : i32 loc(#loc42)
      %13 = arith.addi %12, %c17_i32 : i32 loc(#loc42)
      %14 = arith.andi %13, %c255_i32 : i32 loc(#loc42)
      %15 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      memref.store %14, %alloca_0[%15] : memref<1024xi32> loc(#loc42)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc34)
      %17 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc43)
      scf.condition(%17) %16 : i64 loc(#loc31)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc5])):
      scf.yield %arg0 : i64 loc(#loc31)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc31)
    %cast = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    %cast_3 = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc26)
    call @_Z15sbox_lookup_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc26)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc27)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc27)
    %6 = handshake.esi_instance @_Z15sbox_lookup_dsaPKjS0_Pjj_esi "_Z15sbox_lookup_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc27)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc27)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc47)
      %12 = memref.load %alloca_1[%11] : memref<1024xi32> loc(#loc47)
      %13 = memref.load %alloca_2[%11] : memref<1024xi32> loc(#loc47)
      %14 = arith.cmpi eq, %12, %13 : i32 loc(#loc47)
      %15:3 = scf.if %14 -> (i64, i32, i32) {
        %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
        %18 = arith.cmpi eq, %17, %c1024_i64 : i64 loc(#loc35)
        %19 = arith.extui %18 : i1 to i32 loc(#loc32)
        %20 = arith.cmpi ne, %17, %c1024_i64 : i64 loc(#loc44)
        %21 = arith.extui %20 : i1 to i32 loc(#loc32)
        scf.yield %17, %19, %21 : i64, i32, i32 loc(#loc47)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc47)
      } loc(#loc47)
      %16 = arith.trunci %15#2 : i32 to i1 loc(#loc32)
      scf.condition(%16) %15#0, %14, %15#1 : i64, i1, i32 loc(#loc32)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc9]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc9]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc9])):
      scf.yield %arg0 : i64 loc(#loc32)
    } loc(#loc32)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc32)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc32)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc50)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc50)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc32)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc28)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc28)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc28)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc28)
    } loc(#loc2)
    return %10 : i32 loc(#loc29)
  } loc(#loc25)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z15sbox_lookup_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: i1 loc(fused<#di_subprogram4>[#loc15]), ...) -> i1 attributes {argNames = ["input_data", "input_sbox", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc36)
    %1 = handshake.join %0 : none loc(#loc36)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %1 {value = 255 : i32} : i32 loc(#loc2)
    %5 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc45)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %7 = arith.index_cast %3 : i64 to index loc(#loc45)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %12#0, %15 : index, i32 loc(#loc52)
    %9 = arith.andi %dataResult, %4 : i32 loc(#loc52)
    %10 = arith.extui %9 : i32 to i64 loc(#loc53)
    %11 = arith.index_cast %10 : i64 to index loc(#loc53)
    %dataResult_0, %addressResults_1 = handshake.load [%11] %13#0, %22 : index, i32 loc(#loc53)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %dataResult_0, %20 : index, i32 loc(#loc53)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc36)
    %13:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc36)
    %14 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc36)
    %15 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc45)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %12#1 : none loc(#loc45)
    %16 = handshake.constant %1 {value = 0 : index} : index loc(#loc45)
    %17 = handshake.constant %1 {value = 1 : index} : index loc(#loc45)
    %18 = arith.select %5, %17, %16 : index loc(#loc45)
    %19 = handshake.mux %18 [%falseResult_4, %trueResult] : index, none loc(#loc45)
    %20 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc45)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %14 : none loc(#loc45)
    %21 = handshake.mux %18 [%falseResult_6, %trueResult] : index, none loc(#loc45)
    %22 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc45)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %13#1 : none loc(#loc45)
    %23 = handshake.mux %18 [%falseResult_8, %trueResult] : index, none loc(#loc45)
    %24 = handshake.join %19, %21, %23 : none, none, none loc(#loc36)
    %25 = handshake.constant %24 {value = true} : i1 loc(#loc36)
    handshake.return %25 : i1 loc(#loc36)
  } loc(#loc36)
  handshake.func @_Z15sbox_lookup_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc15]), %arg3: i32 loc(fused<#di_subprogram4>[#loc15]), %arg4: none loc(fused<#di_subprogram4>[#loc15]), ...) -> none attributes {argNames = ["input_data", "input_sbox", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc36)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = handshake.constant %0 {value = 255 : i32} : i32 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc48)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc45)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc45)
    %6 = arith.index_cast %2 : i64 to index loc(#loc45)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc45)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc45)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc45)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %14 : index, i32 loc(#loc52)
    %8 = arith.andi %dataResult, %3 : i32 loc(#loc52)
    %9 = arith.extui %8 : i32 to i64 loc(#loc53)
    %10 = arith.index_cast %9 : i64 to index loc(#loc53)
    %dataResult_0, %addressResults_1 = handshake.load [%10] %12#0, %21 : index, i32 loc(#loc53)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %dataResult_0, %19 : index, i32 loc(#loc53)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc36)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc36)
    %13 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc36)
    %14 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc45)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %11#1 : none loc(#loc45)
    %15 = handshake.constant %0 {value = 0 : index} : index loc(#loc45)
    %16 = handshake.constant %0 {value = 1 : index} : index loc(#loc45)
    %17 = arith.select %4, %16, %15 : index loc(#loc45)
    %18 = handshake.mux %17 [%falseResult_4, %trueResult] : index, none loc(#loc45)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc45)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %13 : none loc(#loc45)
    %20 = handshake.mux %17 [%falseResult_6, %trueResult] : index, none loc(#loc45)
    %21 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc45)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %12#1 : none loc(#loc45)
    %22 = handshake.mux %17 [%falseResult_8, %trueResult] : index, none loc(#loc45)
    %23 = handshake.join %18, %20, %22 : none, none, none loc(#loc36)
    handshake.return %23 : none loc(#loc37)
  } loc(#loc36)
  func.func @_Z15sbox_lookup_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg3: i32 loc(fused<#di_subprogram5>[#loc20])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc49)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc49)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc54)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc54)
        %5 = arith.andi %4, %c255_i32 : i32 loc(#loc54)
        %6 = arith.extui %5 : i32 to i64 loc(#loc55)
        %7 = arith.index_cast %6 : i64 to index loc(#loc55)
        %8 = memref.load %arg1[%7] : memref<?xi32, strided<[1], offset: ?>> loc(#loc55)
        memref.store %8, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc55)
        %9 = arith.addi %arg4, %c1_i64 : i64 loc(#loc49)
        %10 = arith.cmpi ne, %9, %1 : i64 loc(#loc56)
        scf.condition(%10) %9 : i64 loc(#loc46)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block19>[#loc21])):
        scf.yield %arg4 : i64 loc(#loc46)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    } loc(#loc46)
    return loc(#loc39)
  } loc(#loc38)
} loc(#loc)
#loc = loc("tests/app/sbox_lookup/main.cpp":0:0)
#loc1 = loc("tests/app/sbox_lookup/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/sbox_lookup/main.cpp":21:0)
#loc6 = loc("tests/app/sbox_lookup/main.cpp":26:0)
#loc7 = loc("tests/app/sbox_lookup/main.cpp":30:0)
#loc8 = loc("tests/app/sbox_lookup/main.cpp":33:0)
#loc10 = loc("tests/app/sbox_lookup/main.cpp":37:0)
#loc11 = loc("tests/app/sbox_lookup/main.cpp":38:0)
#loc12 = loc("tests/app/sbox_lookup/main.cpp":39:0)
#loc13 = loc("tests/app/sbox_lookup/main.cpp":43:0)
#loc14 = loc("tests/app/sbox_lookup/main.cpp":45:0)
#loc16 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":31:0)
#loc17 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":32:0)
#loc18 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":33:0)
#loc19 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":35:0)
#loc22 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":18:0)
#loc23 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":19:0)
#loc24 = loc("tests/app/sbox_lookup/sbox_lookup.cpp":21:0)
#loc25 = loc(fused<#di_subprogram3>[#loc1])
#loc26 = loc(fused<#di_subprogram3>[#loc7])
#loc27 = loc(fused<#di_subprogram3>[#loc8])
#loc28 = loc(fused<#di_subprogram3>[#loc13])
#loc29 = loc(fused<#di_subprogram3>[#loc14])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 20>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 25>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 36>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 20>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 25>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 36>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 31>
#loc33 = loc(fused<#di_lexical_block12>[#loc3])
#loc34 = loc(fused<#di_lexical_block13>[#loc5])
#loc35 = loc(fused<#di_lexical_block14>[#loc9])
#loc37 = loc(fused<#di_subprogram4>[#loc19])
#loc39 = loc(fused<#di_subprogram5>[#loc24])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 37>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 31>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 17>
#loc40 = loc(fused<#di_lexical_block15>[#loc4])
#loc41 = loc(fused[#loc30, #loc33])
#loc42 = loc(fused<#di_lexical_block16>[#loc6])
#loc43 = loc(fused[#loc31, #loc34])
#loc44 = loc(fused[#loc32, #loc35])
#loc45 = loc(fused<#di_lexical_block18>[#loc16])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 37>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 31>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 17>
#loc47 = loc(fused<#di_lexical_block20>[#loc10])
#loc48 = loc(fused<#di_lexical_block21>[#loc16])
#loc49 = loc(fused<#di_lexical_block22>[#loc21])
#loc50 = loc(fused<#di_lexical_block23>[#loc11])
#loc51 = loc(fused<#di_lexical_block23>[#loc12])
#loc52 = loc(fused<#di_lexical_block24>[#loc17])
#loc53 = loc(fused<#di_lexical_block24>[#loc18])
#loc54 = loc(fused<#di_lexical_block25>[#loc22])
#loc55 = loc(fused<#di_lexical_block25>[#loc23])
#loc56 = loc(fused[#loc46, #loc49])
