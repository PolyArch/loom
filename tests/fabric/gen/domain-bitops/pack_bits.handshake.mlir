#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/pack_bits/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/pack_bits/pack_bits.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/pack_bits/main.cpp":11:0)
#loc7 = loc("tests/app/pack_bits/main.cpp":26:0)
#loc13 = loc("tests/app/pack_bits/pack_bits.cpp":39:0)
#loc22 = loc("tests/app/pack_bits/pack_bits.cpp":12:0)
#loc24 = loc("tests/app/pack_bits/pack_bits.cpp":18:0)
#loc26 = loc("tests/app/pack_bits/pack_bits.cpp":26:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 11>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 26>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 45>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 18>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 45>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 18>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 3200, elements = #llvm.di_subrange<count = 100 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 128, elements = #llvm.di_subrange<count = 4 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 45>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 18>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 11, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "num_words", file = #di_file1, line = 43, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "word_idx", file = #di_file1, line = 45, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "num_words", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "word_idx", file = #di_file1, line = 18, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 53>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 26>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "num_bits", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "num_words", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_bits", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 16, type = #di_composite_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 17, type = #di_composite_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "num_bits", file = #di_file1, line = 41, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "bits_per_word", file = #di_file1, line = 42, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "packed_word", file = #di_file1, line = 46, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "start_bit", file = #di_file1, line = 47, type = #di_derived_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "end_bit", file = #di_file1, line = 48, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "num_bits", file = #di_file1, line = 14, arg = 3, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "bits_per_word", file = #di_file1, line = 15, type = #di_derived_type2>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "packed_word", file = #di_file1, line = 19, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "start_bit", file = #di_file1, line = 20, type = #di_derived_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "end_bit", file = #di_file1, line = 21, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 53>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 26>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_packed", file = #di_file1, line = 40, arg = 2, type = #di_derived_type5>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "bit_idx", file = #di_file1, line = 53, type = #di_derived_type1>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_packed", file = #di_file1, line = 13, arg = 2, type = #di_derived_type5>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "bit_idx", file = #di_file1, line = 26, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable9, #di_local_variable10, #di_local_variable1>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 11>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 26>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 53>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 26>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_bits", file = #di_file1, line = 39, arg = 1, type = #di_derived_type6>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_bits", file = #di_file1, line = 12, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block14, name = "bit_position", file = #di_file1, line = 54, type = #di_derived_type1>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block15, name = "bit_position", file = #di_file1, line = 27, type = #di_derived_type1>
#loc36 = loc(fused<#di_lexical_block12>[#loc3])
#loc37 = loc(fused<#di_lexical_block13>[#loc7])
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "pack_bits_dsa", linkageName = "_Z13pack_bits_dsaPKjPjj", file = #di_file1, line = 39, scopeLine = 41, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable25, #di_local_variable21, #di_local_variable11, #di_local_variable12, #di_local_variable2, #di_local_variable3, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable22, #di_local_variable27>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "pack_bits_cpu", linkageName = "_Z13pack_bits_cpuPKjPjj", file = #di_file1, line = 12, scopeLine = 14, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable26, #di_local_variable23, #di_local_variable16, #di_local_variable17, #di_local_variable4, #di_local_variable5, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable24, #di_local_variable28>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 18>
#loc43 = loc(fused<#di_subprogram4>[#loc13])
#loc46 = loc(fused<#di_subprogram5>[#loc22])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 18>
#loc51 = loc(fused<#di_lexical_block22>[#loc24])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 18>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 26>
#loc63 = loc(fused<#di_lexical_block30>[#loc26])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<18xi8> = dense<[112, 97, 99, 107, 95, 98, 105, 116, 115, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<18xi8> = dense<[112, 97, 99, 107, 95, 98, 105, 116, 115, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<34xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 112, 97, 99, 107, 95, 98, 105, 116, 115, 47, 112, 97, 99, 107, 95, 98, 105, 116, 115, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc31)
    %false = arith.constant false loc(#loc31)
    %0 = seq.const_clock  low loc(#loc31)
    %c2_i32 = arith.constant 2 : i32 loc(#loc31)
    %1 = ub.poison : i64 loc(#loc31)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c4_i64 = arith.constant 4 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c100_i64 = arith.constant 100 : i64 loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<18xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<18xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<100xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<4xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<4xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc40)
      %11 = arith.remui %10, %c3_i32 : i32 loc(#loc40)
      %12 = arith.cmpi eq, %11, %c0_i32 : i32 loc(#loc40)
      %13 = arith.extui %12 : i1 to i32 loc(#loc40)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc40)
      memref.store %13, %alloca[%14] : memref<100xi32> loc(#loc40)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc38)
      %16 = arith.cmpi ne, %15, %c100_i64 : i64 loc(#loc41)
      scf.condition(%16) %15 : i64 loc(#loc36)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc36)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc36)
    %cast = memref.cast %alloca : memref<100xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc32)
    %cast_2 = memref.cast %alloca_0 : memref<4xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc32)
    call @_Z13pack_bits_cpuPKjPjj(%cast, %cast_2, %c100_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc32)
    %cast_3 = memref.cast %alloca_1 : memref<4xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c100_i32, %true : i32 loc(#loc33)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc33)
    %5 = handshake.esi_instance @_Z13pack_bits_dsaPKjPjj_esi "_Z13pack_bits_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc33)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc33)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      %11 = memref.load %alloca_0[%10] : memref<4xi32> loc(#loc49)
      %12 = memref.load %alloca_1[%10] : memref<4xi32> loc(#loc49)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc49)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc39)
        %17 = arith.cmpi eq, %16, %c4_i64 : i64 loc(#loc39)
        %18 = arith.extui %17 : i1 to i32 loc(#loc37)
        %19 = arith.cmpi ne, %16, %c4_i64 : i64 loc(#loc42)
        %20 = arith.extui %19 : i1 to i32 loc(#loc37)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc49)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc49)
      } loc(#loc49)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc37)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc37)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc7]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc7]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc7])):
      scf.yield %arg0 : i64 loc(#loc37)
    } loc(#loc37)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc37)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc37)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<18xi8> -> index loc(#loc52)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc52)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc52)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc52)
      scf.yield %c1_i32 : i32 loc(#loc53)
    } loc(#loc37)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<18xi8> -> index loc(#loc34)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc34)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc34)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc34)
    } loc(#loc2)
    return %9 : i32 loc(#loc35)
  } loc(#loc31)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z13pack_bits_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: i1 loc(fused<#di_subprogram4>[#loc13]), ...) -> i1 attributes {argNames = ["input_bits", "output_packed", "num_bits", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc43)
    %1 = handshake.join %0 : none loc(#loc43)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 32 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 32 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %1 {value = 5 : i64} : i64 loc(#loc2)
    %7 = handshake.constant %1 {value = 5 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %1 {value = 31 : i32} : i32 loc(#loc2)
    %9 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %10 = arith.addi %arg2, %8 : i32 loc(#loc44)
    %11 = arith.shrui %10, %7 : i32 loc(#loc44)
    %12 = arith.cmpi eq, %11, %3 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %12, %1 : none loc(#loc50)
    %13 = arith.extui %arg2 : i32 to i64 loc(#loc50)
    %14 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc50)
    %15 = arith.index_cast %9 : i64 to index loc(#loc50)
    %16 = arith.index_cast %11 : i32 to index loc(#loc50)
    %index, %willContinue = dataflow.stream %15, %14, %16 {step_op = "+=", stop_cond = "!="} loc(#loc50)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc50)
    %17 = dataflow.carry %willContinue, %9, %43 : i1, i64, i64 -> i64 loc(#loc50)
    %afterValue_0, %afterCond_1 = dataflow.gate %17, %willContinue : i64, i1 -> i64, i1 loc(#loc50)
    handshake.sink %afterCond_1 : i1 loc(#loc50)
    %18 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc50)
    %19 = arith.index_cast %afterValue : index to i64 loc(#loc50)
    %20 = arith.shli %19, %6 : i64 loc(#loc56)
    %21 = dataflow.invariant %afterCond, %13 : i1, i64 -> i64 loc(#loc65)
    %22 = arith.cmpi ult, %20, %21 : i64 loc(#loc65)
    %trueResult_2, %falseResult_3 = handshake.cond_br %22, %18 : none loc(#loc61)
    handshake.sink %falseResult_3 : none loc(#loc61)
    %23 = arith.trunci %20 : i64 to i32 loc(#loc62)
    %24 = arith.addi %23, %5 : i32 loc(#loc62)
    %25 = arith.cmpi ult, %24, %arg2 : i32 loc(#loc62)
    %26 = arith.select %25, %24, %arg2 : i32 loc(#loc62)
    %27 = handshake.constant %trueResult_2 {value = 1 : index} : index loc(#loc61)
    %28 = arith.index_cast %afterValue_0 : i64 to index loc(#loc61)
    %29 = arith.index_cast %26 : i32 to index loc(#loc61)
    %index_4, %willContinue_5 = dataflow.stream %28, %27, %29 {step_op = "+=", stop_cond = "<"} loc(#loc61)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc61)
    %30 = dataflow.carry %willContinue_5, %3, %38 : i1, i32, i32 -> i32 loc(#loc61)
    %afterValue_8, %afterCond_9 = dataflow.gate %30, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc61)
    handshake.sink %afterCond_9 : i1 loc(#loc61)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %30 : i32 loc(#loc61)
    %31 = arith.index_cast %afterValue_6 : index to i64 loc(#loc61)
    %dataResult, %addressResults = handshake.load [%afterValue_6] %44#0, %47 : index, i32 loc(#loc67)
    %32 = arith.andi %dataResult, %2 : i32 loc(#loc67)
    %33 = arith.cmpi eq, %32, %3 : i32 loc(#loc67)
    %34 = arith.subi %31, %20 : i64 loc(#loc67)
    %35 = arith.trunci %34 : i64 to i32 loc(#loc67)
    %36 = arith.shli %2, %35 : i32 loc(#loc67)
    %37 = arith.select %33, %3, %36 : i32 loc(#loc67)
    %38 = arith.ori %37, %afterValue_8 : i32 loc(#loc67)
    %39 = handshake.constant %18 {value = 0 : index} : index loc(#loc61)
    %40 = handshake.constant %18 {value = 1 : index} : index loc(#loc61)
    %41 = arith.select %22, %40, %39 : index loc(#loc61)
    %42 = handshake.mux %41 [%3, %falseResult_11] : index, i32 loc(#loc61)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %42, %56 : index, i32 loc(#loc57)
    %43 = arith.addi %afterValue_0, %4 : i64 loc(#loc50)
    %44:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc43)
    %45 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc43)
    %46 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc50)
    %trueResult_13, %falseResult_14 = handshake.cond_br %22, %46 : none loc(#loc61)
    %47 = dataflow.carry %willContinue_5, %trueResult_13, %trueResult_15 : i1, none, none -> none loc(#loc61)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_5, %44#1 : none loc(#loc61)
    %48 = handshake.constant %46 {value = 0 : index} : index loc(#loc61)
    %49 = handshake.constant %46 {value = 1 : index} : index loc(#loc61)
    %50 = arith.select %22, %49, %48 : index loc(#loc61)
    %51 = handshake.mux %50 [%falseResult_14, %falseResult_16] : index, none loc(#loc61)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %51 : none loc(#loc50)
    %52 = handshake.constant %1 {value = 0 : index} : index loc(#loc50)
    %53 = handshake.constant %1 {value = 1 : index} : index loc(#loc50)
    %54 = arith.select %12, %53, %52 : index loc(#loc50)
    %55 = handshake.mux %54 [%falseResult_18, %trueResult] : index, none loc(#loc50)
    %56 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc50)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %45 : none loc(#loc50)
    %57 = handshake.mux %54 [%falseResult_20, %trueResult] : index, none loc(#loc50)
    %58 = handshake.join %55, %57 : none, none loc(#loc43)
    %59 = handshake.constant %58 {value = true} : i1 loc(#loc43)
    handshake.return %59 : i1 loc(#loc43)
  } loc(#loc43)
  handshake.func @_Z13pack_bits_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: none loc(fused<#di_subprogram4>[#loc13]), ...) -> none attributes {argNames = ["input_bits", "output_packed", "num_bits", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc43)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 32 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 32 : i32} : i32 loc(#loc2)
    %5 = handshake.constant %0 {value = 5 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %0 {value = 5 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %0 {value = 31 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %9 = arith.addi %arg2, %7 : i32 loc(#loc44)
    %10 = arith.shrui %9, %6 : i32 loc(#loc44)
    %11 = arith.cmpi eq, %10, %2 : i32 loc(#loc54)
    %trueResult, %falseResult = handshake.cond_br %11, %0 : none loc(#loc50)
    %12 = arith.extui %arg2 : i32 to i64 loc(#loc50)
    %13 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc50)
    %14 = arith.index_cast %8 : i64 to index loc(#loc50)
    %15 = arith.index_cast %10 : i32 to index loc(#loc50)
    %index, %willContinue = dataflow.stream %14, %13, %15 {step_op = "+=", stop_cond = "!="} loc(#loc50)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc50)
    %16 = dataflow.carry %willContinue, %8, %42 : i1, i64, i64 -> i64 loc(#loc50)
    %afterValue_0, %afterCond_1 = dataflow.gate %16, %willContinue : i64, i1 -> i64, i1 loc(#loc50)
    handshake.sink %afterCond_1 : i1 loc(#loc50)
    %17 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc50)
    %18 = arith.index_cast %afterValue : index to i64 loc(#loc50)
    %19 = arith.shli %18, %5 : i64 loc(#loc56)
    %20 = dataflow.invariant %afterCond, %12 : i1, i64 -> i64 loc(#loc65)
    %21 = arith.cmpi ult, %19, %20 : i64 loc(#loc65)
    %trueResult_2, %falseResult_3 = handshake.cond_br %21, %17 : none loc(#loc61)
    handshake.sink %falseResult_3 : none loc(#loc61)
    %22 = arith.trunci %19 : i64 to i32 loc(#loc62)
    %23 = arith.addi %22, %4 : i32 loc(#loc62)
    %24 = arith.cmpi ult, %23, %arg2 : i32 loc(#loc62)
    %25 = arith.select %24, %23, %arg2 : i32 loc(#loc62)
    %26 = handshake.constant %trueResult_2 {value = 1 : index} : index loc(#loc61)
    %27 = arith.index_cast %afterValue_0 : i64 to index loc(#loc61)
    %28 = arith.index_cast %25 : i32 to index loc(#loc61)
    %index_4, %willContinue_5 = dataflow.stream %27, %26, %28 {step_op = "+=", stop_cond = "<"} loc(#loc61)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc61)
    %29 = dataflow.carry %willContinue_5, %2, %37 : i1, i32, i32 -> i32 loc(#loc61)
    %afterValue_8, %afterCond_9 = dataflow.gate %29, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc61)
    handshake.sink %afterCond_9 : i1 loc(#loc61)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %29 : i32 loc(#loc61)
    %30 = arith.index_cast %afterValue_6 : index to i64 loc(#loc61)
    %dataResult, %addressResults = handshake.load [%afterValue_6] %43#0, %46 : index, i32 loc(#loc67)
    %31 = arith.andi %dataResult, %1 : i32 loc(#loc67)
    %32 = arith.cmpi eq, %31, %2 : i32 loc(#loc67)
    %33 = arith.subi %30, %19 : i64 loc(#loc67)
    %34 = arith.trunci %33 : i64 to i32 loc(#loc67)
    %35 = arith.shli %1, %34 : i32 loc(#loc67)
    %36 = arith.select %32, %2, %35 : i32 loc(#loc67)
    %37 = arith.ori %36, %afterValue_8 : i32 loc(#loc67)
    %38 = handshake.constant %17 {value = 0 : index} : index loc(#loc61)
    %39 = handshake.constant %17 {value = 1 : index} : index loc(#loc61)
    %40 = arith.select %21, %39, %38 : index loc(#loc61)
    %41 = handshake.mux %40 [%2, %falseResult_11] : index, i32 loc(#loc61)
    %dataResult_12, %addressResult = handshake.store [%afterValue] %41, %55 : index, i32 loc(#loc57)
    %42 = arith.addi %afterValue_0, %3 : i64 loc(#loc50)
    %43:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc43)
    %44 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_12, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc43)
    %45 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc50)
    %trueResult_13, %falseResult_14 = handshake.cond_br %21, %45 : none loc(#loc61)
    %46 = dataflow.carry %willContinue_5, %trueResult_13, %trueResult_15 : i1, none, none -> none loc(#loc61)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_5, %43#1 : none loc(#loc61)
    %47 = handshake.constant %45 {value = 0 : index} : index loc(#loc61)
    %48 = handshake.constant %45 {value = 1 : index} : index loc(#loc61)
    %49 = arith.select %21, %48, %47 : index loc(#loc61)
    %50 = handshake.mux %49 [%falseResult_14, %falseResult_16] : index, none loc(#loc61)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %50 : none loc(#loc50)
    %51 = handshake.constant %0 {value = 0 : index} : index loc(#loc50)
    %52 = handshake.constant %0 {value = 1 : index} : index loc(#loc50)
    %53 = arith.select %11, %52, %51 : index loc(#loc50)
    %54 = handshake.mux %53 [%falseResult_18, %trueResult] : index, none loc(#loc50)
    %55 = dataflow.carry %willContinue, %falseResult, %trueResult_19 : i1, none, none -> none loc(#loc50)
    %trueResult_19, %falseResult_20 = handshake.cond_br %willContinue, %44 : none loc(#loc50)
    %56 = handshake.mux %53 [%falseResult_20, %trueResult] : index, none loc(#loc50)
    %57 = handshake.join %54, %56 : none, none loc(#loc43)
    handshake.return %57 : none loc(#loc45)
  } loc(#loc43)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.umin.i32(i32, i32) -> i32 loc(#loc2)
  func.func @_Z13pack_bits_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg2: i32 loc(fused<#di_subprogram5>[#loc22])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c5_i64 = arith.constant 5 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %0 = arith.addi %arg2, %c31_i32 : i32 loc(#loc47)
    %1 = arith.shrui %0, %c5_i32 : i32 loc(#loc47)
    %2 = arith.cmpi eq, %1, %c0_i32 : i32 loc(#loc55)
    scf.if %2 {
    } else {
      %3 = arith.extui %arg2 : i32 to i64 loc(#loc51)
      %4 = arith.extui %1 : i32 to i64 loc(#loc55)
      %5:2 = scf.while (%arg3 = %c0_i64, %arg4 = %c0_i64) : (i64, i64) -> (i64, i64) {
        %6 = arith.shli %arg3, %c5_i64 : i64 loc(#loc58)
        %7 = arith.cmpi ult, %6, %3 : i64 loc(#loc66)
        %8 = scf.if %7 -> (i32) {
          %13 = arith.trunci %6 : i64 to i32 loc(#loc64)
          %14 = arith.addi %13, %c32_i32 : i32 loc(#loc64)
          %15 = arith.cmpi ult, %14, %arg2 : i32 loc(#loc64)
          %16 = arith.select %15, %14, %arg2 : i32 loc(#loc64)
          %17 = arith.extui %16 : i32 to i64 loc(#loc63)
          %18:2 = scf.while (%arg5 = %arg4, %arg6 = %c0_i32) : (i64, i32) -> (i64, i32) {
            %19 = arith.index_cast %arg5 : i64 to index loc(#loc68)
            %20 = memref.load %arg0[%19] : memref<?xi32, strided<[1], offset: ?>> loc(#loc68)
            %21 = arith.andi %20, %c1_i32 : i32 loc(#loc68)
            %22 = arith.cmpi eq, %21, %c0_i32 : i32 loc(#loc68)
            %23 = arith.subi %arg5, %6 : i64 loc(#loc68)
            %24 = arith.trunci %23 : i64 to i32 loc(#loc68)
            %25 = arith.shli %c1_i32, %24 : i32 loc(#loc68)
            %26 = arith.select %22, %c0_i32, %25 : i32 loc(#loc68)
            %27 = arith.ori %26, %arg6 : i32 loc(#loc68)
            %28 = arith.addi %arg5, %c1_i64 : i64 loc(#loc66)
            %29 = arith.cmpi ult, %28, %17 : i64 loc(#loc66)
            scf.condition(%29) %28, %27 : i64, i32 loc(#loc63)
          } do {
          ^bb0(%arg5: i64 loc(fused<#di_lexical_block30>[#loc26]), %arg6: i32 loc(fused<#di_lexical_block30>[#loc26])):
            scf.yield %arg5, %arg6 : i64, i32 loc(#loc63)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc63)
          scf.yield %18#1 : i32 loc(#loc63)
        } else {
          scf.yield %c0_i32 : i32 loc(#loc63)
        } loc(#loc63)
        %9 = arith.index_cast %arg3 : i64 to index loc(#loc59)
        memref.store %8, %arg1[%9] : memref<?xi32, strided<[1], offset: ?>> loc(#loc59)
        %10 = arith.addi %arg3, %c1_i64 : i64 loc(#loc55)
        %11 = arith.addi %arg4, %c32_i64 : i64 loc(#loc51)
        %12 = arith.cmpi ne, %10, %4 : i64 loc(#loc60)
        scf.condition(%12) %10, %11 : i64, i64 loc(#loc51)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block22>[#loc24]), %arg4: i64 loc(fused<#di_lexical_block22>[#loc24])):
        scf.yield %arg3, %arg4 : i64, i64 loc(#loc51)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc51)
    } loc(#loc51)
    return loc(#loc48)
  } loc(#loc46)
} loc(#loc)
#loc = loc("tests/app/pack_bits/main.cpp":0:0)
#loc1 = loc("tests/app/pack_bits/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/pack_bits/main.cpp":12:0)
#loc5 = loc("tests/app/pack_bits/main.cpp":20:0)
#loc6 = loc("tests/app/pack_bits/main.cpp":23:0)
#loc8 = loc("tests/app/pack_bits/main.cpp":27:0)
#loc9 = loc("tests/app/pack_bits/main.cpp":28:0)
#loc10 = loc("tests/app/pack_bits/main.cpp":29:0)
#loc11 = loc("tests/app/pack_bits/main.cpp":33:0)
#loc12 = loc("tests/app/pack_bits/main.cpp":35:0)
#loc14 = loc("tests/app/pack_bits/pack_bits.cpp":43:0)
#loc15 = loc("tests/app/pack_bits/pack_bits.cpp":45:0)
#loc16 = loc("tests/app/pack_bits/pack_bits.cpp":47:0)
#loc17 = loc("tests/app/pack_bits/pack_bits.cpp":53:0)
#loc18 = loc("tests/app/pack_bits/pack_bits.cpp":49:0)
#loc19 = loc("tests/app/pack_bits/pack_bits.cpp":55:0)
#loc20 = loc("tests/app/pack_bits/pack_bits.cpp":60:0)
#loc21 = loc("tests/app/pack_bits/pack_bits.cpp":62:0)
#loc23 = loc("tests/app/pack_bits/pack_bits.cpp":16:0)
#loc25 = loc("tests/app/pack_bits/pack_bits.cpp":20:0)
#loc27 = loc("tests/app/pack_bits/pack_bits.cpp":22:0)
#loc28 = loc("tests/app/pack_bits/pack_bits.cpp":28:0)
#loc29 = loc("tests/app/pack_bits/pack_bits.cpp":33:0)
#loc30 = loc("tests/app/pack_bits/pack_bits.cpp":35:0)
#loc31 = loc(fused<#di_subprogram3>[#loc1])
#loc32 = loc(fused<#di_subprogram3>[#loc5])
#loc33 = loc(fused<#di_subprogram3>[#loc6])
#loc34 = loc(fused<#di_subprogram3>[#loc11])
#loc35 = loc(fused<#di_subprogram3>[#loc12])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 11>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 26>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 11>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 26>
#loc38 = loc(fused<#di_lexical_block16>[#loc3])
#loc39 = loc(fused<#di_lexical_block17>[#loc7])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 27>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 45>
#loc40 = loc(fused<#di_lexical_block18>[#loc4])
#loc41 = loc(fused[#loc36, #loc38])
#loc42 = loc(fused[#loc37, #loc39])
#loc44 = loc(fused<#di_subprogram4>[#loc14])
#loc45 = loc(fused<#di_subprogram4>[#loc21])
#loc47 = loc(fused<#di_subprogram5>[#loc23])
#loc48 = loc(fused<#di_subprogram5>[#loc30])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 27>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 45>
#loc49 = loc(fused<#di_lexical_block20>[#loc8])
#loc50 = loc(fused<#di_lexical_block21>[#loc15])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 45>
#loc52 = loc(fused<#di_lexical_block23>[#loc9])
#loc53 = loc(fused<#di_lexical_block23>[#loc10])
#loc54 = loc(fused<#di_lexical_block24>[#loc15])
#loc55 = loc(fused<#di_lexical_block25>[#loc24])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 53>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 49>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 22>
#loc56 = loc(fused<#di_lexical_block26>[#loc16])
#loc57 = loc(fused<#di_lexical_block26>[#loc20])
#loc58 = loc(fused<#di_lexical_block27>[#loc25])
#loc59 = loc(fused<#di_lexical_block27>[#loc29])
#loc60 = loc(fused[#loc51, #loc55])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 53>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 26>
#loc61 = loc(fused<#di_lexical_block28>[#loc17])
#loc62 = loc(fused<#di_lexical_block29>[#loc18])
#loc64 = loc(fused<#di_lexical_block31>[#loc27])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file1, line = 53>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file1, line = 26>
#loc65 = loc(fused<#di_lexical_block32>[#loc17])
#loc66 = loc(fused<#di_lexical_block33>[#loc26])
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file1, line = 55>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file1, line = 28>
#loc67 = loc(fused<#di_lexical_block36>[#loc19])
#loc68 = loc(fused<#di_lexical_block37>[#loc28])
