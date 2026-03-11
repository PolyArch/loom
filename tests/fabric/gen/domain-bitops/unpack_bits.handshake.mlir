#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/unpack_bits/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/unpack_bits/unpack_bits.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc6 = loc("tests/app/unpack_bits/main.cpp":26:0)
#loc12 = loc("tests/app/unpack_bits/unpack_bits.cpp":31:0)
#loc22 = loc("tests/app/unpack_bits/unpack_bits.cpp":8:0)
#loc24 = loc("tests/app/unpack_bits/unpack_bits.cpp":14:0)
#loc27 = loc("tests/app/unpack_bits/unpack_bits.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 11>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 26>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 37>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 14>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 37>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 14>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 128, elements = #llvm.di_subrange<count = 4 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 3200, elements = #llvm.di_subrange<count = 100 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 37>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 14>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 11, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "num_words", file = #di_file1, line = 35, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "word_idx", file = #di_file1, line = 37, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram2, name = "num_words", file = #di_file1, line = 12, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "word_idx", file = #di_file1, line = 14, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 45>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 22>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "num_bits", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "num_words", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_packed", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 16, type = #di_composite_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 17, type = #di_composite_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "num_bits", file = #di_file1, line = 33, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "bits_per_word", file = #di_file1, line = 34, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "packed_word", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "start_bit", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "end_bit", file = #di_file1, line = 40, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "num_bits", file = #di_file1, line = 10, arg = 3, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "bits_per_word", file = #di_file1, line = 11, type = #di_derived_type2>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "packed_word", file = #di_file1, line = 15, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "start_bit", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "end_bit", file = #di_file1, line = 17, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 45>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 22>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_bits", file = #di_file1, line = 32, arg = 2, type = #di_derived_type5>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "bit_idx", file = #di_file1, line = 45, type = #di_derived_type1>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_bits", file = #di_file1, line = 9, arg = 2, type = #di_derived_type5>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "bit_idx", file = #di_file1, line = 22, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable9, #di_local_variable10, #di_local_variable1>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 26>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 45>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 22>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_packed", file = #di_file1, line = 31, arg = 1, type = #di_derived_type6>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_packed", file = #di_file1, line = 8, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_lexical_block14, name = "bit_position", file = #di_file1, line = 46, type = #di_derived_type1>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_lexical_block15, name = "bit_position", file = #di_file1, line = 23, type = #di_derived_type1>
#loc37 = loc(fused<#di_lexical_block13>[#loc6])
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "unpack_bits_dsa", linkageName = "_Z15unpack_bits_dsaPKjPjj", file = #di_file1, line = 31, scopeLine = 33, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable25, #di_local_variable21, #di_local_variable11, #di_local_variable12, #di_local_variable2, #di_local_variable3, #di_local_variable13, #di_local_variable14, #di_local_variable15, #di_local_variable22, #di_local_variable27>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "unpack_bits_cpu", linkageName = "_Z15unpack_bits_cpuPKjPjj", file = #di_file1, line = 8, scopeLine = 10, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable26, #di_local_variable23, #di_local_variable16, #di_local_variable17, #di_local_variable4, #di_local_variable5, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable24, #di_local_variable28>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 14>
#loc41 = loc(fused<#di_subprogram4>[#loc12])
#loc44 = loc(fused<#di_subprogram5>[#loc22])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 14>
#loc49 = loc(fused<#di_lexical_block22>[#loc24])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 14>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 22>
#loc61 = loc(fused<#di_lexical_block30>[#loc27])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[117, 110, 112, 97, 99, 107, 95, 98, 105, 116, 115, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[117, 110, 112, 97, 99, 107, 95, 98, 105, 116, 115, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 117, 110, 112, 97, 99, 107, 95, 98, 105, 116, 115, 47, 117, 110, 112, 97, 99, 107, 95, 98, 105, 116, 115, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc32)
    %false = arith.constant false loc(#loc32)
    %0 = seq.const_clock  low loc(#loc32)
    %c2_i32 = arith.constant 2 : i32 loc(#loc32)
    %1 = ub.poison : i64 loc(#loc32)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc39)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c-1431655766_i32 = arith.constant -1431655766 : i32 loc(#loc39)
    %c100_i64 = arith.constant 100 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<4xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<100xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<100xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c4 step %c1 {
      memref.store %c-1431655766_i32, %alloca[%arg0] : memref<4xi32> loc(#loc39)
    } loc(#loc39)
    %cast = memref.cast %alloca : memref<4xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %cast_2 = memref.cast %alloca_0 : memref<100xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    call @_Z15unpack_bits_cpuPKjPjj(%cast, %cast_2, %c100_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc33)
    %cast_3 = memref.cast %alloca_1 : memref<100xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c100_i32, %true : i32 loc(#loc34)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc34)
    %4 = handshake.esi_instance @_Z15unpack_bits_dsaPKjPjj_esi "_Z15unpack_bits_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc34)
    %rawOutput, %valid = esi.unwrap.vr %4, %true : i1 loc(#loc34)
    %5:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %9 = arith.index_cast %arg0 : i64 to index loc(#loc47)
      %10 = memref.load %alloca_0[%9] : memref<100xi32> loc(#loc47)
      %11 = memref.load %alloca_1[%9] : memref<100xi32> loc(#loc47)
      %12 = arith.cmpi eq, %10, %11 : i32 loc(#loc47)
      %13:3 = scf.if %12 -> (i64, i32, i32) {
        %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc38)
        %16 = arith.cmpi eq, %15, %c100_i64 : i64 loc(#loc38)
        %17 = arith.extui %16 : i1 to i32 loc(#loc37)
        %18 = arith.cmpi ne, %15, %c100_i64 : i64 loc(#loc40)
        %19 = arith.extui %18 : i1 to i32 loc(#loc37)
        scf.yield %15, %17, %19 : i64, i32, i32 loc(#loc47)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc47)
      } loc(#loc47)
      %14 = arith.trunci %13#2 : i32 to i1 loc(#loc37)
      scf.condition(%14) %13#0, %12, %13#1 : i64, i1, i32 loc(#loc37)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block13>[#loc6]), %arg1: i1 loc(fused<#di_lexical_block13>[#loc6]), %arg2: i32 loc(fused<#di_lexical_block13>[#loc6])):
      scf.yield %arg0 : i64 loc(#loc37)
    } loc(#loc37)
    %6 = arith.index_castui %5#2 : i32 to index loc(#loc37)
    %7 = scf.index_switch %6 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc37)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc50)
      %9 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %10 = llvm.inttoptr %9 : i64 to !llvm.ptr loc(#loc50)
      %11 = llvm.call @puts(%10) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc37)
    %8 = arith.select %5#1, %c0_i32, %7 : i32 loc(#loc2)
    scf.if %5#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc35)
      %9 = arith.index_cast %intptr : index to i64 loc(#loc35)
      %10 = llvm.inttoptr %9 : i64 to !llvm.ptr loc(#loc35)
      %11 = llvm.call @puts(%10) : (!llvm.ptr) -> i32 loc(#loc35)
    } loc(#loc2)
    return %8 : i32 loc(#loc36)
  } loc(#loc32)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.memset.p0.i64(memref<?xi8, strided<[1], offset: ?>>, i8, i64, i1) loc(#loc2)
  handshake.func @_Z15unpack_bits_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc12]), %arg2: i32 loc(fused<#di_subprogram4>[#loc12]), %arg3: i1 loc(fused<#di_subprogram4>[#loc12]), ...) -> i1 attributes {argNames = ["input_packed", "output_bits", "num_bits", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc41)
    %1 = handshake.join %0 : none loc(#loc41)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 32 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 5 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %1 {value = 5 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %1 {value = 32 : i32} : i32 loc(#loc2)
    %8 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %9 = handshake.constant %1 {value = 31 : i32} : i32 loc(#loc2)
    %10 = arith.addi %arg2, %9 : i32 loc(#loc42)
    %11 = arith.shrui %10, %6 : i32 loc(#loc42)
    %12 = arith.cmpi eq, %11, %3 : i32 loc(#loc52)
    %trueResult, %falseResult = handshake.cond_br %12, %1 : none loc(#loc48)
    %13 = arith.extui %arg2 : i32 to i64 loc(#loc48)
    %14 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc48)
    %15 = arith.index_cast %8 : i64 to index loc(#loc48)
    %16 = arith.index_cast %11 : i32 to index loc(#loc48)
    %index, %willContinue = dataflow.stream %15, %14, %16 {step_op = "+=", stop_cond = "!="} loc(#loc48)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc48)
    %17 = dataflow.carry %willContinue, %8, %35 : i1, i64, i64 -> i64 loc(#loc48)
    %afterValue_0, %afterCond_1 = dataflow.gate %17, %willContinue : i64, i1 -> i64, i1 loc(#loc48)
    handshake.sink %afterCond_1 : i1 loc(#loc48)
    %18 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc48)
    %19 = arith.index_cast %afterValue : index to i64 loc(#loc48)
    %dataResult, %addressResults = handshake.load [%afterValue] %36#0, %38 : index, i32 loc(#loc54)
    %20 = arith.shli %19, %5 : i64 loc(#loc55)
    %21 = dataflow.invariant %afterCond, %13 : i1, i64 -> i64 loc(#loc63)
    %22 = arith.cmpi ult, %20, %21 : i64 loc(#loc63)
    %trueResult_2, %falseResult_3 = handshake.cond_br %22, %18 : none loc(#loc59)
    handshake.sink %falseResult_3 : none loc(#loc59)
    %23 = arith.trunci %20 : i64 to i32 loc(#loc60)
    %24 = arith.addi %23, %7 : i32 loc(#loc60)
    %25 = arith.cmpi ult, %24, %arg2 : i32 loc(#loc60)
    %26 = arith.select %25, %24, %arg2 : i32 loc(#loc60)
    %27 = handshake.constant %trueResult_2 {value = 1 : index} : index loc(#loc59)
    %28 = arith.index_cast %afterValue_0 : i64 to index loc(#loc59)
    %29 = arith.index_cast %26 : i32 to index loc(#loc59)
    %index_4, %willContinue_5 = dataflow.stream %28, %27, %29 {step_op = "+=", stop_cond = "<"} loc(#loc59)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc59)
    %30 = arith.index_cast %afterValue_6 : index to i64 loc(#loc59)
    %31 = arith.subi %30, %20 : i64 loc(#loc65)
    %32 = arith.trunci %31 : i64 to i32 loc(#loc66)
    %33 = arith.shrui %dataResult, %32 : i32 loc(#loc66)
    %34 = arith.andi %33, %2 : i32 loc(#loc66)
    %dataResult_8, %addressResult = handshake.store [%afterValue_6] %34, %44 : index, i32 loc(#loc66)
    %35 = arith.addi %afterValue_0, %4 : i64 loc(#loc48)
    %36:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc41)
    %37 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc41)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc48)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %36#1 : none loc(#loc48)
    %39 = handshake.constant %1 {value = 0 : index} : index loc(#loc48)
    %40 = handshake.constant %1 {value = 1 : index} : index loc(#loc48)
    %41 = arith.select %12, %40, %39 : index loc(#loc48)
    %42 = handshake.mux %41 [%falseResult_10, %trueResult] : index, none loc(#loc48)
    %43 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc48)
    %trueResult_11, %falseResult_12 = handshake.cond_br %22, %43 : none loc(#loc59)
    %44 = dataflow.carry %willContinue_5, %trueResult_11, %trueResult_13 : i1, none, none -> none loc(#loc59)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue_5, %37 : none loc(#loc59)
    %45 = handshake.constant %43 {value = 0 : index} : index loc(#loc59)
    %46 = handshake.constant %43 {value = 1 : index} : index loc(#loc59)
    %47 = arith.select %22, %46, %45 : index loc(#loc59)
    %48 = handshake.mux %47 [%falseResult_12, %falseResult_14] : index, none loc(#loc59)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %48 : none loc(#loc48)
    %49 = handshake.mux %41 [%falseResult_16, %trueResult] : index, none loc(#loc48)
    %50 = handshake.join %42, %49 : none, none loc(#loc41)
    %51 = handshake.constant %50 {value = true} : i1 loc(#loc41)
    handshake.return %51 : i1 loc(#loc41)
  } loc(#loc41)
  handshake.func @_Z15unpack_bits_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc12]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc12]), %arg2: i32 loc(fused<#di_subprogram4>[#loc12]), %arg3: none loc(fused<#di_subprogram4>[#loc12]), ...) -> none attributes {argNames = ["input_packed", "output_bits", "num_bits", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc41)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 32 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 5 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %0 {value = 5 : i32} : i32 loc(#loc2)
    %6 = handshake.constant %0 {value = 32 : i32} : i32 loc(#loc2)
    %7 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %8 = handshake.constant %0 {value = 31 : i32} : i32 loc(#loc2)
    %9 = arith.addi %arg2, %8 : i32 loc(#loc42)
    %10 = arith.shrui %9, %5 : i32 loc(#loc42)
    %11 = arith.cmpi eq, %10, %2 : i32 loc(#loc52)
    %trueResult, %falseResult = handshake.cond_br %11, %0 : none loc(#loc48)
    %12 = arith.extui %arg2 : i32 to i64 loc(#loc48)
    %13 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc48)
    %14 = arith.index_cast %7 : i64 to index loc(#loc48)
    %15 = arith.index_cast %10 : i32 to index loc(#loc48)
    %index, %willContinue = dataflow.stream %14, %13, %15 {step_op = "+=", stop_cond = "!="} loc(#loc48)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc48)
    %16 = dataflow.carry %willContinue, %7, %34 : i1, i64, i64 -> i64 loc(#loc48)
    %afterValue_0, %afterCond_1 = dataflow.gate %16, %willContinue : i64, i1 -> i64, i1 loc(#loc48)
    handshake.sink %afterCond_1 : i1 loc(#loc48)
    %17 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc48)
    %18 = arith.index_cast %afterValue : index to i64 loc(#loc48)
    %dataResult, %addressResults = handshake.load [%afterValue] %35#0, %37 : index, i32 loc(#loc54)
    %19 = arith.shli %18, %4 : i64 loc(#loc55)
    %20 = dataflow.invariant %afterCond, %12 : i1, i64 -> i64 loc(#loc63)
    %21 = arith.cmpi ult, %19, %20 : i64 loc(#loc63)
    %trueResult_2, %falseResult_3 = handshake.cond_br %21, %17 : none loc(#loc59)
    handshake.sink %falseResult_3 : none loc(#loc59)
    %22 = arith.trunci %19 : i64 to i32 loc(#loc60)
    %23 = arith.addi %22, %6 : i32 loc(#loc60)
    %24 = arith.cmpi ult, %23, %arg2 : i32 loc(#loc60)
    %25 = arith.select %24, %23, %arg2 : i32 loc(#loc60)
    %26 = handshake.constant %trueResult_2 {value = 1 : index} : index loc(#loc59)
    %27 = arith.index_cast %afterValue_0 : i64 to index loc(#loc59)
    %28 = arith.index_cast %25 : i32 to index loc(#loc59)
    %index_4, %willContinue_5 = dataflow.stream %27, %26, %28 {step_op = "+=", stop_cond = "<"} loc(#loc59)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc59)
    %29 = arith.index_cast %afterValue_6 : index to i64 loc(#loc59)
    %30 = arith.subi %29, %19 : i64 loc(#loc65)
    %31 = arith.trunci %30 : i64 to i32 loc(#loc66)
    %32 = arith.shrui %dataResult, %31 : i32 loc(#loc66)
    %33 = arith.andi %32, %1 : i32 loc(#loc66)
    %dataResult_8, %addressResult = handshake.store [%afterValue_6] %33, %43 : index, i32 loc(#loc66)
    %34 = arith.addi %afterValue_0, %3 : i64 loc(#loc48)
    %35:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc41)
    %36 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_8, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc41)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc48)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %35#1 : none loc(#loc48)
    %38 = handshake.constant %0 {value = 0 : index} : index loc(#loc48)
    %39 = handshake.constant %0 {value = 1 : index} : index loc(#loc48)
    %40 = arith.select %11, %39, %38 : index loc(#loc48)
    %41 = handshake.mux %40 [%falseResult_10, %trueResult] : index, none loc(#loc48)
    %42 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc48)
    %trueResult_11, %falseResult_12 = handshake.cond_br %21, %42 : none loc(#loc59)
    %43 = dataflow.carry %willContinue_5, %trueResult_11, %trueResult_13 : i1, none, none -> none loc(#loc59)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue_5, %36 : none loc(#loc59)
    %44 = handshake.constant %42 {value = 0 : index} : index loc(#loc59)
    %45 = handshake.constant %42 {value = 1 : index} : index loc(#loc59)
    %46 = arith.select %21, %45, %44 : index loc(#loc59)
    %47 = handshake.mux %46 [%falseResult_12, %falseResult_14] : index, none loc(#loc59)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %47 : none loc(#loc48)
    %48 = handshake.mux %40 [%falseResult_16, %trueResult] : index, none loc(#loc48)
    %49 = handshake.join %41, %48 : none, none loc(#loc41)
    handshake.return %49 : none loc(#loc43)
  } loc(#loc41)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.umin.i32(i32, i32) -> i32 loc(#loc2)
  func.func @_Z15unpack_bits_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg2: i32 loc(fused<#di_subprogram5>[#loc22])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c5_i64 = arith.constant 5 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %0 = arith.addi %arg2, %c31_i32 : i32 loc(#loc45)
    %1 = arith.shrui %0, %c5_i32 : i32 loc(#loc45)
    %2 = arith.cmpi eq, %1, %c0_i32 : i32 loc(#loc53)
    scf.if %2 {
    } else {
      %3 = arith.extui %arg2 : i32 to i64 loc(#loc49)
      %4 = arith.extui %1 : i32 to i64 loc(#loc53)
      %5:2 = scf.while (%arg3 = %c0_i64, %arg4 = %c0_i64) : (i64, i64) -> (i64, i64) {
        %6 = arith.index_cast %arg3 : i64 to index loc(#loc56)
        %7 = memref.load %arg0[%6] : memref<?xi32, strided<[1], offset: ?>> loc(#loc56)
        %8 = arith.shli %arg3, %c5_i64 : i64 loc(#loc57)
        %9 = arith.cmpi ult, %8, %3 : i64 loc(#loc64)
        scf.if %9 {
          %13 = arith.trunci %8 : i64 to i32 loc(#loc62)
          %14 = arith.addi %13, %c32_i32 : i32 loc(#loc62)
          %15 = arith.cmpi ult, %14, %arg2 : i32 loc(#loc62)
          %16 = arith.select %15, %14, %arg2 : i32 loc(#loc62)
          %17 = arith.extui %16 : i32 to i64 loc(#loc61)
          %18 = scf.while (%arg5 = %arg4) : (i64) -> i64 {
            %19 = arith.subi %arg5, %8 : i64 loc(#loc67)
            %20 = arith.trunci %19 : i64 to i32 loc(#loc68)
            %21 = arith.shrui %7, %20 : i32 loc(#loc68)
            %22 = arith.andi %21, %c1_i32 : i32 loc(#loc68)
            %23 = arith.index_cast %arg5 : i64 to index loc(#loc68)
            memref.store %22, %arg1[%23] : memref<?xi32, strided<[1], offset: ?>> loc(#loc68)
            %24 = arith.addi %arg5, %c1_i64 : i64 loc(#loc64)
            %25 = arith.cmpi ult, %24, %17 : i64 loc(#loc64)
            scf.condition(%25) %24 : i64 loc(#loc61)
          } do {
          ^bb0(%arg5: i64 loc(fused<#di_lexical_block30>[#loc27])):
            scf.yield %arg5 : i64 loc(#loc61)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc61)
        } loc(#loc61)
        %10 = arith.addi %arg3, %c1_i64 : i64 loc(#loc53)
        %11 = arith.addi %arg4, %c32_i64 : i64 loc(#loc49)
        %12 = arith.cmpi ne, %10, %4 : i64 loc(#loc58)
        scf.condition(%12) %10, %11 : i64, i64 loc(#loc49)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block22>[#loc24]), %arg4: i64 loc(fused<#di_lexical_block22>[#loc24])):
        scf.yield %arg3, %arg4 : i64, i64 loc(#loc49)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc49)
    } loc(#loc49)
    return loc(#loc46)
  } loc(#loc44)
} loc(#loc)
#loc = loc("tests/app/unpack_bits/main.cpp":0:0)
#loc1 = loc("tests/app/unpack_bits/main.cpp":5:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/unpack_bits/main.cpp":12:0)
#loc4 = loc("tests/app/unpack_bits/main.cpp":20:0)
#loc5 = loc("tests/app/unpack_bits/main.cpp":23:0)
#loc7 = loc("tests/app/unpack_bits/main.cpp":27:0)
#loc8 = loc("tests/app/unpack_bits/main.cpp":28:0)
#loc9 = loc("tests/app/unpack_bits/main.cpp":29:0)
#loc10 = loc("tests/app/unpack_bits/main.cpp":33:0)
#loc11 = loc("tests/app/unpack_bits/main.cpp":35:0)
#loc13 = loc("tests/app/unpack_bits/unpack_bits.cpp":35:0)
#loc14 = loc("tests/app/unpack_bits/unpack_bits.cpp":37:0)
#loc15 = loc("tests/app/unpack_bits/unpack_bits.cpp":38:0)
#loc16 = loc("tests/app/unpack_bits/unpack_bits.cpp":39:0)
#loc17 = loc("tests/app/unpack_bits/unpack_bits.cpp":45:0)
#loc18 = loc("tests/app/unpack_bits/unpack_bits.cpp":41:0)
#loc19 = loc("tests/app/unpack_bits/unpack_bits.cpp":46:0)
#loc20 = loc("tests/app/unpack_bits/unpack_bits.cpp":47:0)
#loc21 = loc("tests/app/unpack_bits/unpack_bits.cpp":50:0)
#loc23 = loc("tests/app/unpack_bits/unpack_bits.cpp":12:0)
#loc25 = loc("tests/app/unpack_bits/unpack_bits.cpp":15:0)
#loc26 = loc("tests/app/unpack_bits/unpack_bits.cpp":16:0)
#loc28 = loc("tests/app/unpack_bits/unpack_bits.cpp":18:0)
#loc29 = loc("tests/app/unpack_bits/unpack_bits.cpp":23:0)
#loc30 = loc("tests/app/unpack_bits/unpack_bits.cpp":24:0)
#loc31 = loc("tests/app/unpack_bits/unpack_bits.cpp":27:0)
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 11>
#loc32 = loc(fused<#di_subprogram3>[#loc1])
#loc33 = loc(fused<#di_subprogram3>[#loc4])
#loc34 = loc(fused<#di_subprogram3>[#loc5])
#loc35 = loc(fused<#di_subprogram3>[#loc10])
#loc36 = loc(fused<#di_subprogram3>[#loc11])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 11>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 26>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 11>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 26>
#loc38 = loc(fused<#di_lexical_block17>[#loc6])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 27>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 37>
#loc39 = loc(fused<#di_lexical_block18>[#loc3])
#loc40 = loc(fused[#loc37, #loc38])
#loc42 = loc(fused<#di_subprogram4>[#loc13])
#loc43 = loc(fused<#di_subprogram4>[#loc21])
#loc45 = loc(fused<#di_subprogram5>[#loc23])
#loc46 = loc(fused<#di_subprogram5>[#loc31])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 27>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 37>
#loc47 = loc(fused<#di_lexical_block20>[#loc7])
#loc48 = loc(fused<#di_lexical_block21>[#loc14])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 37>
#loc50 = loc(fused<#di_lexical_block23>[#loc8])
#loc51 = loc(fused<#di_lexical_block23>[#loc9])
#loc52 = loc(fused<#di_lexical_block24>[#loc14])
#loc53 = loc(fused<#di_lexical_block25>[#loc24])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 45>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 41>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 18>
#loc54 = loc(fused<#di_lexical_block26>[#loc15])
#loc55 = loc(fused<#di_lexical_block26>[#loc16])
#loc56 = loc(fused<#di_lexical_block27>[#loc25])
#loc57 = loc(fused<#di_lexical_block27>[#loc26])
#loc58 = loc(fused[#loc49, #loc53])
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file1, line = 45>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 22>
#loc59 = loc(fused<#di_lexical_block28>[#loc17])
#loc60 = loc(fused<#di_lexical_block29>[#loc18])
#loc62 = loc(fused<#di_lexical_block31>[#loc28])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file1, line = 45>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file1, line = 22>
#loc63 = loc(fused<#di_lexical_block32>[#loc17])
#loc64 = loc(fused<#di_lexical_block33>[#loc27])
#loc65 = loc(fused<#di_lexical_block34>[#loc19])
#loc66 = loc(fused<#di_lexical_block34>[#loc20])
#loc67 = loc(fused<#di_lexical_block35>[#loc29])
#loc68 = loc(fused<#di_lexical_block35>[#loc30])
