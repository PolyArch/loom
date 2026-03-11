#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/rotate_bits/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/rotate_bits/rotate_bits.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/rotate_bits/main.cpp":17:0)
#loc8 = loc("tests/app/rotate_bits/main.cpp":29:0)
#loc14 = loc("tests/app/rotate_bits/rotate_bits.cpp":27:0)
#loc20 = loc("tests/app/rotate_bits/rotate_bits.cpp":13:0)
#loc21 = loc("tests/app/rotate_bits/rotate_bits.cpp":17:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 33>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 33>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 17>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 33>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 17>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 29, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 33, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_data", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_shift", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 30, arg = 4, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file1, line = 34, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "shift", file = #di_file1, line = 35, type = #di_derived_type1>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 16, arg = 4, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file1, line = 18, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "shift", file = #di_file1, line = 19, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_result", file = #di_file1, line = 29, arg = 3, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_result", file = #di_file1, line = 15, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 29>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file1, line = 27, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_shift", file = #di_file1, line = 28, arg = 2, type = #di_derived_type6>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_data", file = #di_file1, line = 13, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_shift", file = #di_file1, line = 14, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "rotate_bits_dsa", linkageName = "_Z15rotate_bits_dsaPKjS0_Pjj", file = #di_file1, line = 27, scopeLine = 30, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable15, #di_local_variable9, #di_local_variable2, #di_local_variable10, #di_local_variable11>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "rotate_bits_cpu", linkageName = "_Z15rotate_bits_cpuPKjS0_Pjj", file = #di_file1, line = 13, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable16, #di_local_variable12, #di_local_variable3, #di_local_variable13, #di_local_variable14>
#loc31 = loc(fused<#di_lexical_block8>[#loc3])
#loc32 = loc(fused<#di_lexical_block9>[#loc8])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 17>
#loc35 = loc(fused<#di_subprogram4>[#loc14])
#loc37 = loc(fused<#di_subprogram5>[#loc20])
#loc44 = loc(fused<#di_lexical_block15>[#loc21])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[114, 111, 116, 97, 116, 101, 95, 98, 105, 116, 115, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[114, 111, 116, 97, 116, 101, 95, 98, 105, 116, 115, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 114, 111, 116, 97, 116, 101, 95, 98, 105, 116, 115, 47, 114, 111, 116, 97, 116, 101, 95, 98, 105, 116, 115, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc26)
    %false = arith.constant false loc(#loc26)
    %0 = seq.const_clock  low loc(#loc26)
    %c2_i32 = arith.constant 2 : i32 loc(#loc26)
    %1 = ub.poison : i64 loc(#loc26)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c305419896_i32 = arith.constant 305419896 : i32 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc39)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc39)
      %12 = arith.addi %11, %c305419896_i32 : i32 loc(#loc39)
      memref.store %12, %alloca[%10] : memref<1024xi32> loc(#loc39)
      %13 = arith.andi %11, %c31_i32 : i32 loc(#loc40)
      memref.store %13, %alloca_0[%10] : memref<1024xi32> loc(#loc40)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc33)
      %15 = arith.cmpi ne, %14, %c1024_i64 : i64 loc(#loc41)
      scf.condition(%15) %14 : i64 loc(#loc31)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc31)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc31)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc27)
    call @_Z15rotate_bits_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc27)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc28)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc28)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc28)
    %5 = handshake.esi_instance @_Z15rotate_bits_dsaPKjS0_Pjj_esi "_Z15rotate_bits_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc28)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc28)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc45)
      %11 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc45)
      %12 = memref.load %alloca_2[%10] : memref<1024xi32> loc(#loc45)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc45)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc34)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc34)
        %18 = arith.extui %17 : i1 to i32 loc(#loc32)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc42)
        %20 = arith.extui %19 : i1 to i32 loc(#loc32)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc45)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc45)
      } loc(#loc45)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc32)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc32)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc8]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc8]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc32)
    } loc(#loc32)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc32)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc32)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc48)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc48)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc48)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc48)
      scf.yield %c1_i32 : i32 loc(#loc49)
    } loc(#loc32)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc29)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc29)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc29)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc29)
    } loc(#loc2)
    return %9 : i32 loc(#loc30)
  } loc(#loc26)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z15rotate_bits_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i1 loc(fused<#di_subprogram4>[#loc14]), ...) -> i1 attributes {argNames = ["input_data", "input_shift", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc35)
    %1 = handshake.join %0 : none loc(#loc35)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 32 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 31 : i32} : i32 loc(#loc2)
    %6 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc46)
    %trueResult, %falseResult = handshake.cond_br %6, %1 : none loc(#loc43)
    %7 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc43)
    %8 = arith.index_cast %4 : i64 to index loc(#loc43)
    %9 = arith.index_cast %arg3 : i32 to index loc(#loc43)
    %index, %willContinue = dataflow.stream %8, %7, %9 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %25#0, %27 : index, i32 loc(#loc50)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %24#0, %34 : index, i32 loc(#loc51)
    %11 = arith.andi %dataResult_0, %5 : i32 loc(#loc51)
    %12 = arith.cmpi eq, %11, %2 : i32 loc(#loc52)
    %13 = arith.remui %dataResult_0, %3 : i32 loc(#loc52)
    %14 = arith.cmpi eq, %13, %2 : i32 loc(#loc52)
    %15 = arith.shli %dataResult, %13 : i32 loc(#loc52)
    %16 = arith.subi %3, %13 : i32 loc(#loc52)
    %17 = arith.shrui %dataResult, %16 : i32 loc(#loc52)
    %18 = arith.ori %15, %17 : i32 loc(#loc52)
    %19 = handshake.constant %10 {value = 0 : index} : index loc(#loc52)
    %20 = handshake.constant %10 {value = 1 : index} : index loc(#loc52)
    %21 = arith.select %14, %20, %19 : index loc(#loc52)
    %22 = handshake.mux %21 [%18, %dataResult] : index, i32 loc(#loc52)
    %23 = arith.select %12, %dataResult, %22 : i32 loc(#loc52)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %23, %32 : index, i32 loc(#loc52)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc35)
    %25:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc35)
    %26 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc35)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc43)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %25#1 : none loc(#loc43)
    %28 = handshake.constant %1 {value = 0 : index} : index loc(#loc43)
    %29 = handshake.constant %1 {value = 1 : index} : index loc(#loc43)
    %30 = arith.select %6, %29, %28 : index loc(#loc43)
    %31 = handshake.mux %30 [%falseResult_4, %trueResult] : index, none loc(#loc43)
    %32 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc43)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %26 : none loc(#loc43)
    %33 = handshake.mux %30 [%falseResult_6, %trueResult] : index, none loc(#loc43)
    %34 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc43)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %24#1 : none loc(#loc43)
    %35 = handshake.mux %30 [%falseResult_8, %trueResult] : index, none loc(#loc43)
    %36 = handshake.join %31, %33, %35 : none, none, none loc(#loc35)
    %37 = handshake.constant %36 {value = true} : i1 loc(#loc35)
    handshake.return %37 : i1 loc(#loc35)
  } loc(#loc35)
  handshake.func @_Z15rotate_bits_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: none loc(fused<#di_subprogram4>[#loc14]), ...) -> none attributes {argNames = ["input_data", "input_shift", "output_result", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc35)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 32 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 31 : i32} : i32 loc(#loc2)
    %5 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc46)
    %trueResult, %falseResult = handshake.cond_br %5, %0 : none loc(#loc43)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc43)
    %7 = arith.index_cast %3 : i64 to index loc(#loc43)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc43)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %24#0, %26 : index, i32 loc(#loc50)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %23#0, %33 : index, i32 loc(#loc51)
    %10 = arith.andi %dataResult_0, %4 : i32 loc(#loc51)
    %11 = arith.cmpi eq, %10, %1 : i32 loc(#loc52)
    %12 = arith.remui %dataResult_0, %2 : i32 loc(#loc52)
    %13 = arith.cmpi eq, %12, %1 : i32 loc(#loc52)
    %14 = arith.shli %dataResult, %12 : i32 loc(#loc52)
    %15 = arith.subi %2, %12 : i32 loc(#loc52)
    %16 = arith.shrui %dataResult, %15 : i32 loc(#loc52)
    %17 = arith.ori %14, %16 : i32 loc(#loc52)
    %18 = handshake.constant %9 {value = 0 : index} : index loc(#loc52)
    %19 = handshake.constant %9 {value = 1 : index} : index loc(#loc52)
    %20 = arith.select %13, %19, %18 : index loc(#loc52)
    %21 = handshake.mux %20 [%17, %dataResult] : index, i32 loc(#loc52)
    %22 = arith.select %11, %dataResult, %21 : i32 loc(#loc52)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %22, %31 : index, i32 loc(#loc52)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc35)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc35)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc35)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc43)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %24#1 : none loc(#loc43)
    %27 = handshake.constant %0 {value = 0 : index} : index loc(#loc43)
    %28 = handshake.constant %0 {value = 1 : index} : index loc(#loc43)
    %29 = arith.select %5, %28, %27 : index loc(#loc43)
    %30 = handshake.mux %29 [%falseResult_4, %trueResult] : index, none loc(#loc43)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc43)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %25 : none loc(#loc43)
    %32 = handshake.mux %29 [%falseResult_6, %trueResult] : index, none loc(#loc43)
    %33 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc43)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %23#1 : none loc(#loc43)
    %34 = handshake.mux %29 [%falseResult_8, %trueResult] : index, none loc(#loc43)
    %35 = handshake.join %30, %32, %34 : none, none, none loc(#loc35)
    handshake.return %35 : none loc(#loc36)
  } loc(#loc35)
  func.func private @llvm.fshl.i32(i32, i32, i32) -> i32 loc(#loc2)
  func.func @_Z15rotate_bits_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg3: i32 loc(fused<#di_subprogram5>[#loc20])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c31_i32 = arith.constant 31 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc47)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc47)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc53)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc53)
        %5 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc54)
        %6 = arith.andi %5, %c31_i32 : i32 loc(#loc54)
        %7 = arith.cmpi eq, %6, %c0_i32 : i32 loc(#loc55)
        %8 = arith.remui %5, %c32_i32 : i32 loc(#loc55)
        %9 = arith.cmpi eq, %8, %c0_i32 : i32 loc(#loc55)
        %10 = scf.if %9 -> (i32) {
          scf.yield %4 : i32 loc(#loc55)
        } else {
          %14 = arith.shli %4, %8 : i32 loc(#loc55)
          %15 = arith.subi %c32_i32, %8 : i32 loc(#loc55)
          %16 = arith.shrui %4, %15 : i32 loc(#loc55)
          %17 = arith.ori %14, %16 : i32 loc(#loc55)
          scf.yield %17 : i32 loc(#loc55)
        } loc(#loc55)
        %11 = arith.select %7, %4, %10 : i32 loc(#loc55)
        memref.store %11, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc55)
        %12 = arith.addi %arg4, %c1_i64 : i64 loc(#loc47)
        %13 = arith.cmpi ne, %12, %1 : i64 loc(#loc56)
        scf.condition(%13) %12 : i64 loc(#loc44)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block15>[#loc21])):
        scf.yield %arg4 : i64 loc(#loc44)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc44)
    } loc(#loc44)
    return loc(#loc38)
  } loc(#loc37)
} loc(#loc)
#loc = loc("tests/app/rotate_bits/main.cpp":0:0)
#loc1 = loc("tests/app/rotate_bits/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/rotate_bits/main.cpp":18:0)
#loc5 = loc("tests/app/rotate_bits/main.cpp":19:0)
#loc6 = loc("tests/app/rotate_bits/main.cpp":23:0)
#loc7 = loc("tests/app/rotate_bits/main.cpp":26:0)
#loc9 = loc("tests/app/rotate_bits/main.cpp":30:0)
#loc10 = loc("tests/app/rotate_bits/main.cpp":31:0)
#loc11 = loc("tests/app/rotate_bits/main.cpp":32:0)
#loc12 = loc("tests/app/rotate_bits/main.cpp":36:0)
#loc13 = loc("tests/app/rotate_bits/main.cpp":38:0)
#loc15 = loc("tests/app/rotate_bits/rotate_bits.cpp":33:0)
#loc16 = loc("tests/app/rotate_bits/rotate_bits.cpp":34:0)
#loc17 = loc("tests/app/rotate_bits/rotate_bits.cpp":35:0)
#loc18 = loc("tests/app/rotate_bits/rotate_bits.cpp":37:0)
#loc19 = loc("tests/app/rotate_bits/rotate_bits.cpp":39:0)
#loc22 = loc("tests/app/rotate_bits/rotate_bits.cpp":18:0)
#loc23 = loc("tests/app/rotate_bits/rotate_bits.cpp":19:0)
#loc24 = loc("tests/app/rotate_bits/rotate_bits.cpp":21:0)
#loc25 = loc("tests/app/rotate_bits/rotate_bits.cpp":23:0)
#loc26 = loc(fused<#di_subprogram3>[#loc1])
#loc27 = loc(fused<#di_subprogram3>[#loc6])
#loc28 = loc(fused<#di_subprogram3>[#loc7])
#loc29 = loc(fused<#di_subprogram3>[#loc12])
#loc30 = loc(fused<#di_subprogram3>[#loc13])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 17>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 29>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 17>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 29>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 33>
#loc33 = loc(fused<#di_lexical_block10>[#loc3])
#loc34 = loc(fused<#di_lexical_block11>[#loc8])
#loc36 = loc(fused<#di_subprogram4>[#loc19])
#loc38 = loc(fused<#di_subprogram5>[#loc25])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 30>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 33>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 17>
#loc39 = loc(fused<#di_lexical_block12>[#loc4])
#loc40 = loc(fused<#di_lexical_block12>[#loc5])
#loc41 = loc(fused[#loc31, #loc33])
#loc42 = loc(fused[#loc32, #loc34])
#loc43 = loc(fused<#di_lexical_block14>[#loc15])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 30>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 33>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 17>
#loc45 = loc(fused<#di_lexical_block16>[#loc9])
#loc46 = loc(fused<#di_lexical_block17>[#loc15])
#loc47 = loc(fused<#di_lexical_block18>[#loc21])
#loc48 = loc(fused<#di_lexical_block19>[#loc10])
#loc49 = loc(fused<#di_lexical_block19>[#loc11])
#loc50 = loc(fused<#di_lexical_block20>[#loc16])
#loc51 = loc(fused<#di_lexical_block20>[#loc17])
#loc52 = loc(fused<#di_lexical_block20>[#loc18])
#loc53 = loc(fused<#di_lexical_block21>[#loc22])
#loc54 = loc(fused<#di_lexical_block21>[#loc23])
#loc55 = loc(fused<#di_lexical_block21>[#loc24])
#loc56 = loc(fused[#loc44, #loc47])
