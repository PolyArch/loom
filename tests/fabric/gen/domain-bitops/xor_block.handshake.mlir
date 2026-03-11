#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/xor_block/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/xor_block/xor_block.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/xor_block/main.cpp":17:0)
#loc8 = loc("tests/app/xor_block/main.cpp":29:0)
#loc14 = loc("tests/app/xor_block/xor_block.cpp":19:0)
#loc18 = loc("tests/app/xor_block/xor_block.cpp":8:0)
#loc19 = loc("tests/app/xor_block/xor_block.cpp":12:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 25>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 12>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 29, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 25, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 12, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_A", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "input_B", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 22, arg = 4, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 11, arg = 4, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_C", file = #di_file1, line = 21, arg = 3, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_C", file = #di_file1, line = 10, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 29>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_A", file = #di_file1, line = 19, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_B", file = #di_file1, line = 20, arg = 2, type = #di_derived_type6>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_A", file = #di_file1, line = 8, arg = 1, type = #di_derived_type6>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_B", file = #di_file1, line = 9, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "xor_block_dsa", linkageName = "_Z13xor_block_dsaPKjS0_Pjj", file = #di_file1, line = 19, scopeLine = 22, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable14, #di_local_variable11, #di_local_variable9, #di_local_variable2>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "xor_block_cpu", linkageName = "_Z13xor_block_cpuPKjS0_Pjj", file = #di_file1, line = 8, scopeLine = 11, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable15, #di_local_variable16, #di_local_variable12, #di_local_variable10, #di_local_variable3>
#loc27 = loc(fused<#di_lexical_block4>[#loc3])
#loc28 = loc(fused<#di_lexical_block5>[#loc8])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 12>
#loc31 = loc(fused<#di_subprogram4>[#loc14])
#loc33 = loc(fused<#di_subprogram5>[#loc18])
#loc40 = loc(fused<#di_lexical_block11>[#loc19])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<18xi8> = dense<[120, 111, 114, 95, 98, 108, 111, 99, 107, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<18xi8> = dense<[120, 111, 114, 95, 98, 108, 111, 99, 107, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<34xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 120, 111, 114, 95, 98, 108, 111, 99, 107, 47, 120, 111, 114, 95, 98, 108, 111, 99, 107, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc22)
    %false = arith.constant false loc(#loc22)
    %0 = seq.const_clock  low loc(#loc22)
    %c2_i32 = arith.constant 2 : i32 loc(#loc22)
    %1 = ub.poison : i64 loc(#loc22)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c305419896_i32 = arith.constant 305419896 : i32 loc(#loc2)
    %c-1412567295_i32 = arith.constant -1412567295 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<18xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<18xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc35)
      %11 = arith.muli %10, %c305419896_i32 : i32 loc(#loc35)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc35)
      memref.store %11, %alloca[%12] : memref<1024xi32> loc(#loc35)
      %13 = arith.muli %10, %c-1412567295_i32 : i32 loc(#loc36)
      memref.store %13, %alloca_0[%12] : memref<1024xi32> loc(#loc36)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc29)
      %15 = arith.cmpi ne, %14, %c1024_i64 : i64 loc(#loc37)
      scf.condition(%15) %14 : i64 loc(#loc27)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc27)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc27)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %cast_4 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    call @_Z13xor_block_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc23)
    %cast_5 = memref.cast %alloca_2 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc24)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc24)
    %5 = handshake.esi_instance @_Z13xor_block_dsaPKjS0_Pjj_esi "_Z13xor_block_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc24)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc24)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      %11 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc41)
      %12 = memref.load %alloca_2[%10] : memref<1024xi32> loc(#loc41)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc41)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc30)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc30)
        %18 = arith.extui %17 : i1 to i32 loc(#loc28)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc38)
        %20 = arith.extui %19 : i1 to i32 loc(#loc28)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc41)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc41)
      } loc(#loc41)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc28)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc28)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc8]), %arg1: i1 loc(fused<#di_lexical_block5>[#loc8]), %arg2: i32 loc(fused<#di_lexical_block5>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc28)
    } loc(#loc28)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc28)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc28)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<18xi8> -> index loc(#loc44)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc44)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc44)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc44)
      scf.yield %c1_i32 : i32 loc(#loc45)
    } loc(#loc28)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<18xi8> -> index loc(#loc25)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc25)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc25)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc25)
    } loc(#loc2)
    return %9 : i32 loc(#loc26)
  } loc(#loc22)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z13xor_block_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: i1 loc(fused<#di_subprogram4>[#loc14]), ...) -> i1 attributes {argNames = ["input_A", "input_B", "output_C", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc31)
    %1 = handshake.join %0 : none loc(#loc31)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc42)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc39)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc39)
    %6 = arith.index_cast %3 : i64 to index loc(#loc39)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc39)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc39)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc39)
    %dataResult, %addressResults = handshake.load [%afterValue] %9#0, %12 : index, i32 loc(#loc46)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %10#0, %19 : index, i32 loc(#loc46)
    %8 = arith.xori %dataResult_0, %dataResult : i32 loc(#loc46)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %8, %17 : index, i32 loc(#loc46)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc31)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc31)
    %11 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc31)
    %12 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc39)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %9#1 : none loc(#loc39)
    %13 = handshake.constant %1 {value = 0 : index} : index loc(#loc39)
    %14 = handshake.constant %1 {value = 1 : index} : index loc(#loc39)
    %15 = arith.select %4, %14, %13 : index loc(#loc39)
    %16 = handshake.mux %15 [%falseResult_4, %trueResult] : index, none loc(#loc39)
    %17 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc39)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11 : none loc(#loc39)
    %18 = handshake.mux %15 [%falseResult_6, %trueResult] : index, none loc(#loc39)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc39)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %10#1 : none loc(#loc39)
    %20 = handshake.mux %15 [%falseResult_8, %trueResult] : index, none loc(#loc39)
    %21 = handshake.join %16, %18, %20 : none, none, none loc(#loc31)
    %22 = handshake.constant %21 {value = true} : i1 loc(#loc31)
    handshake.return %22 : i1 loc(#loc31)
  } loc(#loc31)
  handshake.func @_Z13xor_block_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc14]), %arg3: i32 loc(fused<#di_subprogram4>[#loc14]), %arg4: none loc(fused<#di_subprogram4>[#loc14]), ...) -> none attributes {argNames = ["input_A", "input_B", "output_C", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc31)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc42)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc39)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc39)
    %5 = arith.index_cast %2 : i64 to index loc(#loc39)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc39)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc39)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc39)
    %dataResult, %addressResults = handshake.load [%afterValue] %8#0, %11 : index, i32 loc(#loc46)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %9#0, %18 : index, i32 loc(#loc46)
    %7 = arith.xori %dataResult_0, %dataResult : i32 loc(#loc46)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %7, %16 : index, i32 loc(#loc46)
    %8:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc31)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (i32, none) loc(#loc31)
    %10 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc31)
    %11 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc39)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %8#1 : none loc(#loc39)
    %12 = handshake.constant %0 {value = 0 : index} : index loc(#loc39)
    %13 = handshake.constant %0 {value = 1 : index} : index loc(#loc39)
    %14 = arith.select %3, %13, %12 : index loc(#loc39)
    %15 = handshake.mux %14 [%falseResult_4, %trueResult] : index, none loc(#loc39)
    %16 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc39)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %10 : none loc(#loc39)
    %17 = handshake.mux %14 [%falseResult_6, %trueResult] : index, none loc(#loc39)
    %18 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc39)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %9#1 : none loc(#loc39)
    %19 = handshake.mux %14 [%falseResult_8, %trueResult] : index, none loc(#loc39)
    %20 = handshake.join %15, %17, %19 : none, none, none loc(#loc31)
    handshake.return %20 : none loc(#loc32)
  } loc(#loc31)
  func.func @_Z13xor_block_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc18]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc18]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc18]), %arg3: i32 loc(fused<#di_subprogram5>[#loc18])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc43)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc43)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc47)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc47)
        %5 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc47)
        %6 = arith.xori %5, %4 : i32 loc(#loc47)
        memref.store %6, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc47)
        %7 = arith.addi %arg4, %c1_i64 : i64 loc(#loc43)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc48)
        scf.condition(%8) %7 : i64 loc(#loc40)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block11>[#loc19])):
        scf.yield %arg4 : i64 loc(#loc40)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc40)
    } loc(#loc40)
    return loc(#loc34)
  } loc(#loc33)
} loc(#loc)
#loc = loc("tests/app/xor_block/main.cpp":0:0)
#loc1 = loc("tests/app/xor_block/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/xor_block/main.cpp":18:0)
#loc5 = loc("tests/app/xor_block/main.cpp":19:0)
#loc6 = loc("tests/app/xor_block/main.cpp":23:0)
#loc7 = loc("tests/app/xor_block/main.cpp":26:0)
#loc9 = loc("tests/app/xor_block/main.cpp":30:0)
#loc10 = loc("tests/app/xor_block/main.cpp":31:0)
#loc11 = loc("tests/app/xor_block/main.cpp":32:0)
#loc12 = loc("tests/app/xor_block/main.cpp":36:0)
#loc13 = loc("tests/app/xor_block/main.cpp":38:0)
#loc15 = loc("tests/app/xor_block/xor_block.cpp":25:0)
#loc16 = loc("tests/app/xor_block/xor_block.cpp":26:0)
#loc17 = loc("tests/app/xor_block/xor_block.cpp":28:0)
#loc20 = loc("tests/app/xor_block/xor_block.cpp":13:0)
#loc21 = loc("tests/app/xor_block/xor_block.cpp":15:0)
#loc22 = loc(fused<#di_subprogram3>[#loc1])
#loc23 = loc(fused<#di_subprogram3>[#loc6])
#loc24 = loc(fused<#di_subprogram3>[#loc7])
#loc25 = loc(fused<#di_subprogram3>[#loc12])
#loc26 = loc(fused<#di_subprogram3>[#loc13])
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 29>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 17>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 29>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 25>
#loc29 = loc(fused<#di_lexical_block6>[#loc3])
#loc30 = loc(fused<#di_lexical_block7>[#loc8])
#loc32 = loc(fused<#di_subprogram4>[#loc17])
#loc34 = loc(fused<#di_subprogram5>[#loc21])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 30>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 25>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 12>
#loc35 = loc(fused<#di_lexical_block8>[#loc4])
#loc36 = loc(fused<#di_lexical_block8>[#loc5])
#loc37 = loc(fused[#loc27, #loc29])
#loc38 = loc(fused[#loc28, #loc30])
#loc39 = loc(fused<#di_lexical_block10>[#loc15])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 30>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 25>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 12>
#loc41 = loc(fused<#di_lexical_block12>[#loc9])
#loc42 = loc(fused<#di_lexical_block13>[#loc15])
#loc43 = loc(fused<#di_lexical_block14>[#loc19])
#loc44 = loc(fused<#di_lexical_block15>[#loc10])
#loc45 = loc(fused<#di_lexical_block15>[#loc11])
#loc46 = loc(fused<#di_lexical_block16>[#loc16])
#loc47 = loc(fused<#di_lexical_block17>[#loc20])
#loc48 = loc(fused[#loc40, #loc43])
