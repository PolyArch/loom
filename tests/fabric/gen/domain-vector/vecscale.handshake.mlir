#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/vecscale/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/vecscale/vecscale.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/vecscale/main.cpp":17:0)
#loc7 = loc("tests/app/vecscale/main.cpp":28:0)
#loc13 = loc("tests/app/vecscale/vecscale.cpp":23:0)
#loc17 = loc("tests/app/vecscale/vecscale.cpp":11:0)
#loc18 = loc("tests/app/vecscale/vecscale.cpp":15:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 28>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 29>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 15>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 28, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 29, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 15, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "alpha", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_B", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_B", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "alpha", file = #di_file1, line = 24, arg = 2, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 26, arg = 4, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "alpha", file = #di_file1, line = 12, arg = 2, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 14, arg = 4, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "B", file = #di_file1, line = 25, arg = 3, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "B", file = #di_file1, line = 13, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 28>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file1, line = 23, arg = 1, type = #di_derived_type6>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 11, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type2, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecscale_dsa", linkageName = "_Z12vecscale_dsaPKjjPjj", file = #di_file1, line = 23, scopeLine = 26, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable15, #di_local_variable9, #di_local_variable13, #di_local_variable10, #di_local_variable2>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecscale_cpu", linkageName = "_Z12vecscale_cpuPKjjPjj", file = #di_file1, line = 11, scopeLine = 14, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable11, #di_local_variable14, #di_local_variable12, #di_local_variable3>
#loc26 = loc(fused<#di_lexical_block4>[#loc3])
#loc27 = loc(fused<#di_lexical_block5>[#loc7])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 15>
#loc30 = loc(fused<#di_subprogram4>[#loc13])
#loc32 = loc(fused<#di_subprogram5>[#loc17])
#loc38 = loc(fused<#di_lexical_block11>[#loc18])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<17xi8> = dense<[118, 101, 99, 115, 99, 97, 108, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<17xi8> = dense<[118, 101, 99, 115, 99, 97, 108, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<32xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 118, 101, 99, 115, 99, 97, 108, 101, 47, 118, 101, 99, 115, 99, 97, 108, 101, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc21)
    %false = arith.constant false loc(#loc21)
    %0 = seq.const_clock  low loc(#loc21)
    %c2_i32 = arith.constant 2 : i32 loc(#loc21)
    %1 = ub.poison : i64 loc(#loc21)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<17xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<17xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc34)
      %11 = arith.remui %10, %c100_i32 : i32 loc(#loc34)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc34)
      memref.store %11, %alloca[%12] : memref<1024xi32> loc(#loc34)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc28)
      %14 = arith.cmpi ne, %13, %c1024_i64 : i64 loc(#loc35)
      scf.condition(%14) %13 : i64 loc(#loc26)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc26)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc26)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc22)
    %cast_2 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc22)
    call @_Z12vecscale_cpuPKjjPjj(%cast, %c7_i32, %cast_2, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, i32, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc22)
    %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput_4, %ready_5 = esi.wrap.vr %c7_i32, %true : i32 loc(#loc23)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput_8, %ready_9 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc23)
    %chanOutput_10, %ready_11 = esi.wrap.vr %true, %true : i1 loc(#loc23)
    %5 = handshake.esi_instance @_Z12vecscale_dsaPKjjPjj_esi "_Z12vecscale_dsaPKjjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8, %chanOutput_10) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc23)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc23)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc39)
      %11 = memref.load %alloca_0[%10] : memref<1024xi32> loc(#loc39)
      %12 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc39)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc39)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc29)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc29)
        %18 = arith.extui %17 : i1 to i32 loc(#loc27)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc36)
        %20 = arith.extui %19 : i1 to i32 loc(#loc27)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc39)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc39)
      } loc(#loc39)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc27)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc27)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc7]), %arg1: i1 loc(fused<#di_lexical_block5>[#loc7]), %arg2: i32 loc(fused<#di_lexical_block5>[#loc7])):
      scf.yield %arg0 : i64 loc(#loc27)
    } loc(#loc27)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc27)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc27)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<17xi8> -> index loc(#loc42)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc42)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc42)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc42)
      scf.yield %c1_i32 : i32 loc(#loc43)
    } loc(#loc27)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<17xi8> -> index loc(#loc24)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc24)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc24)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc24)
    } loc(#loc2)
    return %9 : i32 loc(#loc25)
  } loc(#loc21)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z12vecscale_dsaPKjjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: i32 loc(fused<#di_subprogram4>[#loc13]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg3: i32 loc(fused<#di_subprogram4>[#loc13]), %arg4: i1 loc(fused<#di_subprogram4>[#loc13]), ...) -> i1 attributes {argNames = ["A", "alpha", "B", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc30)
    %1 = handshake.join %0 : none loc(#loc30)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc40)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc37)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc37)
    %6 = arith.index_cast %3 : i64 to index loc(#loc37)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc37)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=1 max=1024"], step_op = "+=", stop_cond = "!="} loc(#loc37)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc37)
    %dataResult, %addressResults = handshake.load [%afterValue] %9#0, %11 : index, i32 loc(#loc44)
    %8 = arith.muli %dataResult, %arg1 : i32 loc(#loc44)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %8, %16 : index, i32 loc(#loc44)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc30)
    %10 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc30)
    %11 = dataflow.carry %willContinue, %falseResult, %trueResult_1 : i1, none, none -> none loc(#loc37)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %9#1 : none loc(#loc37)
    %12 = handshake.constant %1 {value = 0 : index} : index loc(#loc37)
    %13 = handshake.constant %1 {value = 1 : index} : index loc(#loc37)
    %14 = arith.select %4, %13, %12 : index loc(#loc37)
    %15 = handshake.mux %14 [%falseResult_2, %trueResult] : index, none loc(#loc37)
    %16 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc37)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %10 : none loc(#loc37)
    %17 = handshake.mux %14 [%falseResult_4, %trueResult] : index, none loc(#loc37)
    %18 = handshake.join %15, %17 : none, none loc(#loc30)
    %19 = handshake.constant %18 {value = true} : i1 loc(#loc30)
    handshake.return %19 : i1 loc(#loc30)
  } loc(#loc30)
  handshake.func @_Z12vecscale_dsaPKjjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: i32 loc(fused<#di_subprogram4>[#loc13]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg3: i32 loc(fused<#di_subprogram4>[#loc13]), %arg4: none loc(fused<#di_subprogram4>[#loc13]), ...) -> none attributes {argNames = ["A", "alpha", "B", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc30)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc40)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc37)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc37)
    %5 = arith.index_cast %2 : i64 to index loc(#loc37)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc37)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=1 max=1024"], step_op = "+=", stop_cond = "!="} loc(#loc37)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc37)
    %dataResult, %addressResults = handshake.load [%afterValue] %8#0, %10 : index, i32 loc(#loc44)
    %7 = arith.muli %dataResult, %arg1 : i32 loc(#loc44)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %7, %15 : index, i32 loc(#loc44)
    %8:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc30)
    %9 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc30)
    %10 = dataflow.carry %willContinue, %falseResult, %trueResult_1 : i1, none, none -> none loc(#loc37)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %8#1 : none loc(#loc37)
    %11 = handshake.constant %0 {value = 0 : index} : index loc(#loc37)
    %12 = handshake.constant %0 {value = 1 : index} : index loc(#loc37)
    %13 = arith.select %3, %12, %11 : index loc(#loc37)
    %14 = handshake.mux %13 [%falseResult_2, %trueResult] : index, none loc(#loc37)
    %15 = dataflow.carry %willContinue, %falseResult, %trueResult_3 : i1, none, none -> none loc(#loc37)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %9 : none loc(#loc37)
    %16 = handshake.mux %13 [%falseResult_4, %trueResult] : index, none loc(#loc37)
    %17 = handshake.join %14, %16 : none, none loc(#loc30)
    handshake.return %17 : none loc(#loc31)
  } loc(#loc30)
  func.func @_Z12vecscale_cpuPKjjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc17]), %arg1: i32 loc(fused<#di_subprogram5>[#loc17]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc17]), %arg3: i32 loc(fused<#di_subprogram5>[#loc17])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc41)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc41)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc45)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
        %5 = arith.muli %4, %arg1 : i32 loc(#loc45)
        memref.store %5, %arg2[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc45)
        %6 = arith.addi %arg4, %c1_i64 : i64 loc(#loc41)
        %7 = arith.cmpi ne, %6, %1 : i64 loc(#loc46)
        scf.condition(%7) %6 : i64 loc(#loc38)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block11>[#loc18])):
        scf.yield %arg4 : i64 loc(#loc38)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc38)
    } loc(#loc38)
    return loc(#loc33)
  } loc(#loc32)
} loc(#loc)
#loc = loc("tests/app/vecscale/main.cpp":0:0)
#loc1 = loc("tests/app/vecscale/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/vecscale/main.cpp":18:0)
#loc5 = loc("tests/app/vecscale/main.cpp":22:0)
#loc6 = loc("tests/app/vecscale/main.cpp":25:0)
#loc8 = loc("tests/app/vecscale/main.cpp":29:0)
#loc9 = loc("tests/app/vecscale/main.cpp":30:0)
#loc10 = loc("tests/app/vecscale/main.cpp":31:0)
#loc11 = loc("tests/app/vecscale/main.cpp":35:0)
#loc12 = loc("tests/app/vecscale/main.cpp":37:0)
#loc14 = loc("tests/app/vecscale/vecscale.cpp":29:0)
#loc15 = loc("tests/app/vecscale/vecscale.cpp":30:0)
#loc16 = loc("tests/app/vecscale/vecscale.cpp":32:0)
#loc19 = loc("tests/app/vecscale/vecscale.cpp":16:0)
#loc20 = loc("tests/app/vecscale/vecscale.cpp":18:0)
#loc21 = loc(fused<#di_subprogram3>[#loc1])
#loc22 = loc(fused<#di_subprogram3>[#loc5])
#loc23 = loc(fused<#di_subprogram3>[#loc6])
#loc24 = loc(fused<#di_subprogram3>[#loc11])
#loc25 = loc(fused<#di_subprogram3>[#loc12])
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 17>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 28>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 17>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 28>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 29>
#loc28 = loc(fused<#di_lexical_block6>[#loc3])
#loc29 = loc(fused<#di_lexical_block7>[#loc7])
#loc31 = loc(fused<#di_subprogram4>[#loc16])
#loc33 = loc(fused<#di_subprogram5>[#loc20])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 29>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 29>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 15>
#loc34 = loc(fused<#di_lexical_block8>[#loc4])
#loc35 = loc(fused[#loc26, #loc28])
#loc36 = loc(fused[#loc27, #loc29])
#loc37 = loc(fused<#di_lexical_block10>[#loc14])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 29>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file1, line = 29>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 15>
#loc39 = loc(fused<#di_lexical_block12>[#loc8])
#loc40 = loc(fused<#di_lexical_block13>[#loc14])
#loc41 = loc(fused<#di_lexical_block14>[#loc18])
#loc42 = loc(fused<#di_lexical_block15>[#loc9])
#loc43 = loc(fused<#di_lexical_block15>[#loc10])
#loc44 = loc(fused<#di_lexical_block16>[#loc15])
#loc45 = loc(fused<#di_lexical_block17>[#loc19])
#loc46 = loc(fused[#loc38, #loc41])
