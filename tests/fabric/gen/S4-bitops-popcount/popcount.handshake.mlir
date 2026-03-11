#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/popcount/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/popcount/popcount.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/popcount/main.cpp":10:0)
#loc7 = loc("tests/app/popcount/main.cpp":25:0)
#loc13 = loc("tests/app/popcount/popcount.cpp":32:0)
#loc20 = loc("tests/app/popcount/popcount.cpp":13:0)
#loc21 = loc("tests/app/popcount/popcount.cpp":16:0)
#loc23 = loc("tests/app/popcount/popcount.cpp":20:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 10>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 25>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 37>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 37>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 16>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 37>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 16>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 10, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 25, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 37, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_count", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_count", file = #di_file, line = 16, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 34, arg = 3, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "count", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 15, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file1, line = 17, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "count", file = #di_file1, line = 18, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_count", file = #di_file1, line = 33, arg = 2, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_count", file = #di_file1, line = 14, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable, #di_local_variable6, #di_local_variable7, #di_local_variable1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 10>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 25>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file1, line = 32, arg = 1, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_data", file = #di_file1, line = 13, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "popcount_dsa", linkageName = "_Z12popcount_dsaPKjPjj", file = #di_file1, line = 32, scopeLine = 34, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable14, #di_local_variable8, #di_local_variable2, #di_local_variable9, #di_local_variable10>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "popcount_cpu", linkageName = "_Z12popcount_cpuPKjPjj", file = #di_file1, line = 13, scopeLine = 15, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable15, #di_local_variable11, #di_local_variable3, #di_local_variable12, #di_local_variable13>
#loc33 = loc(fused<#di_lexical_block8>[#loc3])
#loc34 = loc(fused<#di_lexical_block9>[#loc7])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 16>
#loc37 = loc(fused<#di_subprogram4>[#loc13])
#loc39 = loc(fused<#di_subprogram5>[#loc20])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 16>
#loc45 = loc(fused<#di_lexical_block15>[#loc21])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 16>
#loc55 = loc(fused<#di_lexical_block21>[#loc23])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<17xi8> = dense<[112, 111, 112, 99, 111, 117, 110, 116, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<17xi8> = dense<[112, 111, 112, 99, 111, 117, 110, 116, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<32xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 112, 111, 112, 99, 111, 117, 110, 116, 47, 112, 111, 112, 99, 111, 117, 110, 116, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc28)
    %false = arith.constant false loc(#loc28)
    %0 = seq.const_clock  low loc(#loc28)
    %c2_i32 = arith.constant 2 : i32 loc(#loc28)
    %1 = ub.poison : i64 loc(#loc28)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c305485432_i32 = arith.constant 305485432 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<17xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<17xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc41)
      %11 = arith.muli %10, %c305485432_i32 : i32 loc(#loc41)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      memref.store %11, %alloca[%12] : memref<256xi32> loc(#loc41)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc42)
      scf.condition(%14) %13 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    call @_Z12popcount_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc29)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %5 = handshake.esi_instance @_Z12popcount_dsaPKjPjj_esi "_Z12popcount_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc30)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc46)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc46)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc46)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc46)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc36)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc36)
        %18 = arith.extui %17 : i1 to i32 loc(#loc34)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc43)
        %20 = arith.extui %19 : i1 to i32 loc(#loc34)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc46)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc46)
      } loc(#loc46)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc34)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc7]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc7]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc7])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc34)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<17xi8> -> index loc(#loc49)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc49)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc49)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc49)
      scf.yield %c1_i32 : i32 loc(#loc50)
    } loc(#loc34)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<17xi8> -> index loc(#loc31)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc31)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc31)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc31)
    } loc(#loc2)
    return %9 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z12popcount_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: i1 loc(fused<#di_subprogram4>[#loc13]), ...) -> i1 attributes {argNames = ["input_data", "output_count", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc37)
    %1 = handshake.join %0 : none loc(#loc37)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc44)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %7 = arith.index_cast %4 : i64 to index loc(#loc44)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %22#0, %24 : index, i32 loc(#loc51)
    %10 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc52)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %9 : none loc(#loc52)
    %11 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc52)
    %12 = arith.index_cast %dataResult : i32 to index loc(#loc52)
    %13 = arith.index_cast %3 : i32 to index loc(#loc52)
    %index_2, %willContinue_3 = dataflow.stream %12, %11, %13 {step_op = ">>=", stop_cond = "!="} loc(#loc52)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc52)
    %14 = dataflow.carry %willContinue_3, %3, %17 : i1, i32, i32 -> i32 loc(#loc52)
    %afterValue_6, %afterCond_7 = dataflow.gate %14, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc52)
    handshake.sink %afterCond_7 : i1 loc(#loc52)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %14 : i32 loc(#loc52)
    %15 = arith.index_cast %afterValue_4 : index to i32 loc(#loc52)
    %16 = arith.andi %15, %2 : i32 loc(#loc58)
    %17 = arith.addi %afterValue_6, %16 : i32 loc(#loc58)
    %18 = handshake.constant %9 {value = 0 : index} : index loc(#loc52)
    %19 = handshake.constant %9 {value = 1 : index} : index loc(#loc52)
    %20 = arith.select %10, %19, %18 : index loc(#loc52)
    %21 = handshake.mux %20 [%falseResult_9, %3] : index, i32 loc(#loc52)
    %dataResult_10, %addressResult = handshake.store [%afterValue] %21, %29 : index, i32 loc(#loc53)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %23 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc37)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc44)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %22#1 : none loc(#loc44)
    %25 = handshake.constant %1 {value = 0 : index} : index loc(#loc44)
    %26 = handshake.constant %1 {value = 1 : index} : index loc(#loc44)
    %27 = arith.select %5, %26, %25 : index loc(#loc44)
    %28 = handshake.mux %27 [%falseResult_12, %trueResult] : index, none loc(#loc44)
    %29 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc44)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %23 : none loc(#loc44)
    %30 = handshake.mux %27 [%falseResult_14, %trueResult] : index, none loc(#loc44)
    %31 = handshake.join %28, %30 : none, none loc(#loc37)
    %32 = handshake.constant %31 {value = true} : i1 loc(#loc37)
    handshake.return %32 : i1 loc(#loc37)
  } loc(#loc37)
  handshake.func @_Z12popcount_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: none loc(fused<#di_subprogram4>[#loc13]), ...) -> none attributes {argNames = ["input_data", "output_count", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc37)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc47)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc44)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc44)
    %6 = arith.index_cast %3 : i64 to index loc(#loc44)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc44)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc44)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc44)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc44)
    %dataResult, %addressResults = handshake.load [%afterValue] %21#0, %23 : index, i32 loc(#loc51)
    %9 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc52)
    %trueResult_0, %falseResult_1 = handshake.cond_br %9, %8 : none loc(#loc52)
    %10 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc52)
    %11 = arith.index_cast %dataResult : i32 to index loc(#loc52)
    %12 = arith.index_cast %2 : i32 to index loc(#loc52)
    %index_2, %willContinue_3 = dataflow.stream %11, %10, %12 {step_op = ">>=", stop_cond = "!="} loc(#loc52)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc52)
    %13 = dataflow.carry %willContinue_3, %2, %16 : i1, i32, i32 -> i32 loc(#loc52)
    %afterValue_6, %afterCond_7 = dataflow.gate %13, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc52)
    handshake.sink %afterCond_7 : i1 loc(#loc52)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %13 : i32 loc(#loc52)
    %14 = arith.index_cast %afterValue_4 : index to i32 loc(#loc52)
    %15 = arith.andi %14, %1 : i32 loc(#loc58)
    %16 = arith.addi %afterValue_6, %15 : i32 loc(#loc58)
    %17 = handshake.constant %8 {value = 0 : index} : index loc(#loc52)
    %18 = handshake.constant %8 {value = 1 : index} : index loc(#loc52)
    %19 = arith.select %9, %18, %17 : index loc(#loc52)
    %20 = handshake.mux %19 [%falseResult_9, %2] : index, i32 loc(#loc52)
    %dataResult_10, %addressResult = handshake.store [%afterValue] %20, %28 : index, i32 loc(#loc53)
    %21:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %22 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc37)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc44)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %21#1 : none loc(#loc44)
    %24 = handshake.constant %0 {value = 0 : index} : index loc(#loc44)
    %25 = handshake.constant %0 {value = 1 : index} : index loc(#loc44)
    %26 = arith.select %4, %25, %24 : index loc(#loc44)
    %27 = handshake.mux %26 [%falseResult_12, %trueResult] : index, none loc(#loc44)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc44)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %22 : none loc(#loc44)
    %29 = handshake.mux %26 [%falseResult_14, %trueResult] : index, none loc(#loc44)
    %30 = handshake.join %27, %29 : none, none loc(#loc37)
    handshake.return %30 : none loc(#loc38)
  } loc(#loc37)
  func.func @_Z12popcount_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc20]), %arg2: i32 loc(fused<#di_subprogram5>[#loc20])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc48)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc48)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc54)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc54)
        %5 = arith.cmpi eq, %4, %c0_i32 : i32 loc(#loc55)
        %6 = scf.if %5 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc55)
        } else {
          %9:2 = scf.while (%arg4 = %c0_i32, %arg5 = %4) : (i32, i32) -> (i32, i32) {
            %10 = arith.andi %arg5, %c1_i32 : i32 loc(#loc59)
            %11 = arith.addi %arg4, %10 : i32 loc(#loc59)
            %12 = arith.shrui %arg5, %c1_i32 : i32 loc(#loc60)
            %13 = arith.cmpi ne, %12, %c0_i32 : i32 loc(#loc55)
            scf.condition(%13) %11, %12 : i32, i32 loc(#loc55)
          } do {
          ^bb0(%arg4: i32 loc(fused<#di_lexical_block21>[#loc23]), %arg5: i32 loc(fused<#di_lexical_block21>[#loc23])):
            scf.yield %arg4, %arg5 : i32, i32 loc(#loc55)
          } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc55)
          scf.yield %9#0 : i32 loc(#loc55)
        } loc(#loc55)
        memref.store %6, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc56)
        %7 = arith.addi %arg3, %c1_i64 : i64 loc(#loc48)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc57)
        scf.condition(%8) %7 : i64 loc(#loc45)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block15>[#loc21])):
        scf.yield %arg3 : i64 loc(#loc45)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc45)
    } loc(#loc45)
    return loc(#loc40)
  } loc(#loc39)
} loc(#loc)
#loc = loc("tests/app/popcount/main.cpp":0:0)
#loc1 = loc("tests/app/popcount/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/popcount/main.cpp":11:0)
#loc5 = loc("tests/app/popcount/main.cpp":19:0)
#loc6 = loc("tests/app/popcount/main.cpp":22:0)
#loc8 = loc("tests/app/popcount/main.cpp":26:0)
#loc9 = loc("tests/app/popcount/main.cpp":27:0)
#loc10 = loc("tests/app/popcount/main.cpp":28:0)
#loc11 = loc("tests/app/popcount/main.cpp":32:0)
#loc12 = loc("tests/app/popcount/main.cpp":34:0)
#loc14 = loc("tests/app/popcount/popcount.cpp":37:0)
#loc15 = loc("tests/app/popcount/popcount.cpp":38:0)
#loc16 = loc("tests/app/popcount/popcount.cpp":41:0)
#loc17 = loc("tests/app/popcount/popcount.cpp":42:0)
#loc18 = loc("tests/app/popcount/popcount.cpp":46:0)
#loc19 = loc("tests/app/popcount/popcount.cpp":48:0)
#loc22 = loc("tests/app/popcount/popcount.cpp":17:0)
#loc24 = loc("tests/app/popcount/popcount.cpp":21:0)
#loc25 = loc("tests/app/popcount/popcount.cpp":22:0)
#loc26 = loc("tests/app/popcount/popcount.cpp":25:0)
#loc27 = loc("tests/app/popcount/popcount.cpp":27:0)
#loc28 = loc(fused<#di_subprogram3>[#loc1])
#loc29 = loc(fused<#di_subprogram3>[#loc5])
#loc30 = loc(fused<#di_subprogram3>[#loc6])
#loc31 = loc(fused<#di_subprogram3>[#loc11])
#loc32 = loc(fused<#di_subprogram3>[#loc12])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 10>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 25>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 10>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 25>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 37>
#loc35 = loc(fused<#di_lexical_block10>[#loc3])
#loc36 = loc(fused<#di_lexical_block11>[#loc7])
#loc38 = loc(fused<#di_subprogram4>[#loc19])
#loc40 = loc(fused<#di_subprogram5>[#loc27])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 26>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 37>
#loc41 = loc(fused<#di_lexical_block12>[#loc4])
#loc42 = loc(fused[#loc33, #loc35])
#loc43 = loc(fused[#loc34, #loc36])
#loc44 = loc(fused<#di_lexical_block14>[#loc14])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 26>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 37>
#loc46 = loc(fused<#di_lexical_block16>[#loc8])
#loc47 = loc(fused<#di_lexical_block17>[#loc14])
#loc48 = loc(fused<#di_lexical_block18>[#loc21])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 41>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 20>
#loc49 = loc(fused<#di_lexical_block19>[#loc9])
#loc50 = loc(fused<#di_lexical_block19>[#loc10])
#loc51 = loc(fused<#di_lexical_block20>[#loc15])
#loc52 = loc(fused<#di_lexical_block20>[#loc16])
#loc53 = loc(fused<#di_lexical_block20>[#loc18])
#loc54 = loc(fused<#di_lexical_block21>[#loc22])
#loc56 = loc(fused<#di_lexical_block21>[#loc26])
#loc57 = loc(fused[#loc45, #loc48])
#loc58 = loc(fused<#di_lexical_block22>[#loc17])
#loc59 = loc(fused<#di_lexical_block23>[#loc24])
#loc60 = loc(fused<#di_lexical_block23>[#loc25])
