#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/vecsum/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/vecsum/vecsum.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/vecsum/main.cpp":13:0)
#loc9 = loc("tests/app/vecsum/vecsum.cpp":26:0)
#loc13 = loc("tests/app/vecsum/vecsum.cpp":12:0)
#loc14 = loc("tests/app/vecsum/vecsum.cpp":16:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 13>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 32>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 13, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_result", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_result", file = #di_file, line = 21, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram1, name = "sum", file = #di_file1, line = 30, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file1, line = 32, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "sum", file = #di_file1, line = 15, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "init_value", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "init_value", file = #di_file1, line = 27, arg = 2, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 28, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "init_value", file = #di_file1, line = 13, arg = 2, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 14, arg = 3, type = #di_derived_type2>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 13>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file1, line = 26, arg = 1, type = #di_derived_type4>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 12, arg = 1, type = #di_derived_type4>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_derived_type1, #di_derived_type4, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecsum_dsa", linkageName = "_Z10vecsum_dsaPKjjj", file = #di_file1, line = 26, scopeLine = 28, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable14, #di_local_variable10, #di_local_variable11, #di_local_variable3, #di_local_variable4>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecsum_cpu", linkageName = "_Z10vecsum_cpuPKjjj", file = #di_file1, line = 12, scopeLine = 14, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable15, #di_local_variable12, #di_local_variable13, #di_local_variable5, #di_local_variable6>
#loc22 = loc(fused<#di_lexical_block3>[#loc3])
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 16>
#loc25 = loc(fused<#di_subprogram4>[#loc9])
#loc27 = loc(fused<#di_subprogram5>[#loc13])
#loc32 = loc(fused<#di_lexical_block8>[#loc14])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[118, 101, 99, 115, 117, 109, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[118, 101, 99, 115, 117, 109, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<14xi8> = dense<[108, 111, 111, 109, 46, 114, 101, 100, 117, 99, 101, 61, 43, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 118, 101, 99, 115, 117, 109, 47, 118, 101, 99, 115, 117, 109, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc17)
    %false = arith.constant false loc(#loc17)
    %0 = seq.const_clock  low loc(#loc17)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c100_i32 = arith.constant 100 : i32 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %1 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc29)
      %13 = arith.trunci %arg0 : i64 to i32 loc(#loc29)
      memref.store %13, %alloca[%12] : memref<1024xi32> loc(#loc29)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc24)
      %15 = arith.cmpi ne, %14, %c1024_i64 : i64 loc(#loc30)
      scf.condition(%15) %14 : i64 loc(#loc22)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block3>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc22)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc22)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc18)
    %4 = call @_Z10vecsum_cpuPKjjj(%cast, %c100_i32, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> i32 loc(#loc18)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc19)
    %chanOutput_0, %ready_1 = esi.wrap.vr %c100_i32, %true : i32 loc(#loc19)
    %chanOutput_2, %ready_3 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc19)
    %chanOutput_4, %ready_5 = esi.wrap.vr %true, %true : i1 loc(#loc19)
    %5:2 = handshake.esi_instance @_Z10vecsum_dsaPKjjj_esi "_Z10vecsum_dsaPKjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>) loc(#loc19)
    %rawOutput, %valid = esi.unwrap.vr %5#0, %true : i32 loc(#loc19)
    %rawOutput_6, %valid_7 = esi.unwrap.vr %5#1, %true : i1 loc(#loc19)
    %6 = arith.cmpi ne, %4, %rawOutput : i32 loc(#loc23)
    %cast_8 = memref.cast %1 : memref<15xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc20)
    %cast_9 = memref.cast %2 : memref<15xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc20)
    %7 = arith.select %6, %cast_8, %cast_9 : memref<?xi8, strided<[1], offset: ?>> loc(#loc20)
    %8 = arith.extui %6 : i1 to i32 loc(#loc20)
    %intptr = memref.extract_aligned_pointer_as_index %7 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc20)
    %9 = arith.index_cast %intptr : index to i64 loc(#loc20)
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr loc(#loc20)
    %11 = llvm.call @puts(%10) : (!llvm.ptr) -> i32 loc(#loc20)
    return %8 : i32 loc(#loc21)
  } loc(#loc17)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10vecsum_dsaPKjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg1: i32 loc(fused<#di_subprogram4>[#loc9]), %arg2: i32 loc(fused<#di_subprogram4>[#loc9]), %arg3: i1 loc(fused<#di_subprogram4>[#loc9]), ...) -> (i32, i1) attributes {argNames = ["A", "init_value", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["sum", "done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc25)
    %1 = handshake.join %0 : none loc(#loc25)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc33)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc31)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc31)
    %6 = arith.index_cast %3 : i64 to index loc(#loc31)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc31)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc31)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc31)
    %8 = dataflow.carry %willContinue, %arg1, %9 : i1, i32, i32 -> i32 loc(#loc31)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc31)
    handshake.sink %afterCond_1 : i1 loc(#loc31)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %8 : i32 loc(#loc31)
    %dataResult, %addressResults = handshake.load [%afterValue] %14#0, %15 : index, i32 loc(#loc35)
    %9 = arith.addi %afterValue_0, %dataResult : i32 loc(#loc35)
    %10 = handshake.constant %1 {value = 0 : index} : index loc(#loc31)
    %11 = handshake.constant %1 {value = 1 : index} : index loc(#loc31)
    %12 = arith.select %4, %11, %10 : index loc(#loc31)
    %13 = handshake.mux %12 [%falseResult_3, %arg1] : index, i32 loc(#loc31)
    %14:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc25)
    %15 = dataflow.carry %willContinue, %falseResult, %trueResult_4 : i1, none, none -> none loc(#loc31)
    %trueResult_4, %falseResult_5 = handshake.cond_br %willContinue, %14#1 : none loc(#loc31)
    %16 = handshake.mux %12 [%falseResult_5, %trueResult] : index, none loc(#loc31)
    %17 = handshake.constant %16 {value = true} : i1 loc(#loc25)
    handshake.return %13, %17 : i32, i1 loc(#loc25)
  } loc(#loc25)
  handshake.func @_Z10vecsum_dsaPKjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc9]), %arg1: i32 loc(fused<#di_subprogram4>[#loc9]), %arg2: i32 loc(fused<#di_subprogram4>[#loc9]), %arg3: none loc(fused<#di_subprogram4>[#loc9]), ...) -> (i32, none) attributes {argNames = ["A", "init_value", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["sum", "done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc25)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg2, %1 : i32 loc(#loc33)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc31)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc31)
    %5 = arith.index_cast %2 : i64 to index loc(#loc31)
    %6 = arith.index_cast %arg2 : i32 to index loc(#loc31)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"], step_op = "+=", stop_cond = "!="} loc(#loc31)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc31)
    %7 = dataflow.carry %willContinue, %arg1, %8 : i1, i32, i32 -> i32 loc(#loc31)
    %afterValue_0, %afterCond_1 = dataflow.gate %7, %willContinue : i32, i1 -> i32, i1 loc(#loc31)
    handshake.sink %afterCond_1 : i1 loc(#loc31)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %7 : i32 loc(#loc31)
    %dataResult, %addressResults = handshake.load [%afterValue] %13#0, %14 : index, i32 loc(#loc35)
    %8 = arith.addi %afterValue_0, %dataResult : i32 loc(#loc35)
    %9 = handshake.constant %0 {value = 0 : index} : index loc(#loc31)
    %10 = handshake.constant %0 {value = 1 : index} : index loc(#loc31)
    %11 = arith.select %3, %10, %9 : index loc(#loc31)
    %12 = handshake.mux %11 [%falseResult_3, %arg1] : index, i32 loc(#loc31)
    %13:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc25)
    %14 = dataflow.carry %willContinue, %falseResult, %trueResult_4 : i1, none, none -> none loc(#loc31)
    %trueResult_4, %falseResult_5 = handshake.cond_br %willContinue, %13#1 : none loc(#loc31)
    %15 = handshake.mux %11 [%falseResult_5, %trueResult] : index, none loc(#loc31)
    handshake.return %12, %15 : i32, none loc(#loc26)
  } loc(#loc25)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z10vecsum_cpuPKjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc13]), %arg1: i32 loc(fused<#di_subprogram5>[#loc13]), %arg2: i32 loc(fused<#di_subprogram5>[#loc13])) -> i32 {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc34)
    %1 = scf.if %0 -> (i32) {
      scf.yield %arg1 : i32 loc(#loc32)
    } else {
      %2 = arith.extui %arg2 : i32 to i64 loc(#loc34)
      %3:2 = scf.while (%arg3 = %c0_i64, %arg4 = %arg1) : (i64, i32) -> (i64, i32) {
        %4 = arith.index_cast %arg3 : i64 to index loc(#loc36)
        %5 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc36)
        %6 = arith.addi %5, %arg4 : i32 loc(#loc36)
        %7 = arith.addi %arg3, %c1_i64 : i64 loc(#loc34)
        %8 = arith.cmpi ne, %7, %2 : i64 loc(#loc37)
        scf.condition(%8) %7, %6 : i64, i32 loc(#loc32)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block8>[#loc14]), %arg4: i32 loc(fused<#di_lexical_block8>[#loc14])):
        scf.yield %arg3, %arg4 : i64, i32 loc(#loc32)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc32)
      scf.yield %3#1 : i32 loc(#loc32)
    } loc(#loc32)
    return %1 : i32 loc(#loc28)
  } loc(#loc27)
} loc(#loc)
#loc = loc("tests/app/vecsum/main.cpp":0:0)
#loc1 = loc("tests/app/vecsum/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/vecsum/main.cpp":14:0)
#loc5 = loc("tests/app/vecsum/main.cpp":18:0)
#loc6 = loc("tests/app/vecsum/main.cpp":21:0)
#loc7 = loc("tests/app/vecsum/main.cpp":24:0)
#loc8 = loc("tests/app/vecsum/main.cpp":31:0)
#loc10 = loc("tests/app/vecsum/vecsum.cpp":32:0)
#loc11 = loc("tests/app/vecsum/vecsum.cpp":33:0)
#loc12 = loc("tests/app/vecsum/vecsum.cpp":35:0)
#loc15 = loc("tests/app/vecsum/vecsum.cpp":17:0)
#loc16 = loc("tests/app/vecsum/vecsum.cpp":19:0)
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 24>
#loc17 = loc(fused<#di_subprogram3>[#loc1])
#loc18 = loc(fused<#di_subprogram3>[#loc5])
#loc19 = loc(fused<#di_subprogram3>[#loc6])
#loc20 = loc(fused<#di_subprogram3>[#loc])
#loc21 = loc(fused<#di_subprogram3>[#loc8])
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 13>
#loc23 = loc(fused<#di_lexical_block4>[#loc7])
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 13>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 32>
#loc24 = loc(fused<#di_lexical_block5>[#loc3])
#loc26 = loc(fused<#di_subprogram4>[#loc12])
#loc28 = loc(fused<#di_subprogram5>[#loc16])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 32>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 16>
#loc29 = loc(fused<#di_lexical_block6>[#loc4])
#loc30 = loc(fused[#loc22, #loc24])
#loc31 = loc(fused<#di_lexical_block7>[#loc10])
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file1, line = 32>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file1, line = 16>
#loc33 = loc(fused<#di_lexical_block9>[#loc10])
#loc34 = loc(fused<#di_lexical_block10>[#loc14])
#loc35 = loc(fused<#di_lexical_block11>[#loc11])
#loc36 = loc(fused<#di_lexical_block12>[#loc15])
#loc37 = loc(fused[#loc32, #loc34])
