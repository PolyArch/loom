#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned long", sizeInBits = 64, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/cdma/cdma.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/cdma/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/cdma/cdma.cpp":12:0)
#loc6 = loc("tests/app/cdma/cdma.cpp":23:0)
#loc11 = loc("tests/app/cdma/main.cpp":16:0)
#loc15 = loc("tests/app/cdma/main.cpp":27:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "size_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 15>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 26>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 27>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 15, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 26, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 27, type = #di_derived_type1>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 14, arg = 3, type = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file, line = 25, arg = 3, type = #di_derived_type3>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 6, type = #di_derived_type3>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram2, name = "SRC", file = #di_file1, line = 9, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_DST", file = #di_file1, line = 12, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_DST", file = #di_file1, line = 13, type = #di_composite_type>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type6>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "DST", file = #di_file, line = 13, arg = 2, type = #di_derived_type7>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "DST", file = #di_file, line = 24, arg = 2, type = #di_derived_type7>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable2, #di_local_variable3>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 16>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 27>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "SRC", file = #di_file, line = 12, arg = 1, type = #di_derived_type8>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "SRC", file = #di_file, line = 23, arg = 1, type = #di_derived_type8>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type8, #di_derived_type7, #di_derived_type3>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "cdma_cpu", linkageName = "_Z8cdma_cpuPKjPjm", file = #di_file, line = 12, scopeLine = 14, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable12, #di_local_variable10, #di_local_variable4, #di_local_variable>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "cdma_dsa", linkageName = "_Z8cdma_dsaPKjPjm", file = #di_file, line = 23, scopeLine = 25, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable11, #di_local_variable5, #di_local_variable1>
#loc26 = loc(fused<#di_lexical_block4>[#loc11])
#loc27 = loc(fused<#di_lexical_block5>[#loc15])
#loc28 = loc(fused<#di_subprogram4>[#loc1])
#loc30 = loc(fused<#di_subprogram5>[#loc6])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<24xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 99, 100, 109, 97, 47, 99, 100, 109, 97, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<13xi8> = dense<[99, 100, 109, 97, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<13xi8> = dense<[99, 100, 109, 97, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z8cdma_cpuPKjPjm(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i64 loc(fused<#di_subprogram4>[#loc1])) {
    %c1 = arith.constant 1 : index loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c2_i64 = arith.constant 2 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i64 : i64 loc(#loc39)
    scf.if %0 {
    } else {
      %1 = arith.shli %arg2, %c2_i64 : i64 loc(#loc34)
      %2 = arith.index_cast %1 : i64 to index loc(#loc42)
      %3 = arith.divui %2, %c4 : index loc(#loc42)
      scf.for %arg3 = %c0 to %3 step %c1 {
        %4 = memref.load %arg0[%arg3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc42)
        memref.store %4, %arg1[%arg3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc42)
      } loc(#loc42)
    } loc(#loc34)
    return loc(#loc29)
  } loc(#loc28)
  handshake.func @_Z8cdma_dsaPKjPjm_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc6]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc6]), %arg2: i64 loc(fused<#di_subprogram5>[#loc6]), %arg3: i1 loc(fused<#di_subprogram5>[#loc6]), ...) -> i1 attributes {argNames = ["SRC", "DST", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc30)
    %1 = handshake.join %0 : none loc(#loc30)
    %2 = handshake.constant %1 {value = 1 : index} : index loc(#loc35)
    %3 = handshake.constant %1 {value = 4 : index} : index loc(#loc2)
    %4 = handshake.constant %1 {value = 2 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = handshake.constant %1 {value = 0 : index} : index loc(#loc35)
    %7 = arith.cmpi eq, %arg2, %5 : i64 loc(#loc40)
    %8 = arith.shli %arg2, %4 : i64 loc(#loc35)
    %9 = arith.index_cast %8 : i64 to index loc(#loc43)
    %10 = arith.divui %9, %3 : index loc(#loc43)
    %index, %willContinue = dataflow.stream %6, %2, %10 loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %11#0, %trueResult_1 : index, i32 loc(#loc43)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %dataResult, %trueResult_3 : index, i32 loc(#loc43)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc30)
    %12 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc30)
    %trueResult, %falseResult = handshake.cond_br %7, %1 : none loc(#loc35)
    %13 = dataflow.carry %willContinue, %falseResult, %11#1 : i1, none, none -> none loc(#loc43)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %13 : none loc(#loc43)
    %14 = arith.select %7, %2, %6 : index loc(#loc35)
    %15 = handshake.mux %14 [%falseResult_2, %trueResult] : index, none loc(#loc35)
    %16 = dataflow.carry %willContinue, %falseResult, %12 : i1, none, none -> none loc(#loc43)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %16 : none loc(#loc43)
    %17 = handshake.mux %14 [%falseResult_4, %trueResult] : index, none loc(#loc35)
    %18 = handshake.join %15, %17 : none, none loc(#loc30)
    %19 = handshake.constant %18 {value = true} : i1 loc(#loc30)
    handshake.return %19 : i1 loc(#loc30)
  } loc(#loc30)
  handshake.func @_Z8cdma_dsaPKjPjm(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc6]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc6]), %arg2: i64 loc(fused<#di_subprogram5>[#loc6]), %arg3: none loc(fused<#di_subprogram5>[#loc6]), ...) -> none attributes {argNames = ["SRC", "DST", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc30)
    %1 = handshake.constant %0 {value = 1 : index} : index loc(#loc35)
    %2 = handshake.constant %0 {value = 4 : index} : index loc(#loc2)
    %3 = handshake.constant %0 {value = 2 : i64} : i64 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %5 = handshake.constant %0 {value = 0 : index} : index loc(#loc35)
    %6 = arith.cmpi eq, %arg2, %4 : i64 loc(#loc40)
    %7 = arith.shli %arg2, %3 : i64 loc(#loc35)
    %8 = arith.index_cast %7 : i64 to index loc(#loc43)
    %9 = arith.divui %8, %2 : index loc(#loc43)
    %index, %willContinue = dataflow.stream %5, %1, %9 loc(#loc43)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc43)
    %dataResult, %addressResults = handshake.load [%afterValue] %10#0, %trueResult_1 : index, i32 loc(#loc43)
    %dataResult_0, %addressResult = handshake.store [%afterValue] %dataResult, %trueResult_3 : index, i32 loc(#loc43)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc30)
    %11 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_0, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc30)
    %trueResult, %falseResult = handshake.cond_br %6, %0 : none loc(#loc35)
    %12 = dataflow.carry %willContinue, %falseResult, %10#1 : i1, none, none -> none loc(#loc43)
    %trueResult_1, %falseResult_2 = handshake.cond_br %willContinue, %12 : none loc(#loc43)
    %13 = arith.select %6, %1, %5 : index loc(#loc35)
    %14 = handshake.mux %13 [%falseResult_2, %trueResult] : index, none loc(#loc35)
    %15 = dataflow.carry %willContinue, %falseResult, %11 : i1, none, none -> none loc(#loc43)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %15 : none loc(#loc43)
    %16 = handshake.mux %13 [%falseResult_4, %trueResult] : index, none loc(#loc35)
    %17 = handshake.join %14, %16 : none, none loc(#loc30)
    handshake.return %17 : none loc(#loc31)
  } loc(#loc30)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc21)
    %false = arith.constant false loc(#loc21)
    %0 = seq.const_clock  low loc(#loc21)
    %c2_i32 = arith.constant 2 : i32 loc(#loc21)
    %1 = ub.poison : i64 loc(#loc21)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<13xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<13xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc36)
      %11 = arith.muli %10, %c3_i32 : i32 loc(#loc36)
      %12 = arith.addi %11, %c7_i32 : i32 loc(#loc36)
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc36)
      memref.store %12, %alloca[%13] : memref<1024xi32> loc(#loc36)
      %14 = arith.addi %arg0, %c1_i64 : i64 loc(#loc32)
      %15 = arith.cmpi ne, %14, %c1024_i64 : i64 loc(#loc37)
      scf.condition(%15) %14 : i64 loc(#loc26)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block4>[#loc11])):
      scf.yield %arg0 : i64 loc(#loc26)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc26)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc22)
    %cast_2 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc22)
    call @_Z8cdma_cpuPKjPjm(%cast, %cast_2, %c1024_i64) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i64) -> () loc(#loc22)
    %cast_3 = memref.cast %alloca_1 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc23)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c1024_i64, %true : i64 loc(#loc23)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc23)
    %5 = handshake.esi_instance @_Z8cdma_dsaPKjPjm_esi "_Z8cdma_dsaPKjPjm_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i64>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc23)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc23)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      %11 = memref.load %alloca_0[%10] : memref<1024xi32> loc(#loc41)
      %12 = memref.load %alloca_1[%10] : memref<1024xi32> loc(#loc41)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc41)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc33)
        %17 = arith.cmpi eq, %16, %c1024_i64 : i64 loc(#loc33)
        %18 = arith.extui %17 : i1 to i32 loc(#loc27)
        %19 = arith.cmpi ne, %16, %c1024_i64 : i64 loc(#loc38)
        %20 = arith.extui %19 : i1 to i32 loc(#loc27)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc41)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc41)
      } loc(#loc41)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc27)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc27)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block5>[#loc15]), %arg1: i1 loc(fused<#di_lexical_block5>[#loc15]), %arg2: i32 loc(fused<#di_lexical_block5>[#loc15])):
      scf.yield %arg0 : i64 loc(#loc27)
    } loc(#loc27)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc27)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc27)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<13xi8> -> index loc(#loc44)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc44)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc44)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc44)
      scf.yield %c1_i32 : i32 loc(#loc45)
    } loc(#loc27)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<13xi8> -> index loc(#loc24)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc24)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc24)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc24)
    } loc(#loc2)
    return %9 : i32 loc(#loc25)
  } loc(#loc21)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/cdma/cdma.cpp":0:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/cdma/cdma.cpp":15:0)
#loc4 = loc("tests/app/cdma/cdma.cpp":16:0)
#loc5 = loc("tests/app/cdma/cdma.cpp":18:0)
#loc7 = loc("tests/app/cdma/cdma.cpp":26:0)
#loc8 = loc("tests/app/cdma/cdma.cpp":27:0)
#loc9 = loc("tests/app/cdma/cdma.cpp":29:0)
#loc10 = loc("tests/app/cdma/main.cpp":5:0)
#loc12 = loc("tests/app/cdma/main.cpp":17:0)
#loc13 = loc("tests/app/cdma/main.cpp":21:0)
#loc14 = loc("tests/app/cdma/main.cpp":24:0)
#loc16 = loc("tests/app/cdma/main.cpp":28:0)
#loc17 = loc("tests/app/cdma/main.cpp":29:0)
#loc18 = loc("tests/app/cdma/main.cpp":30:0)
#loc19 = loc("tests/app/cdma/main.cpp":34:0)
#loc20 = loc("tests/app/cdma/main.cpp":36:0)
#loc21 = loc(fused<#di_subprogram3>[#loc10])
#loc22 = loc(fused<#di_subprogram3>[#loc13])
#loc23 = loc(fused<#di_subprogram3>[#loc14])
#loc24 = loc(fused<#di_subprogram3>[#loc19])
#loc25 = loc(fused<#di_subprogram3>[#loc20])
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 16>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 27>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 15>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 26>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 16>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 27>
#loc29 = loc(fused<#di_subprogram4>[#loc5])
#loc31 = loc(fused<#di_subprogram5>[#loc9])
#loc32 = loc(fused<#di_lexical_block6>[#loc11])
#loc33 = loc(fused<#di_lexical_block7>[#loc15])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 15>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 26>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file1, line = 28>
#loc34 = loc(fused<#di_lexical_block8>[#loc3])
#loc35 = loc(fused<#di_lexical_block9>[#loc7])
#loc36 = loc(fused<#di_lexical_block10>[#loc12])
#loc37 = loc(fused[#loc26, #loc32])
#loc38 = loc(fused[#loc27, #loc33])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 15>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 26>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 28>
#loc39 = loc(fused<#di_lexical_block12>[#loc3])
#loc40 = loc(fused<#di_lexical_block13>[#loc7])
#loc41 = loc(fused<#di_lexical_block14>[#loc16])
#loc42 = loc(fused<#di_lexical_block15>[#loc4])
#loc43 = loc(fused<#di_lexical_block16>[#loc8])
#loc44 = loc(fused<#di_lexical_block17>[#loc17])
#loc45 = loc(fused<#di_lexical_block17>[#loc18])
