#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/transpose/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/transpose/transpose.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/transpose/main.cpp":17:0)
#loc7 = loc("tests/app/transpose/main.cpp":28:0)
#loc13 = loc("tests/app/transpose/transpose.cpp":27:0)
#loc18 = loc("tests/app/transpose/transpose.cpp":13:0)
#loc19 = loc("tests/app/transpose/transpose.cpp":17:0)
#loc20 = loc("tests/app/transpose/transpose.cpp":18:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 17>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 28>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 31>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 17>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 31>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 17>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 65536, elements = #llvm.di_subrange<count = 2048 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 31>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 17>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 17, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 28, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 31, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 17, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 32>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 18>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "M", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "A", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_B", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_B", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "M", file = #di_file1, line = 29, arg = 3, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 30, arg = 4, type = #di_derived_type2>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "M", file = #di_file1, line = 15, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 16, arg = 4, type = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram1, name = "B", file = #di_file1, line = 28, arg = 2, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file1, line = 32, type = #di_derived_type1>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "B", file = #di_file1, line = 14, arg = 2, type = #di_derived_type5>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file1, line = 18, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable, #di_local_variable1>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 17>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 28>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "A", file = #di_file1, line = 27, arg = 1, type = #di_derived_type6>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "A", file = #di_file1, line = 13, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "transpose_dsa", linkageName = "_Z13transpose_dsaPKjPjjj", file = #di_file1, line = 27, scopeLine = 30, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable13, #di_local_variable9, #di_local_variable10, #di_local_variable2, #di_local_variable14>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "transpose_cpu", linkageName = "_Z13transpose_cpuPKjPjjj", file = #di_file1, line = 13, scopeLine = 16, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable18, #di_local_variable15, #di_local_variable11, #di_local_variable12, #di_local_variable3, #di_local_variable16>
#loc28 = loc(fused<#di_lexical_block10>[#loc3])
#loc29 = loc(fused<#di_lexical_block11>[#loc7])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 17>
#loc32 = loc(fused<#di_subprogram4>[#loc13])
#loc34 = loc(fused<#di_subprogram5>[#loc18])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 17>
#loc40 = loc(fused<#di_lexical_block17>[#loc19])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 17>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 18>
#loc48 = loc(fused<#di_lexical_block25>[#loc20])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<18xi8> = dense<[116, 114, 97, 110, 115, 112, 111, 115, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<18xi8> = dense<[116, 114, 97, 110, 115, 112, 111, 115, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<34xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 116, 114, 97, 110, 115, 112, 111, 115, 101, 47, 116, 114, 97, 110, 115, 112, 111, 115, 101, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc23)
    %false = arith.constant false loc(#loc23)
    %0 = seq.const_clock  low loc(#loc23)
    %c2_i32 = arith.constant 2 : i32 loc(#loc23)
    %1 = ub.poison : i64 loc(#loc23)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c2048_i64 = arith.constant 2048 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c64_i32 = arith.constant 64 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<18xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<18xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<2048xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<2048xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<2048xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc36)
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc36)
      memref.store %11, %alloca[%10] : memref<2048xi32> loc(#loc36)
      %12 = arith.addi %arg0, %c1_i64 : i64 loc(#loc30)
      %13 = arith.cmpi ne, %12, %c2048_i64 : i64 loc(#loc37)
      scf.condition(%13) %12 : i64 loc(#loc28)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc28)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc28)
    %cast = memref.cast %alloca : memref<2048xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    %cast_2 = memref.cast %alloca_0 : memref<2048xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc24)
    call @_Z13transpose_cpuPKjPjjj(%cast, %cast_2, %c32_i32, %c64_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc24)
    %cast_3 = memref.cast %alloca_1 : memref<2048xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc25)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc25)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc25)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc25)
    %chanOutput_8, %ready_9 = esi.wrap.vr %c64_i32, %true : i32 loc(#loc25)
    %chanOutput_10, %ready_11 = esi.wrap.vr %true, %true : i1 loc(#loc25)
    %5 = handshake.esi_instance @_Z13transpose_dsaPKjPjjj_esi "_Z13transpose_dsaPKjPjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8, %chanOutput_10) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc25)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc25)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      %11 = memref.load %alloca_0[%10] : memref<2048xi32> loc(#loc41)
      %12 = memref.load %alloca_1[%10] : memref<2048xi32> loc(#loc41)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc41)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc31)
        %17 = arith.cmpi eq, %16, %c2048_i64 : i64 loc(#loc31)
        %18 = arith.extui %17 : i1 to i32 loc(#loc29)
        %19 = arith.cmpi ne, %16, %c2048_i64 : i64 loc(#loc38)
        %20 = arith.extui %19 : i1 to i32 loc(#loc29)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc41)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc41)
      } loc(#loc41)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc29)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc29)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc7]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc7]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc7])):
      scf.yield %arg0 : i64 loc(#loc29)
    } loc(#loc29)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc29)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc29)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<18xi8> -> index loc(#loc44)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc44)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc44)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc44)
      scf.yield %c1_i32 : i32 loc(#loc45)
    } loc(#loc29)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<18xi8> -> index loc(#loc26)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc26)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc26)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc26)
    } loc(#loc2)
    return %9 : i32 loc(#loc27)
  } loc(#loc23)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z13transpose_dsaPKjPjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: i32 loc(fused<#di_subprogram4>[#loc13]), %arg4: i1 loc(fused<#di_subprogram4>[#loc13]), ...) -> i1 attributes {argNames = ["A", "B", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc32)
    %1 = handshake.join %0 : none loc(#loc32)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc42)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc39)
    %5 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc2)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc39)
    %7 = arith.index_cast %2 : i32 to index loc(#loc39)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc39)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc39)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc39)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc39)
    %10 = arith.index_cast %afterValue : index to i32 loc(#loc39)
    %11 = dataflow.invariant %afterCond, %5 : i1, i1 -> i1 loc(#loc47)
    %trueResult_0, %falseResult_1 = handshake.cond_br %11, %9 : none loc(#loc47)
    %12 = arith.muli %10, %arg3 : i32 loc(#loc2)
    %13 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc47)
    %14 = arith.index_cast %3 : i64 to index loc(#loc47)
    %15 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index_2, %willContinue_3 = dataflow.stream %14, %13, %15 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc47)
    %16 = arith.index_cast %afterValue_4 : index to i64 loc(#loc47)
    %17 = arith.trunci %16 : i64 to i32 loc(#loc50)
    %18 = dataflow.invariant %afterCond_5, %12 : i1, i32 -> i32 loc(#loc50)
    %19 = arith.addi %18, %17 : i32 loc(#loc50)
    %20 = arith.extui %19 : i32 to i64 loc(#loc50)
    %21 = arith.index_cast %20 : i64 to index loc(#loc50)
    %dataResult, %addressResults = handshake.load [%21] %26#0, %29 : index, i32 loc(#loc50)
    %22 = arith.muli %arg2, %17 : i32 loc(#loc50)
    %23 = arith.addi %22, %10 : i32 loc(#loc50)
    %24 = arith.extui %23 : i32 to i64 loc(#loc50)
    %25 = arith.index_cast %24 : i64 to index loc(#loc50)
    %dataResult_6, %addressResult = handshake.store [%25] %dataResult, %39 : index, i32 loc(#loc50)
    %26:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc32)
    %27 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc32)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc39)
    %trueResult_7, %falseResult_8 = handshake.cond_br %11, %28 : none loc(#loc47)
    %29 = dataflow.carry %willContinue_3, %falseResult_8, %trueResult_9 : i1, none, none -> none loc(#loc47)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue_3, %26#1 : none loc(#loc47)
    %30 = handshake.constant %28 {value = 0 : index} : index loc(#loc47)
    %31 = handshake.constant %28 {value = 1 : index} : index loc(#loc47)
    %32 = arith.select %11, %31, %30 : index loc(#loc47)
    %33 = handshake.mux %32 [%falseResult_10, %trueResult_7] : index, none loc(#loc47)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %33 : none loc(#loc39)
    %34 = handshake.constant %1 {value = 0 : index} : index loc(#loc39)
    %35 = handshake.constant %1 {value = 1 : index} : index loc(#loc39)
    %36 = arith.select %4, %35, %34 : index loc(#loc39)
    %37 = handshake.mux %36 [%falseResult_12, %trueResult] : index, none loc(#loc39)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc39)
    %trueResult_13, %falseResult_14 = handshake.cond_br %11, %38 : none loc(#loc47)
    %39 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc47)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %27 : none loc(#loc47)
    %40 = handshake.constant %38 {value = 0 : index} : index loc(#loc47)
    %41 = handshake.constant %38 {value = 1 : index} : index loc(#loc47)
    %42 = arith.select %11, %41, %40 : index loc(#loc47)
    %43 = handshake.mux %42 [%falseResult_16, %trueResult_13] : index, none loc(#loc47)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %43 : none loc(#loc39)
    %44 = handshake.mux %36 [%falseResult_18, %trueResult] : index, none loc(#loc39)
    %45 = handshake.join %37, %44 : none, none loc(#loc32)
    %46 = handshake.constant %45 {value = true} : i1 loc(#loc32)
    handshake.return %46 : i1 loc(#loc32)
  } loc(#loc32)
  handshake.func @_Z13transpose_dsaPKjPjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc13]), %arg2: i32 loc(fused<#di_subprogram4>[#loc13]), %arg3: i32 loc(fused<#di_subprogram4>[#loc13]), %arg4: none loc(fused<#di_subprogram4>[#loc13]), ...) -> none attributes {argNames = ["A", "B", "M", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc32)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg2, %1 : i32 loc(#loc42)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc39)
    %4 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc2)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc39)
    %6 = arith.index_cast %1 : i32 to index loc(#loc39)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc39)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc39)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc39)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc39)
    %9 = arith.index_cast %afterValue : index to i32 loc(#loc39)
    %10 = dataflow.invariant %afterCond, %4 : i1, i1 -> i1 loc(#loc47)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %8 : none loc(#loc47)
    %11 = arith.muli %9, %arg3 : i32 loc(#loc2)
    %12 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc47)
    %13 = arith.index_cast %2 : i64 to index loc(#loc47)
    %14 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index_2, %willContinue_3 = dataflow.stream %13, %12, %14 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc47)
    %15 = arith.index_cast %afterValue_4 : index to i64 loc(#loc47)
    %16 = arith.trunci %15 : i64 to i32 loc(#loc50)
    %17 = dataflow.invariant %afterCond_5, %11 : i1, i32 -> i32 loc(#loc50)
    %18 = arith.addi %17, %16 : i32 loc(#loc50)
    %19 = arith.extui %18 : i32 to i64 loc(#loc50)
    %20 = arith.index_cast %19 : i64 to index loc(#loc50)
    %dataResult, %addressResults = handshake.load [%20] %25#0, %28 : index, i32 loc(#loc50)
    %21 = arith.muli %arg2, %16 : i32 loc(#loc50)
    %22 = arith.addi %21, %9 : i32 loc(#loc50)
    %23 = arith.extui %22 : i32 to i64 loc(#loc50)
    %24 = arith.index_cast %23 : i64 to index loc(#loc50)
    %dataResult_6, %addressResult = handshake.store [%24] %dataResult, %38 : index, i32 loc(#loc50)
    %25:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc32)
    %26 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_6, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc32)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc39)
    %trueResult_7, %falseResult_8 = handshake.cond_br %10, %27 : none loc(#loc47)
    %28 = dataflow.carry %willContinue_3, %falseResult_8, %trueResult_9 : i1, none, none -> none loc(#loc47)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue_3, %25#1 : none loc(#loc47)
    %29 = handshake.constant %27 {value = 0 : index} : index loc(#loc47)
    %30 = handshake.constant %27 {value = 1 : index} : index loc(#loc47)
    %31 = arith.select %10, %30, %29 : index loc(#loc47)
    %32 = handshake.mux %31 [%falseResult_10, %trueResult_7] : index, none loc(#loc47)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %32 : none loc(#loc39)
    %33 = handshake.constant %0 {value = 0 : index} : index loc(#loc39)
    %34 = handshake.constant %0 {value = 1 : index} : index loc(#loc39)
    %35 = arith.select %3, %34, %33 : index loc(#loc39)
    %36 = handshake.mux %35 [%falseResult_12, %trueResult] : index, none loc(#loc39)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_17 : i1, none, none -> none loc(#loc39)
    %trueResult_13, %falseResult_14 = handshake.cond_br %10, %37 : none loc(#loc47)
    %38 = dataflow.carry %willContinue_3, %falseResult_14, %trueResult_15 : i1, none, none -> none loc(#loc47)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_3, %26 : none loc(#loc47)
    %39 = handshake.constant %37 {value = 0 : index} : index loc(#loc47)
    %40 = handshake.constant %37 {value = 1 : index} : index loc(#loc47)
    %41 = arith.select %10, %40, %39 : index loc(#loc47)
    %42 = handshake.mux %41 [%falseResult_16, %trueResult_13] : index, none loc(#loc47)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %42 : none loc(#loc39)
    %43 = handshake.mux %35 [%falseResult_18, %trueResult] : index, none loc(#loc39)
    %44 = handshake.join %36, %43 : none, none loc(#loc32)
    handshake.return %44 : none loc(#loc33)
  } loc(#loc32)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z13transpose_cpuPKjPjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc18]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc18]), %arg2: i32 loc(fused<#di_subprogram5>[#loc18]), %arg3: i32 loc(fused<#di_subprogram5>[#loc18])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc43)
    scf.if %0 {
    } else {
      %1 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc2)
      %2 = arith.extui %arg3 : i32 to i64 loc(#loc2)
      %3 = scf.while (%arg4 = %c0_i32) : (i32) -> i32 {
        scf.if %1 {
        } else {
          %6 = arith.muli %arg4, %arg3 : i32 loc(#loc2)
          %7 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
            %8 = arith.trunci %arg5 : i64 to i32 loc(#loc51)
            %9 = arith.addi %6, %8 : i32 loc(#loc51)
            %10 = arith.extui %9 : i32 to i64 loc(#loc51)
            %11 = arith.index_cast %10 : i64 to index loc(#loc51)
            %12 = memref.load %arg0[%11] : memref<?xi32, strided<[1], offset: ?>> loc(#loc51)
            %13 = arith.muli %arg2, %8 : i32 loc(#loc51)
            %14 = arith.addi %13, %arg4 : i32 loc(#loc51)
            %15 = arith.extui %14 : i32 to i64 loc(#loc51)
            %16 = arith.index_cast %15 : i64 to index loc(#loc51)
            memref.store %12, %arg1[%16] : memref<?xi32, strided<[1], offset: ?>> loc(#loc51)
            %17 = arith.addi %arg5, %c1_i64 : i64 loc(#loc49)
            %18 = arith.cmpi ne, %17, %2 : i64 loc(#loc52)
            scf.condition(%18) %17 : i64 loc(#loc48)
          } do {
          ^bb0(%arg5: i64 loc(fused<#di_lexical_block25>[#loc20])):
            scf.yield %arg5 : i64 loc(#loc48)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc48)
        } loc(#loc48)
        %4 = arith.addi %arg4, %c1_i32 : i32 loc(#loc43)
        %5 = arith.cmpi ne, %4, %arg2 : i32 loc(#loc46)
        scf.condition(%5) %4 : i32 loc(#loc40)
      } do {
      ^bb0(%arg4: i32 loc(fused<#di_lexical_block17>[#loc19])):
        scf.yield %arg4 : i32 loc(#loc40)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc40)
    } loc(#loc40)
    return loc(#loc35)
  } loc(#loc34)
} loc(#loc)
#loc = loc("tests/app/transpose/main.cpp":0:0)
#loc1 = loc("tests/app/transpose/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/transpose/main.cpp":18:0)
#loc5 = loc("tests/app/transpose/main.cpp":22:0)
#loc6 = loc("tests/app/transpose/main.cpp":25:0)
#loc8 = loc("tests/app/transpose/main.cpp":29:0)
#loc9 = loc("tests/app/transpose/main.cpp":30:0)
#loc10 = loc("tests/app/transpose/main.cpp":31:0)
#loc11 = loc("tests/app/transpose/main.cpp":35:0)
#loc12 = loc("tests/app/transpose/main.cpp":37:0)
#loc14 = loc("tests/app/transpose/transpose.cpp":31:0)
#loc15 = loc("tests/app/transpose/transpose.cpp":32:0)
#loc16 = loc("tests/app/transpose/transpose.cpp":33:0)
#loc17 = loc("tests/app/transpose/transpose.cpp":36:0)
#loc21 = loc("tests/app/transpose/transpose.cpp":19:0)
#loc22 = loc("tests/app/transpose/transpose.cpp":22:0)
#loc23 = loc(fused<#di_subprogram3>[#loc1])
#loc24 = loc(fused<#di_subprogram3>[#loc5])
#loc25 = loc(fused<#di_subprogram3>[#loc6])
#loc26 = loc(fused<#di_subprogram3>[#loc11])
#loc27 = loc(fused<#di_subprogram3>[#loc12])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 17>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 28>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 17>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 28>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 31>
#loc30 = loc(fused<#di_lexical_block12>[#loc3])
#loc31 = loc(fused<#di_lexical_block13>[#loc7])
#loc33 = loc(fused<#di_subprogram4>[#loc17])
#loc35 = loc(fused<#di_subprogram5>[#loc22])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 29>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 31>
#loc36 = loc(fused<#di_lexical_block14>[#loc4])
#loc37 = loc(fused[#loc28, #loc30])
#loc38 = loc(fused[#loc29, #loc31])
#loc39 = loc(fused<#di_lexical_block16>[#loc14])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 29>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 31>
#loc41 = loc(fused<#di_lexical_block18>[#loc8])
#loc42 = loc(fused<#di_lexical_block19>[#loc14])
#loc43 = loc(fused<#di_lexical_block20>[#loc19])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 32>
#loc44 = loc(fused<#di_lexical_block21>[#loc9])
#loc45 = loc(fused<#di_lexical_block21>[#loc10])
#loc46 = loc(fused[#loc40, #loc43])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 32>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 18>
#loc47 = loc(fused<#di_lexical_block24>[#loc15])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 32>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 18>
#loc49 = loc(fused<#di_lexical_block27>[#loc20])
#loc50 = loc(fused<#di_lexical_block28>[#loc16])
#loc51 = loc(fused<#di_lexical_block29>[#loc21])
#loc52 = loc(fused[#loc48, #loc49])
