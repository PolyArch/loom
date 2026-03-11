#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/parity/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/parity/parity.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc8 = loc("tests/app/parity/main.cpp":16:0)
#loc12 = loc("tests/app/parity/main.cpp":31:0)
#loc18 = loc("tests/app/parity/parity.cpp":32:0)
#loc25 = loc("tests/app/parity/parity.cpp":13:0)
#loc26 = loc("tests/app/parity/parity.cpp":16:0)
#loc28 = loc("tests/app/parity/parity.cpp":20:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 16>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 31>
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
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 16, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 31, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 37, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 16, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_parity", file = #di_file, line = 21, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_parity", file = #di_file, line = 22, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 34, arg = 3, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "parity", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 15, arg = 3, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file1, line = 17, type = #di_derived_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "parity", file = #di_file1, line = 18, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_parity", file = #di_file1, line = 33, arg = 2, type = #di_derived_type5>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_parity", file = #di_file1, line = 14, arg = 2, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable, #di_local_variable6, #di_local_variable7, #di_local_variable1>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 16>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 31>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_data", file = #di_file1, line = 32, arg = 1, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_data", file = #di_file1, line = 13, arg = 1, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "parity_dsa", linkageName = "_Z10parity_dsaPKjPjj", file = #di_file1, line = 32, scopeLine = 34, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable16, #di_local_variable14, #di_local_variable8, #di_local_variable2, #di_local_variable9, #di_local_variable10>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "parity_cpu", linkageName = "_Z10parity_cpuPKjPjj", file = #di_file1, line = 13, scopeLine = 15, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable17, #di_local_variable15, #di_local_variable11, #di_local_variable3, #di_local_variable12, #di_local_variable13>
#loc43 = loc(fused<#di_lexical_block8>[#loc8])
#loc44 = loc(fused<#di_lexical_block9>[#loc12])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 16>
#loc47 = loc(fused<#di_subprogram4>[#loc18])
#loc49 = loc(fused<#di_subprogram5>[#loc25])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 16>
#loc55 = loc(fused<#di_lexical_block15>[#loc26])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 16>
#loc65 = loc(fused<#di_lexical_block21>[#loc28])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<15xi8> = dense<[112, 97, 114, 105, 116, 121, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[112, 97, 114, 105, 116, 121, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 112, 97, 114, 105, 116, 121, 47, 112, 97, 114, 105, 116, 121, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc33)
    %false = arith.constant false loc(#loc33)
    %0 = seq.const_clock  low loc(#loc33)
    %c2_i32 = arith.constant 2 : i32 loc(#loc33)
    %1 = ub.poison : i64 loc(#loc33)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c4 = arith.constant 4 : index loc(#loc34)
    %c3 = arith.constant 3 : index loc(#loc35)
    %c2 = arith.constant 2 : index loc(#loc36)
    %c1 = arith.constant 1 : index loc(#loc37)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc2)
    %c5_i64 = arith.constant 5 : i64 loc(#loc2)
    %c-1698898192_i32 = arith.constant -1698898192 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    memref.store %c0_i32, %alloca[%c0] : memref<256xi32> loc(#loc38)
    memref.store %c1_i32, %alloca[%c1] : memref<256xi32> loc(#loc37)
    memref.store %c3_i32, %alloca[%c2] : memref<256xi32> loc(#loc36)
    memref.store %c7_i32, %alloca[%c3] : memref<256xi32> loc(#loc35)
    memref.store %c-1_i32, %alloca[%c4] : memref<256xi32> loc(#loc34)
    %4 = scf.while (%arg0 = %c5_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc51)
      %11 = arith.muli %10, %c-1698898192_i32 : i32 loc(#loc51)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc51)
      memref.store %11, %alloca[%12] : memref<256xi32> loc(#loc51)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
      %14 = arith.cmpi ne, %13, %c256_i64 : i64 loc(#loc52)
      scf.condition(%14) %13 : i64 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block8>[#loc8])):
      scf.yield %arg0 : i64 loc(#loc43)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc43)
    %cast = memref.cast %alloca : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %cast_2 = memref.cast %alloca_0 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    call @_Z10parity_cpuPKjPjj(%cast, %cast_2, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc39)
    %cast_3 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc40)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc40)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc40)
    %5 = handshake.esi_instance @_Z10parity_dsaPKjPjj_esi "_Z10parity_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc40)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc40)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc56)
      %11 = memref.load %alloca_0[%10] : memref<256xi32> loc(#loc56)
      %12 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc56)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc56)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc46)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc46)
        %18 = arith.extui %17 : i1 to i32 loc(#loc44)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc53)
        %20 = arith.extui %19 : i1 to i32 loc(#loc44)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc56)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc56)
      } loc(#loc56)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc44)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc44)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc12]), %arg1: i1 loc(fused<#di_lexical_block9>[#loc12]), %arg2: i32 loc(fused<#di_lexical_block9>[#loc12])):
      scf.yield %arg0 : i64 loc(#loc44)
    } loc(#loc44)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc44)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc44)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc59)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc59)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc59)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc59)
      scf.yield %c1_i32 : i32 loc(#loc60)
    } loc(#loc44)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc41)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc41)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc41)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc41)
    } loc(#loc2)
    return %9 : i32 loc(#loc42)
  } loc(#loc33)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10parity_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc18]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc18]), %arg2: i32 loc(fused<#di_subprogram4>[#loc18]), %arg3: i1 loc(fused<#di_subprogram4>[#loc18]), ...) -> i1 attributes {argNames = ["input_data", "output_parity", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc47)
    %1 = handshake.join %0 : none loc(#loc47)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc57)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc54)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc54)
    %7 = arith.index_cast %4 : i64 to index loc(#loc54)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc54)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc54)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc54)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc54)
    %dataResult, %addressResults = handshake.load [%afterValue] %22#0, %24 : index, i32 loc(#loc61)
    %10 = arith.cmpi eq, %dataResult, %3 : i32 loc(#loc62)
    %trueResult_0, %falseResult_1 = handshake.cond_br %10, %9 : none loc(#loc62)
    %11 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc62)
    %12 = arith.index_cast %dataResult : i32 to index loc(#loc62)
    %13 = arith.index_cast %3 : i32 to index loc(#loc62)
    %index_2, %willContinue_3 = dataflow.stream %12, %11, %13 {step_op = ">>=", stop_cond = "!="} loc(#loc62)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc62)
    %14 = dataflow.carry %willContinue_3, %3, %17 : i1, i32, i32 -> i32 loc(#loc62)
    %afterValue_6, %afterCond_7 = dataflow.gate %14, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc62)
    handshake.sink %afterCond_7 : i1 loc(#loc62)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %14 : i32 loc(#loc62)
    %15 = arith.index_cast %afterValue_4 : index to i32 loc(#loc62)
    %16 = arith.andi %15, %2 : i32 loc(#loc68)
    %17 = arith.xori %afterValue_6, %16 : i32 loc(#loc68)
    %18 = handshake.constant %9 {value = 0 : index} : index loc(#loc62)
    %19 = handshake.constant %9 {value = 1 : index} : index loc(#loc62)
    %20 = arith.select %10, %19, %18 : index loc(#loc62)
    %21 = handshake.mux %20 [%falseResult_9, %3] : index, i32 loc(#loc62)
    %dataResult_10, %addressResult = handshake.store [%afterValue] %21, %29 : index, i32 loc(#loc63)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc47)
    %23 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc47)
    %24 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc54)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %22#1 : none loc(#loc54)
    %25 = handshake.constant %1 {value = 0 : index} : index loc(#loc54)
    %26 = handshake.constant %1 {value = 1 : index} : index loc(#loc54)
    %27 = arith.select %5, %26, %25 : index loc(#loc54)
    %28 = handshake.mux %27 [%falseResult_12, %trueResult] : index, none loc(#loc54)
    %29 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc54)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %23 : none loc(#loc54)
    %30 = handshake.mux %27 [%falseResult_14, %trueResult] : index, none loc(#loc54)
    %31 = handshake.join %28, %30 : none, none loc(#loc47)
    %32 = handshake.constant %31 {value = true} : i1 loc(#loc47)
    handshake.return %32 : i1 loc(#loc47)
  } loc(#loc47)
  handshake.func @_Z10parity_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc18]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc18]), %arg2: i32 loc(fused<#di_subprogram4>[#loc18]), %arg3: none loc(fused<#di_subprogram4>[#loc18]), ...) -> none attributes {argNames = ["input_data", "output_parity", "N", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc47)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc57)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc54)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc54)
    %6 = arith.index_cast %3 : i64 to index loc(#loc54)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc54)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel=auto", "loom.loop.unroll factor=8"], step_op = "+=", stop_cond = "!="} loc(#loc54)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc54)
    %8 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc54)
    %dataResult, %addressResults = handshake.load [%afterValue] %21#0, %23 : index, i32 loc(#loc61)
    %9 = arith.cmpi eq, %dataResult, %2 : i32 loc(#loc62)
    %trueResult_0, %falseResult_1 = handshake.cond_br %9, %8 : none loc(#loc62)
    %10 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc62)
    %11 = arith.index_cast %dataResult : i32 to index loc(#loc62)
    %12 = arith.index_cast %2 : i32 to index loc(#loc62)
    %index_2, %willContinue_3 = dataflow.stream %11, %10, %12 {step_op = ">>=", stop_cond = "!="} loc(#loc62)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc62)
    %13 = dataflow.carry %willContinue_3, %2, %16 : i1, i32, i32 -> i32 loc(#loc62)
    %afterValue_6, %afterCond_7 = dataflow.gate %13, %willContinue_3 : i32, i1 -> i32, i1 loc(#loc62)
    handshake.sink %afterCond_7 : i1 loc(#loc62)
    %trueResult_8, %falseResult_9 = handshake.cond_br %willContinue_3, %13 : i32 loc(#loc62)
    %14 = arith.index_cast %afterValue_4 : index to i32 loc(#loc62)
    %15 = arith.andi %14, %1 : i32 loc(#loc68)
    %16 = arith.xori %afterValue_6, %15 : i32 loc(#loc68)
    %17 = handshake.constant %8 {value = 0 : index} : index loc(#loc62)
    %18 = handshake.constant %8 {value = 1 : index} : index loc(#loc62)
    %19 = arith.select %9, %18, %17 : index loc(#loc62)
    %20 = handshake.mux %19 [%falseResult_9, %2] : index, i32 loc(#loc62)
    %dataResult_10, %addressResult = handshake.store [%afterValue] %20, %28 : index, i32 loc(#loc63)
    %21:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc47)
    %22 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_10, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc47)
    %23 = dataflow.carry %willContinue, %falseResult, %trueResult_11 : i1, none, none -> none loc(#loc54)
    %trueResult_11, %falseResult_12 = handshake.cond_br %willContinue, %21#1 : none loc(#loc54)
    %24 = handshake.constant %0 {value = 0 : index} : index loc(#loc54)
    %25 = handshake.constant %0 {value = 1 : index} : index loc(#loc54)
    %26 = arith.select %4, %25, %24 : index loc(#loc54)
    %27 = handshake.mux %26 [%falseResult_12, %trueResult] : index, none loc(#loc54)
    %28 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc54)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %22 : none loc(#loc54)
    %29 = handshake.mux %26 [%falseResult_14, %trueResult] : index, none loc(#loc54)
    %30 = handshake.join %27, %29 : none, none loc(#loc47)
    handshake.return %30 : none loc(#loc48)
  } loc(#loc47)
  func.func @_Z10parity_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc25]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc25]), %arg2: i32 loc(fused<#di_subprogram5>[#loc25])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc58)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg2 : i32 to i64 loc(#loc58)
      %2 = scf.while (%arg3 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg3 : i64 to index loc(#loc64)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc64)
        %5 = arith.cmpi eq, %4, %c0_i32 : i32 loc(#loc65)
        %6 = scf.if %5 -> (i32) {
          scf.yield %c0_i32 : i32 loc(#loc65)
        } else {
          %9:2 = scf.while (%arg4 = %c0_i32, %arg5 = %4) : (i32, i32) -> (i32, i32) {
            %10 = arith.andi %arg5, %c1_i32 : i32 loc(#loc69)
            %11 = arith.xori %arg4, %10 : i32 loc(#loc69)
            %12 = arith.shrui %arg5, %c1_i32 : i32 loc(#loc70)
            %13 = arith.cmpi ne, %12, %c0_i32 : i32 loc(#loc65)
            scf.condition(%13) %11, %12 : i32, i32 loc(#loc65)
          } do {
          ^bb0(%arg4: i32 loc(fused<#di_lexical_block21>[#loc28]), %arg5: i32 loc(fused<#di_lexical_block21>[#loc28])):
            scf.yield %arg4, %arg5 : i32, i32 loc(#loc65)
          } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc65)
          scf.yield %9#0 : i32 loc(#loc65)
        } loc(#loc65)
        memref.store %6, %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc66)
        %7 = arith.addi %arg3, %c1_i64 : i64 loc(#loc58)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc67)
        scf.condition(%8) %7 : i64 loc(#loc55)
      } do {
      ^bb0(%arg3: i64 loc(fused<#di_lexical_block15>[#loc26])):
        scf.yield %arg3 : i64 loc(#loc55)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc55)
    } loc(#loc55)
    return loc(#loc50)
  } loc(#loc49)
} loc(#loc)
#loc = loc("tests/app/parity/main.cpp":0:0)
#loc1 = loc("tests/app/parity/main.cpp":5:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/parity/main.cpp":14:0)
#loc4 = loc("tests/app/parity/main.cpp":13:0)
#loc5 = loc("tests/app/parity/main.cpp":12:0)
#loc6 = loc("tests/app/parity/main.cpp":11:0)
#loc7 = loc("tests/app/parity/main.cpp":10:0)
#loc9 = loc("tests/app/parity/main.cpp":17:0)
#loc10 = loc("tests/app/parity/main.cpp":25:0)
#loc11 = loc("tests/app/parity/main.cpp":28:0)
#loc13 = loc("tests/app/parity/main.cpp":32:0)
#loc14 = loc("tests/app/parity/main.cpp":33:0)
#loc15 = loc("tests/app/parity/main.cpp":34:0)
#loc16 = loc("tests/app/parity/main.cpp":38:0)
#loc17 = loc("tests/app/parity/main.cpp":40:0)
#loc19 = loc("tests/app/parity/parity.cpp":37:0)
#loc20 = loc("tests/app/parity/parity.cpp":38:0)
#loc21 = loc("tests/app/parity/parity.cpp":41:0)
#loc22 = loc("tests/app/parity/parity.cpp":42:0)
#loc23 = loc("tests/app/parity/parity.cpp":46:0)
#loc24 = loc("tests/app/parity/parity.cpp":48:0)
#loc27 = loc("tests/app/parity/parity.cpp":17:0)
#loc29 = loc("tests/app/parity/parity.cpp":21:0)
#loc30 = loc("tests/app/parity/parity.cpp":22:0)
#loc31 = loc("tests/app/parity/parity.cpp":25:0)
#loc32 = loc("tests/app/parity/parity.cpp":27:0)
#loc33 = loc(fused<#di_subprogram3>[#loc1])
#loc34 = loc(fused<#di_subprogram3>[#loc3])
#loc35 = loc(fused<#di_subprogram3>[#loc4])
#loc36 = loc(fused<#di_subprogram3>[#loc5])
#loc37 = loc(fused<#di_subprogram3>[#loc6])
#loc38 = loc(fused<#di_subprogram3>[#loc7])
#loc39 = loc(fused<#di_subprogram3>[#loc10])
#loc40 = loc(fused<#di_subprogram3>[#loc11])
#loc41 = loc(fused<#di_subprogram3>[#loc16])
#loc42 = loc(fused<#di_subprogram3>[#loc17])
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 16>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 31>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 16>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 31>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 37>
#loc45 = loc(fused<#di_lexical_block10>[#loc8])
#loc46 = loc(fused<#di_lexical_block11>[#loc12])
#loc48 = loc(fused<#di_subprogram4>[#loc24])
#loc50 = loc(fused<#di_subprogram5>[#loc32])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 32>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 37>
#loc51 = loc(fused<#di_lexical_block12>[#loc9])
#loc52 = loc(fused[#loc43, #loc45])
#loc53 = loc(fused[#loc44, #loc46])
#loc54 = loc(fused<#di_lexical_block14>[#loc19])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 32>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 37>
#loc56 = loc(fused<#di_lexical_block16>[#loc13])
#loc57 = loc(fused<#di_lexical_block17>[#loc19])
#loc58 = loc(fused<#di_lexical_block18>[#loc26])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 41>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 20>
#loc59 = loc(fused<#di_lexical_block19>[#loc14])
#loc60 = loc(fused<#di_lexical_block19>[#loc15])
#loc61 = loc(fused<#di_lexical_block20>[#loc20])
#loc62 = loc(fused<#di_lexical_block20>[#loc21])
#loc63 = loc(fused<#di_lexical_block20>[#loc23])
#loc64 = loc(fused<#di_lexical_block21>[#loc27])
#loc66 = loc(fused<#di_lexical_block21>[#loc31])
#loc67 = loc(fused[#loc55, #loc58])
#loc68 = loc(fused<#di_lexical_block22>[#loc22])
#loc69 = loc(fused<#di_lexical_block23>[#loc29])
#loc70 = loc(fused<#di_lexical_block23>[#loc30])
