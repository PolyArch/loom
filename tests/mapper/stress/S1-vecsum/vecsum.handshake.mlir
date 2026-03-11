#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file1 = #llvm.di_file<"tests/app/vecsum/vecsum.cpp" in "/home/sihao/github.com/PolyArch/loom">
#loc11 = loc("tests/app/vecsum/vecsum.cpp":26:0)
#loc15 = loc("tests/app/vecsum/vecsum.cpp":12:0)
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = false, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_derived_type1, #di_derived_type4, #di_derived_type2, #di_derived_type2>
#di_subprogram1 = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecsum_dsa", linkageName = "_Z10vecsum_dsaPKjjj", file = #di_file1, line = 26, scopeLine = 28, subprogramFlags = Definition, type = #di_subroutine_type1>
#di_subprogram2 = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "vecsum_cpu", linkageName = "_Z10vecsum_cpuPKjjj", file = #di_file1, line = 12, scopeLine = 14, subprogramFlags = Definition, type = #di_subroutine_type1>
#loc28 = loc(fused<#di_subprogram1>[#loc11])
#loc30 = loc(fused<#di_subprogram2>[#loc15])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<16xi8> = dense<[118, 101, 99, 115, 117, 109, 58, 32, 70, 65, 73, 76, 69, 68, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.1" : memref<16xi8> = dense<[118, 101, 99, 115, 117, 109, 58, 32, 80, 65, 83, 83, 69, 68, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.5" : memref<14xi8> = dense<[108, 111, 111, 109, 46, 114, 101, 100, 117, 99, 101, 61, 43, 0]> loc(#loc)
  memref.global constant @".str.1.4" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 118, 101, 99, 115, 117, 109, 47, 118, 101, 99, 115, 117, 109, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<21xi8> = dense<[108, 111, 111, 109, 46, 116, 97, 114, 103, 101, 116, 61, 116, 101, 109, 112, 111, 114, 97, 108, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc19)
    %false = arith.constant false loc(#loc19)
    %0 = seq.const_clock  low loc(#loc19)
    %c1 = arith.constant 1 : index loc(#loc25)
    %c1024 = arith.constant 1024 : index loc(#loc25)
    %c0 = arith.constant 0 : index loc(#loc25)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc3)
    %c100_i32 = arith.constant 100 : i32 loc(#loc3)
    %1 = memref.get_global @".str.1" : memref<16xi8> loc(#loc3)
    %2 = memref.get_global @".str" : memref<16xi8> loc(#loc3)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc3)
    scf.for unsigned %arg0 = %c0 to %c1024 step %c1 {
      %7 = arith.index_cast %arg0 : index to i32 loc(#loc25)
      %8 = arith.extui %7 : i32 to i64 loc(#loc27)
      %9 = arith.index_cast %8 : i64 to index loc(#loc27)
      memref.store %7, %alloca[%9] : memref<1024xi32> loc(#loc27)
    } loc(#loc25)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc20)
    %3 = call @_Z10vecsum_cpuPKjjj(%cast, %c100_i32, %c1024_i32) : (memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> i32 loc(#loc20)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc21)
    %chanOutput_0, %ready_1 = esi.wrap.vr %c100_i32, %true : i32 loc(#loc21)
    %chanOutput_2, %ready_3 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc21)
    %chanOutput_4, %ready_5 = esi.wrap.vr %true, %true : i1 loc(#loc21)
    %4:2 = handshake.esi_instance @_Z10vecsum_dsaPKjjj_esi "_Z10vecsum_dsaPKjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_0, %chanOutput_2, %chanOutput_4) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> (!esi.channel<i32>, !esi.channel<i1>) loc(#loc21)
    %rawOutput, %valid = esi.unwrap.vr %4#0, %true : i32 loc(#loc21)
    %rawOutput_6, %valid_7 = esi.unwrap.vr %4#1, %true : i1 loc(#loc21)
    %5 = arith.cmpi ne, %3, %rawOutput : i32 loc(#loc24)
    %6 = arith.extui %5 : i1 to i32 loc(#loc24)
    scf.if %5 {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<16xi8> -> index loc(#loc26)
      %7 = arith.index_cast %intptr : index to i64 loc(#loc26)
      %8 = llvm.inttoptr %7 : i64 to !llvm.ptr loc(#loc26)
      %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc26)
    } else {
      %intptr = memref.extract_aligned_pointer_as_index %1 : memref<16xi8> -> index loc(#loc22)
      %7 = arith.index_cast %intptr : index to i64 loc(#loc22)
      %8 = llvm.inttoptr %7 : i64 to !llvm.ptr loc(#loc22)
      %9 = llvm.call @printf(%8) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc22)
    } loc(#loc24)
    return %6 : i32 loc(#loc23)
  } loc(#loc19)
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc3)
  handshake.func @_Z10vecsum_dsaPKjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram1>[#loc11]), %arg1: i32 loc(fused<#di_subprogram1>[#loc11]), %arg2: i32 loc(fused<#di_subprogram1>[#loc11]), %arg3: i1 loc(fused<#di_subprogram1>[#loc11]), ...) -> (i32, i1) attributes {argNames = ["A", "init_value", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["sum", "done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc28)
    %1 = handshake.join %0 : none loc(#loc28)
    %2 = handshake.constant %1 {value = 1 : index} : index loc(#loc32)
    %3 = handshake.constant %1 {value = 0 : index} : index loc(#loc32)
    %4 = arith.index_cast %arg2 : i32 to index loc(#loc32)
    %index, %willContinue = dataflow.stream %3, %2, %4 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"]} loc(#loc32)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc32)
    %5 = dataflow.carry %willContinue, %arg1, %9 : i1, i32, i32 -> i32 loc(#loc32)
    %afterValue_0, %afterCond_1 = dataflow.gate %5, %willContinue : i32, i1 -> i32, i1 loc(#loc32)
    handshake.sink %afterCond_1 : i1 loc(#loc32)
    %trueResult, %falseResult = handshake.cond_br %willContinue, %5 : i32 loc(#loc32)
    %6 = arith.index_cast %afterValue : index to i32 loc(#loc32)
    %7 = arith.extui %6 : i32 to i64 loc(#loc34)
    %8 = arith.index_cast %7 : i64 to index loc(#loc34)
    %dataResult, %addressResults = handshake.load [%8] %10#0, %trueResult_2 : index, i32 loc(#loc34)
    %9 = arith.addi %afterValue_0, %dataResult : i32 loc(#loc34)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc28)
    %11 = dataflow.carry %willContinue, %1, %10#1 : i1, none, none -> none loc(#loc32)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %11 : none loc(#loc32)
    %12 = handshake.constant %falseResult_3 {value = true} : i1 loc(#loc28)
    handshake.return %falseResult, %12 : i32, i1 loc(#loc28)
  } loc(#loc28)
  handshake.func @_Z10vecsum_dsaPKjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram1>[#loc11]), %arg1: i32 loc(fused<#di_subprogram1>[#loc11]), %arg2: i32 loc(fused<#di_subprogram1>[#loc11]), %arg3: none loc(fused<#di_subprogram1>[#loc11]), ...) -> (i32, none) attributes {argNames = ["A", "init_value", "N", "start_token"], loom.annotations = ["loom.target=temporal", "loom.accel"], resNames = ["sum", "done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc28)
    %1 = handshake.constant %0 {value = 1 : index} : index loc(#loc32)
    %2 = handshake.constant %0 {value = 0 : index} : index loc(#loc32)
    %3 = arith.index_cast %arg2 : i32 to index loc(#loc32)
    %index, %willContinue = dataflow.stream %2, %1, %3 {loom.annotations = ["loom.loop.parallel degree=4 schedule=0"]} loc(#loc32)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc32)
    %4 = dataflow.carry %willContinue, %arg1, %8 : i1, i32, i32 -> i32 loc(#loc32)
    %afterValue_0, %afterCond_1 = dataflow.gate %4, %willContinue : i32, i1 -> i32, i1 loc(#loc32)
    handshake.sink %afterCond_1 : i1 loc(#loc32)
    %trueResult, %falseResult = handshake.cond_br %willContinue, %4 : i32 loc(#loc32)
    %5 = arith.index_cast %afterValue : index to i32 loc(#loc32)
    %6 = arith.extui %5 : i32 to i64 loc(#loc34)
    %7 = arith.index_cast %6 : i64 to index loc(#loc34)
    %dataResult, %addressResults = handshake.load [%7] %9#0, %trueResult_2 : index, i32 loc(#loc34)
    %8 = arith.addi %afterValue_0, %dataResult : i32 loc(#loc34)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc28)
    %10 = dataflow.carry %willContinue, %0, %9#1 : i1, none, none -> none loc(#loc32)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %10 : none loc(#loc32)
    handshake.return %falseResult, %falseResult_3 : i32, none loc(#loc29)
  } loc(#loc28)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc3)
  func.func @_Z10vecsum_cpuPKjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram2>[#loc15]), %arg1: i32 loc(fused<#di_subprogram2>[#loc15]), %arg2: i32 loc(fused<#di_subprogram2>[#loc15])) -> i32 {
    %c1 = arith.constant 1 : index loc(#loc33)
    %c0 = arith.constant 0 : index loc(#loc33)
    %0 = arith.index_cast %arg2 : i32 to index loc(#loc33)
    %1 = scf.for unsigned %arg3 = %c0 to %0 step %c1 iter_args(%arg4 = %arg1) -> (i32) {
      %2 = arith.index_cast %arg3 : index to i32 loc(#loc33)
      %3 = arith.extui %2 : i32 to i64 loc(#loc35)
      %4 = arith.index_cast %3 : i64 to index loc(#loc35)
      %5 = memref.load %arg0[%4] : memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
      %6 = arith.addi %arg4, %5 : i32 loc(#loc35)
      scf.yield %6 : i32 loc(#loc33)
    } loc(#loc33)
    return %1 : i32 loc(#loc31)
  } loc(#loc30)
} loc(#loc)
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/vecsum/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#loc = loc("tests/app/vecsum/main.cpp":0:0)
#loc1 = loc("tests/app/vecsum/main.cpp":5:0)
#loc2 = loc("tests/app/vecsum/main.cpp":13:0)
#loc3 = loc(unknown)
#loc4 = loc("tests/app/vecsum/main.cpp":14:0)
#loc5 = loc("tests/app/vecsum/main.cpp":18:0)
#loc6 = loc("tests/app/vecsum/main.cpp":21:0)
#loc7 = loc("tests/app/vecsum/main.cpp":24:0)
#loc8 = loc("tests/app/vecsum/main.cpp":25:0)
#loc9 = loc("tests/app/vecsum/main.cpp":29:0)
#loc10 = loc("tests/app/vecsum/main.cpp":31:0)
#loc12 = loc("tests/app/vecsum/vecsum.cpp":32:0)
#loc13 = loc("tests/app/vecsum/vecsum.cpp":33:0)
#loc14 = loc("tests/app/vecsum/vecsum.cpp":35:0)
#loc16 = loc("tests/app/vecsum/vecsum.cpp":16:0)
#loc17 = loc("tests/app/vecsum/vecsum.cpp":17:0)
#loc18 = loc("tests/app/vecsum/vecsum.cpp":19:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = false, emissionKind = Full, nameTableKind = None>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[4]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = Definition, type = #di_subroutine_type>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 13>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 24>
#loc19 = loc(fused<#di_subprogram>[#loc1])
#loc20 = loc(fused<#di_subprogram>[#loc5])
#loc21 = loc(fused<#di_subprogram>[#loc6])
#loc22 = loc(fused<#di_subprogram>[#loc9])
#loc23 = loc(fused<#di_subprogram>[#loc10])
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 13>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 24>
#loc24 = loc(fused<#di_lexical_block1>[#loc7])
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file, line = 13>
#loc25 = loc(fused<#di_lexical_block2>[#loc2])
#loc26 = loc(fused<#di_lexical_block3>[#loc8])
#loc27 = loc(fused<#di_lexical_block4>[#loc4])
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 32>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 16>
#loc29 = loc(fused<#di_subprogram1>[#loc14])
#loc31 = loc(fused<#di_subprogram2>[#loc18])
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 32>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 16>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 32>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file1, line = 16>
#loc32 = loc(fused<#di_lexical_block7>[#loc12])
#loc33 = loc(fused<#di_lexical_block8>[#loc16])
#loc34 = loc(fused<#di_lexical_block9>[#loc13])
#loc35 = loc(fused<#di_lexical_block10>[#loc17])
