#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/stream_update/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file2 = #llvm.di_file<"tests/app/stream_update/stream_update.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/stream_update/main.cpp":10:0)
#loc15 = loc("tests/app/stream_update/stream_update.cpp":20:0)
#loc22 = loc("tests/app/stream_update/stream_update.cpp":6:0)
#loc23 = loc("tests/app/stream_update/stream_update.cpp":10:0)
#loc24 = loc("tests/app/stream_update/stream_update.cpp":11:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file2, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_label = #llvm.di_label<scope = #di_subprogram1, name = "outer_loop", file = #di_file2, line = 24>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 10>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file2, line = 27>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file2, line = 10>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "passed", file = #di_file, line = 19, type = #di_basic_type2>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file2, line = 27>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file2, line = 10>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 1024, elements = #llvm.di_subrange<count = 32 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 32, elements = #llvm.di_subrange<count = 1 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file2, line = 27>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file2, line = 10>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 10, type = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "n", file = #di_file2, line = 22, arg = 3, type = #di_derived_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram1, name = "step", file = #di_file2, line = 22, arg = 4, type = #di_derived_type2>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram1, name = "acc", file = #di_file2, line = 23, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file2, line = 27, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "n", file = #di_file2, line = 8, arg = 3, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram2, name = "step", file = #di_file2, line = 8, arg = 4, type = #di_derived_type2>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram2, name = "acc", file = #di_file2, line = 9, type = #di_derived_type2>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file2, line = 10, type = #di_derived_type2>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type6>
#di_label1 = #llvm.di_label<scope = #di_lexical_block5, name = "inner_loop", file = #di_file2, line = 28>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file2, line = 31>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file2, line = 11>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "n", file = #di_file, line = 7, type = #di_derived_type4>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "step", file = #di_file, line = 8, type = #di_derived_type4>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "cpu_out", file = #di_file, line = 13, type = #di_composite_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "dsa_out", file = #di_file, line = 14, type = #di_composite_type1>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type7>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file2, line = 31>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file2, line = 11>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file2, line = 21, arg = 2, type = #di_derived_type8>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "j", file = #di_file2, line = 31, type = #di_derived_type2>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output", file = #di_file2, line = 7, arg = 2, type = #di_derived_type8>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file2, line = 11, type = #di_derived_type2>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable10, #di_local_variable11, #di_local_variable12, #di_local_variable1, #di_local_variable13, #di_local_variable14, #di_local_variable>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 10>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file2, line = 31>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file2, line = 11>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file2, line = 20, arg = 1, type = #di_derived_type9>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file2, line = 6, arg = 1, type = #di_derived_type9>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type9, #di_derived_type8, #di_derived_type2, #di_derived_type2>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block13, name = "idx", file = #di_file2, line = 32, type = #di_derived_type2>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_lexical_block14, name = "idx", file = #di_file2, line = 12, type = #di_derived_type2>
#loc38 = loc(fused<#di_lexical_block11>[#loc3])
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file2, name = "stream_update_dsa", linkageName = "_Z17stream_update_dsaPKjPjjj", file = #di_file2, line = 20, scopeLine = 22, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable19, #di_local_variable15, #di_local_variable2, #di_local_variable3, #di_local_variable4, #di_label, #di_local_variable5, #di_label1, #di_local_variable16, #di_local_variable21>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file2, name = "stream_update_cpu", linkageName = "_Z17stream_update_cpuPKjPjjj", file = #di_file2, line = 6, scopeLine = 8, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable20, #di_local_variable17, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable18, #di_local_variable22>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file2, line = 10>
#loc45 = loc(fused<#di_subprogram5>[#loc15])
#loc48 = loc(fused<#di_subprogram6>[#loc22])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file2, line = 10>
#loc52 = loc(fused<#di_lexical_block18>[#loc23])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file2, line = 10>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file2, line = 11>
#loc56 = loc(fused<#di_lexical_block24>[#loc24])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<23xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 99, 112, 117, 61, 37, 117, 32, 100, 115, 97, 61, 37, 117, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.1" : memref<33xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 115, 116, 114, 101, 97, 109, 95, 117, 112, 100, 97, 116, 101, 32, 114, 101, 115, 117, 108, 116, 61, 37, 117, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 115, 116, 114, 101, 97, 109, 95, 117, 112, 100, 97, 116, 101, 0]> loc(#loc)
  memref.global constant @".str.1.5" : memref<42xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 115, 116, 114, 101, 97, 109, 95, 117, 112, 100, 97, 116, 101, 47, 115, 116, 114, 101, 97, 109, 95, 117, 112, 100, 97, 116, 101, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc29)
    %false = arith.constant false loc(#loc29)
    %0 = seq.const_clock  low loc(#loc29)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c32_i64 = arith.constant 32 : i64 loc(#loc2)
    %c32_i32 = arith.constant 32 : i32 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %1 = memref.get_global @".str" : memref<23xi8> loc(#loc2)
    %2 = memref.get_global @".str.1" : memref<33xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<32xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1xi32> loc(#loc2)
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc40)
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc40)
      %12 = arith.trunci %10 : i64 to i32 loc(#loc40)
      %13 = arith.shli %12, %c1_i32 : i32 loc(#loc40)
      memref.store %13, %alloca[%11] : memref<32xi32> loc(#loc40)
      %14 = arith.cmpi ne, %10, %c32_i64 : i64 loc(#loc44)
      scf.condition(%14) %10 : i64 loc(#loc38)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc38)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc38)
    memref.store %c0_i32, %alloca_0[%c0] : memref<1xi32> loc(#loc30)
    memref.store %c0_i32, %alloca_1[%c0] : memref<1xi32> loc(#loc31)
    %cast = memref.cast %alloca : memref<32xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc32)
    %cast_2 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc32)
    call @_Z17stream_update_cpuPKjPjjj(%cast, %cast_2, %c32_i32, %c3_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc32)
    %cast_3 = memref.cast %alloca_1 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc33)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c32_i32, %true : i32 loc(#loc33)
    %chanOutput_8, %ready_9 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc33)
    %chanOutput_10, %ready_11 = esi.wrap.vr %true, %true : i1 loc(#loc33)
    %4 = handshake.esi_instance @_Z17stream_update_dsaPKjPjjj_esi "_Z17stream_update_dsaPKjPjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8, %chanOutput_10) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc33)
    %rawOutput, %valid = esi.unwrap.vr %4, %true : i1 loc(#loc33)
    %5 = memref.load %alloca_0[%c0] : memref<1xi32> loc(#loc34)
    %6 = memref.load %alloca_1[%c0] : memref<1xi32> loc(#loc34)
    %7 = arith.cmpi eq, %5, %6 : i32 loc(#loc34)
    %8 = arith.cmpi ne, %5, %6 : i32 loc(#loc42)
    %9 = arith.extui %8 : i1 to i32 loc(#loc39)
    scf.if %7 {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<33xi8> -> index loc(#loc35)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc35)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc35)
      %12 = llvm.call @printf(%11, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32 loc(#loc35)
    } else {
      %intptr = memref.extract_aligned_pointer_as_index %1 : memref<23xi8> -> index loc(#loc43)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc43)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc43)
      %12 = llvm.call @printf(%11, %5, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32 loc(#loc43)
    } loc(#loc39)
    return %9 : i32 loc(#loc36)
  } loc(#loc29)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc37)
  handshake.func @_Z17stream_update_dsaPKjPjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: i32 loc(fused<#di_subprogram5>[#loc15]), %arg3: i32 loc(fused<#di_subprogram5>[#loc15]), %arg4: i1 loc(fused<#di_subprogram5>[#loc15]), ...) -> i1 attributes {argNames = ["input", "output", "n", "step", "start_token"], loom.annotations = ["loom.accel=stream_update"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc45)
    %1 = handshake.join %0 : none loc(#loc45)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : index} : index loc(#loc51)
    %4 = arith.cmpi ult, %arg3, %arg2 : i32 loc(#loc53)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc51)
    handshake.sink %falseResult : none loc(#loc51)
    %5 = arith.index_cast %arg3 : i32 to index loc(#loc51)
    %6 = arith.index_cast %arg2 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %5, %5, %6 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=32", "loom.loop.parallel degree=2 schedule=1"], step_op = "+=", stop_cond = "<"} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %7 = dataflow.carry %willContinue, %2, %falseResult_13 : i1, i32, i32 -> i32 loc(#loc51)
    %afterValue_0, %afterCond_1 = dataflow.gate %7, %willContinue : i32, i1 -> i32, i1 loc(#loc51)
    handshake.sink %afterCond_1 : i1 loc(#loc51)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %7 : i32 loc(#loc51)
    %8 = dataflow.carry %willContinue, %2, %10 : i1, i32, i32 -> i32 loc(#loc51)
    %afterValue_4, %afterCond_5 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc51)
    handshake.sink %afterCond_5 : i1 loc(#loc51)
    %9 = dataflow.invariant %afterCond, %trueResult : i1, none -> none loc(#loc51)
    %10 = arith.index_cast %afterValue : index to i32 loc(#loc51)
    %11 = handshake.constant %9 {value = 1 : index} : index loc(#loc55)
    %12 = arith.index_cast %2 : i32 to index loc(#loc55)
    %index_6, %willContinue_7 = dataflow.stream %6, %11, %12 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = ">>=", stop_cond = "!="} loc(#loc55)
    %afterValue_8, %afterCond_9 = dataflow.gate %index_6, %willContinue_7 : index, i1 -> index, i1 loc(#loc55)
    %13 = dataflow.carry %willContinue_7, %afterValue_0, %20 : i1, i32, i32 -> i32 loc(#loc55)
    %afterValue_10, %afterCond_11 = dataflow.gate %13, %willContinue_7 : i32, i1 -> i32, i1 loc(#loc55)
    handshake.sink %afterCond_11 : i1 loc(#loc55)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_7, %13 : i32 loc(#loc55)
    %14 = arith.index_cast %afterValue_8 : index to i32 loc(#loc55)
    %15 = dataflow.invariant %afterCond_9, %afterValue_4 : i1, i32 -> i32 loc(#loc58)
    %16 = arith.addi %14, %15 : i32 loc(#loc58)
    %17 = arith.remui %16, %arg2 : i32 loc(#loc58)
    %18 = arith.extui %17 : i32 to i64 loc(#loc59)
    %19 = arith.index_cast %18 : i64 to index loc(#loc59)
    %dataResult, %addressResults = handshake.load [%19] %24#0, %27 : index, i32 loc(#loc59)
    %20 = arith.addi %dataResult, %afterValue_10 : i32 loc(#loc59)
    %21 = handshake.constant %1 {value = 1 : index} : index loc(#loc51)
    %22 = arith.select %4, %21, %3 : index loc(#loc51)
    %23 = handshake.mux %22 [%2, %falseResult_3] : index, i32 loc(#loc51)
    %dataResult_14, %addressResult = handshake.store [%3] %23, %1 : index, i32 loc(#loc46)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc45)
    %26 = dataflow.carry %willContinue, %trueResult, %trueResult_17 : i1, none, none -> none loc(#loc51)
    %27 = dataflow.carry %willContinue_7, %26, %trueResult_15 : i1, none, none -> none loc(#loc55)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_7, %24#1 : none loc(#loc55)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %falseResult_16 : none loc(#loc51)
    %28 = handshake.mux %22 [%falseResult, %falseResult_18] : index, none loc(#loc51)
    %29 = handshake.join %28, %25 : none, none loc(#loc45)
    %30 = handshake.constant %29 {value = true} : i1 loc(#loc45)
    handshake.return %30 : i1 loc(#loc45)
  } loc(#loc45)
  handshake.func @_Z17stream_update_dsaPKjPjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: i32 loc(fused<#di_subprogram5>[#loc15]), %arg3: i32 loc(fused<#di_subprogram5>[#loc15]), %arg4: none loc(fused<#di_subprogram5>[#loc15]), ...) -> none attributes {argNames = ["input", "output", "n", "step", "start_token"], loom.annotations = ["loom.accel=stream_update"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc45)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : index} : index loc(#loc51)
    %3 = arith.cmpi ult, %arg3, %arg2 : i32 loc(#loc53)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc51)
    handshake.sink %falseResult : none loc(#loc51)
    %4 = arith.index_cast %arg3 : i32 to index loc(#loc51)
    %5 = arith.index_cast %arg2 : i32 to index loc(#loc51)
    %index, %willContinue = dataflow.stream %4, %4, %5 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=32", "loom.loop.parallel degree=2 schedule=1"], step_op = "+=", stop_cond = "<"} loc(#loc51)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc51)
    %6 = dataflow.carry %willContinue, %1, %falseResult_13 : i1, i32, i32 -> i32 loc(#loc51)
    %afterValue_0, %afterCond_1 = dataflow.gate %6, %willContinue : i32, i1 -> i32, i1 loc(#loc51)
    handshake.sink %afterCond_1 : i1 loc(#loc51)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %6 : i32 loc(#loc51)
    %7 = dataflow.carry %willContinue, %1, %9 : i1, i32, i32 -> i32 loc(#loc51)
    %afterValue_4, %afterCond_5 = dataflow.gate %7, %willContinue : i32, i1 -> i32, i1 loc(#loc51)
    handshake.sink %afterCond_5 : i1 loc(#loc51)
    %8 = dataflow.invariant %afterCond, %trueResult : i1, none -> none loc(#loc51)
    %9 = arith.index_cast %afterValue : index to i32 loc(#loc51)
    %10 = handshake.constant %8 {value = 1 : index} : index loc(#loc55)
    %11 = arith.index_cast %1 : i32 to index loc(#loc55)
    %index_6, %willContinue_7 = dataflow.stream %5, %10, %11 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = ">>=", stop_cond = "!="} loc(#loc55)
    %afterValue_8, %afterCond_9 = dataflow.gate %index_6, %willContinue_7 : index, i1 -> index, i1 loc(#loc55)
    %12 = dataflow.carry %willContinue_7, %afterValue_0, %19 : i1, i32, i32 -> i32 loc(#loc55)
    %afterValue_10, %afterCond_11 = dataflow.gate %12, %willContinue_7 : i32, i1 -> i32, i1 loc(#loc55)
    handshake.sink %afterCond_11 : i1 loc(#loc55)
    %trueResult_12, %falseResult_13 = handshake.cond_br %willContinue_7, %12 : i32 loc(#loc55)
    %13 = arith.index_cast %afterValue_8 : index to i32 loc(#loc55)
    %14 = dataflow.invariant %afterCond_9, %afterValue_4 : i1, i32 -> i32 loc(#loc58)
    %15 = arith.addi %13, %14 : i32 loc(#loc58)
    %16 = arith.remui %15, %arg2 : i32 loc(#loc58)
    %17 = arith.extui %16 : i32 to i64 loc(#loc59)
    %18 = arith.index_cast %17 : i64 to index loc(#loc59)
    %dataResult, %addressResults = handshake.load [%18] %23#0, %26 : index, i32 loc(#loc59)
    %19 = arith.addi %dataResult, %afterValue_10 : i32 loc(#loc59)
    %20 = handshake.constant %0 {value = 1 : index} : index loc(#loc51)
    %21 = arith.select %3, %20, %2 : index loc(#loc51)
    %22 = handshake.mux %21 [%1, %falseResult_3] : index, i32 loc(#loc51)
    %dataResult_14, %addressResult = handshake.store [%2] %22, %0 : index, i32 loc(#loc46)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc45)
    %24 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_14, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc45)
    %25 = dataflow.carry %willContinue, %trueResult, %trueResult_17 : i1, none, none -> none loc(#loc51)
    %26 = dataflow.carry %willContinue_7, %25, %trueResult_15 : i1, none, none -> none loc(#loc55)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue_7, %23#1 : none loc(#loc55)
    %trueResult_17, %falseResult_18 = handshake.cond_br %willContinue, %falseResult_16 : none loc(#loc51)
    %27 = handshake.mux %21 [%falseResult, %falseResult_18] : index, none loc(#loc51)
    %28 = handshake.join %27, %24 : none, none loc(#loc45)
    handshake.return %28 : none loc(#loc47)
  } loc(#loc45)
  func.func @_Z17stream_update_cpuPKjPjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram6>[#loc22]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram6>[#loc22]), %arg2: i32 loc(fused<#di_subprogram6>[#loc22]), %arg3: i32 loc(fused<#di_subprogram6>[#loc22])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %0 = arith.cmpi ult, %arg3, %arg2 : i32 loc(#loc54)
    %1 = scf.if %0 -> (i32) {
      %2:3 = scf.while (%arg4 = %arg3, %arg5 = %c0_i32, %arg6 = %c0_i32) : (i32, i32, i32) -> (i32, i32, i32) {
        %3:2 = scf.while (%arg7 = %arg5, %arg8 = %arg2) : (i32, i32) -> (i32, i32) {
          %6 = arith.addi %arg8, %arg6 : i32 loc(#loc60)
          %7 = arith.remui %6, %arg2 : i32 loc(#loc60)
          %8 = arith.extui %7 : i32 to i64 loc(#loc61)
          %9 = arith.index_cast %8 : i64 to index loc(#loc61)
          %10 = memref.load %arg0[%9] : memref<?xi32, strided<[1], offset: ?>> loc(#loc61)
          %11 = arith.addi %10, %arg7 : i32 loc(#loc61)
          %12 = arith.shrui %arg8, %c1_i32 : i32 loc(#loc57)
          %13 = arith.cmpi ne, %12, %c0_i32 : i32 loc(#loc62)
          scf.condition(%13) %11, %12 : i32, i32 loc(#loc56)
        } do {
        ^bb0(%arg7: i32 loc(fused<#di_lexical_block24>[#loc24]), %arg8: i32 loc(fused<#di_lexical_block24>[#loc24])):
          scf.yield %arg7, %arg8 : i32, i32 loc(#loc56)
        } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc56)
        %4 = arith.addi %arg4, %arg3 : i32 loc(#loc54)
        %5 = arith.cmpi ult, %4, %arg2 : i32 loc(#loc54)
        scf.condition(%5) %4, %3#0, %arg4 : i32, i32, i32 loc(#loc52)
      } do {
      ^bb0(%arg4: i32 loc(fused<#di_lexical_block18>[#loc23]), %arg5: i32 loc(fused<#di_lexical_block18>[#loc23]), %arg6: i32 loc(fused<#di_lexical_block18>[#loc23])):
        scf.yield %arg4, %arg5, %arg6 : i32, i32, i32 loc(#loc52)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc52)
      scf.yield %2#1 : i32 loc(#loc52)
    } else {
      scf.yield %c0_i32 : i32 loc(#loc52)
    } loc(#loc52)
    memref.store %1, %arg1[%c0] : memref<?xi32, strided<[1], offset: ?>> loc(#loc49)
    return loc(#loc50)
  } loc(#loc48)
} loc(#loc)
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file1 = #llvm.di_file<"/usr/include/stdio.h" in "">
#loc = loc("tests/app/stream_update/main.cpp":0:0)
#loc1 = loc("tests/app/stream_update/main.cpp":6:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/stream_update/main.cpp":11:0)
#loc5 = loc("tests/app/stream_update/main.cpp":13:0)
#loc6 = loc("tests/app/stream_update/main.cpp":14:0)
#loc7 = loc("tests/app/stream_update/main.cpp":16:0)
#loc8 = loc("tests/app/stream_update/main.cpp":17:0)
#loc9 = loc("tests/app/stream_update/main.cpp":19:0)
#loc10 = loc("tests/app/stream_update/main.cpp":20:0)
#loc11 = loc("tests/app/stream_update/main.cpp":24:0)
#loc12 = loc("tests/app/stream_update/main.cpp":21:0)
#loc13 = loc("tests/app/stream_update/main.cpp":26:0)
#loc14 = loc("/usr/include/stdio.h":363:0)
#loc16 = loc("tests/app/stream_update/stream_update.cpp":27:0)
#loc17 = loc("tests/app/stream_update/stream_update.cpp":31:0)
#loc18 = loc("tests/app/stream_update/stream_update.cpp":32:0)
#loc19 = loc("tests/app/stream_update/stream_update.cpp":33:0)
#loc20 = loc("tests/app/stream_update/stream_update.cpp":36:0)
#loc21 = loc("tests/app/stream_update/stream_update.cpp":37:0)
#loc25 = loc("tests/app/stream_update/stream_update.cpp":12:0)
#loc26 = loc("tests/app/stream_update/stream_update.cpp":13:0)
#loc27 = loc("tests/app/stream_update/stream_update.cpp":16:0)
#loc28 = loc("tests/app/stream_update/stream_update.cpp":17:0)
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type3>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type, #di_derived_type5, #di_null_type>
#di_subprogram4 = #llvm.di_subprogram<scope = #di_file1, name = "printf", file = #di_file1, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type1>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 20>
#loc29 = loc(fused<#di_subprogram3>[#loc1])
#loc30 = loc(fused<#di_subprogram3>[#loc5])
#loc31 = loc(fused<#di_subprogram3>[#loc6])
#loc32 = loc(fused<#di_subprogram3>[#loc7])
#loc33 = loc(fused<#di_subprogram3>[#loc8])
#loc34 = loc(fused<#di_subprogram3>[#loc9])
#loc35 = loc(fused<#di_subprogram3>[#loc11])
#loc36 = loc(fused<#di_subprogram3>[#loc13])
#loc37 = loc(fused<#di_subprogram4>[#loc14])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 10>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 20>
#loc39 = loc(fused<#di_lexical_block12>[#loc10])
#loc40 = loc(fused<#di_lexical_block15>[#loc4])
#loc41 = loc(fused<#di_lexical_block15>[#loc3])
#loc42 = loc(fused[#loc39, #loc34])
#loc43 = loc(fused<#di_lexical_block16>[#loc12])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file2, line = 27>
#loc44 = loc(fused[#loc38, #loc41])
#loc46 = loc(fused<#di_subprogram5>[#loc20])
#loc47 = loc(fused<#di_subprogram5>[#loc21])
#loc49 = loc(fused<#di_subprogram6>[#loc27])
#loc50 = loc(fused<#di_subprogram6>[#loc28])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file2, line = 27>
#loc51 = loc(fused<#di_lexical_block17>[#loc16])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file2, line = 27>
#loc53 = loc(fused<#di_lexical_block19>[#loc16])
#loc54 = loc(fused<#di_lexical_block20>[#loc23])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file2, line = 31>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file2, line = 31>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file2, line = 11>
#loc55 = loc(fused<#di_lexical_block23>[#loc17])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file2, line = 31>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file2, line = 11>
#loc57 = loc(fused<#di_lexical_block26>[#loc24])
#loc58 = loc(fused<#di_lexical_block27>[#loc18])
#loc59 = loc(fused<#di_lexical_block27>[#loc19])
#loc60 = loc(fused<#di_lexical_block28>[#loc25])
#loc61 = loc(fused<#di_lexical_block28>[#loc26])
#loc62 = loc(fused[#loc56, #loc57])
