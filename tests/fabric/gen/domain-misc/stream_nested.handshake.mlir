#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/stream_nested/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file2 = #llvm.di_file<"tests/app/stream_nested/stream_nested.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/stream_nested/main.cpp":9:0)
#loc15 = loc("tests/app/stream_nested/stream_nested.cpp":21:0)
#loc23 = loc("tests/app/stream_nested/stream_nested.cpp":6:0)
#loc24 = loc("tests/app/stream_nested/stream_nested.cpp":9:0)
#loc25 = loc("tests/app/stream_nested/stream_nested.cpp":10:0)
#loc26 = loc("tests/app/stream_nested/stream_nested.cpp":11:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file2, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_label = #llvm.di_label<scope = #di_subprogram1, name = "outer_loop", file = #di_file2, line = 24>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 9>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file2, line = 27>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file2, line = 9>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "passed", file = #di_file, line = 18, type = #di_basic_type2>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file2, line = 27>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file2, line = 9>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 512, elements = #llvm.di_subrange<count = 16 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type2, sizeInBits = 32, elements = #llvm.di_subrange<count = 1 : i64>>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file2, line = 27>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file2, line = 9>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 9, type = #di_derived_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram1, name = "n", file = #di_file2, line = 22, arg = 3, type = #di_derived_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram1, name = "acc", file = #di_file2, line = 23, type = #di_derived_type2>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file2, line = 27, type = #di_derived_type2>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram2, name = "n", file = #di_file2, line = 7, arg = 3, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram2, name = "acc", file = #di_file2, line = 8, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file2, line = 9, type = #di_derived_type2>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type4, sizeInBits = 64>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type6>
#di_label1 = #llvm.di_label<scope = #di_lexical_block5, name = "middle_loop", file = #di_file2, line = 28>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file2, line = 31>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file2, line = 10>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "n", file = #di_file, line = 7, type = #di_derived_type4>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 8, type = #di_composite_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "cpu_out", file = #di_file, line = 12, type = #di_composite_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "dsa_out", file = #di_file, line = 13, type = #di_composite_type1>
#di_derived_type9 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type7>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file2, line = 31>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file2, line = 10>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file2, line = 22, arg = 2, type = #di_derived_type8>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "j", file = #di_file2, line = 31, type = #di_derived_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output", file = #di_file2, line = 7, arg = 2, type = #di_derived_type8>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file2, line = 10, type = #di_derived_type2>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable8, #di_local_variable9, #di_local_variable1, #di_local_variable10, #di_local_variable11, #di_local_variable>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 9>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file2, line = 31>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file2, line = 10>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file2, line = 21, arg = 1, type = #di_derived_type9>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file2, line = 6, arg = 1, type = #di_derived_type9>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type9, #di_derived_type8, #di_derived_type2>
#di_label2 = #llvm.di_label<scope = #di_lexical_block13, name = "inner_loop", file = #di_file2, line = 32>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file2, line = 35>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file2, line = 11>
#loc40 = loc(fused<#di_lexical_block11>[#loc3])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file2, line = 35>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file2, line = 11>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block17, name = "k", file = #di_file2, line = 35, type = #di_derived_type2>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block18, name = "k", file = #di_file2, line = 11, type = #di_derived_type2>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file2, line = 35>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file2, line = 11>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_lexical_block21, name = "idx", file = #di_file2, line = 36, type = #di_derived_type2>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block22, name = "idx", file = #di_file2, line = 12, type = #di_derived_type2>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file2, name = "stream_nested_dsa", linkageName = "_Z17stream_nested_dsaPKjPjj", file = #di_file2, line = 21, scopeLine = 22, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable16, #di_local_variable12, #di_local_variable2, #di_local_variable3, #di_label, #di_local_variable4, #di_label1, #di_local_variable13, #di_label2, #di_local_variable18, #di_local_variable20>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file2, name = "stream_nested_cpu", linkageName = "_Z17stream_nested_cpuPKjPjj", file = #di_file2, line = 6, scopeLine = 7, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable17, #di_local_variable14, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable15, #di_local_variable19, #di_local_variable21>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file2, line = 9>
#loc47 = loc(fused<#di_subprogram5>[#loc15])
#loc50 = loc(fused<#di_subprogram6>[#loc23])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file2, line = 9>
#loc54 = loc(fused<#di_lexical_block24>[#loc24])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file2, line = 9>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file2, line = 10>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file2, line = 10>
#loc58 = loc(fused<#di_lexical_block30>[#loc25])
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file2, line = 10>
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file2, line = 11>
#loc62 = loc(fused<#di_lexical_block36>[#loc26])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<23xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 99, 112, 117, 61, 37, 117, 32, 100, 115, 97, 61, 37, 117, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.1" : memref<33xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 115, 116, 114, 101, 97, 109, 95, 110, 101, 115, 116, 101, 100, 32, 114, 101, 115, 117, 108, 116, 61, 37, 117, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 61, 115, 116, 114, 101, 97, 109, 95, 110, 101, 115, 116, 101, 100, 0]> loc(#loc)
  memref.global constant @".str.1.5" : memref<42xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 115, 116, 114, 101, 97, 109, 95, 110, 101, 115, 116, 101, 100, 47, 115, 116, 114, 101, 97, 109, 95, 110, 101, 115, 116, 101, 100, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc31)
    %false = arith.constant false loc(#loc31)
    %0 = seq.const_clock  low loc(#loc31)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c16_i64 = arith.constant 16 : i64 loc(#loc2)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %1 = memref.get_global @".str" : memref<23xi8> loc(#loc2)
    %2 = memref.get_global @".str.1" : memref<33xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<16xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<1xi32> loc(#loc2)
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.addi %arg0, %c1_i64 : i64 loc(#loc42)
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc42)
      %12 = arith.trunci %10 : i64 to i32 loc(#loc42)
      memref.store %12, %alloca[%11] : memref<16xi32> loc(#loc42)
      %13 = arith.cmpi ne, %10, %c16_i64 : i64 loc(#loc46)
      scf.condition(%13) %10 : i64 loc(#loc40)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc40)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc40)
    memref.store %c0_i32, %alloca_0[%c0] : memref<1xi32> loc(#loc32)
    memref.store %c0_i32, %alloca_1[%c0] : memref<1xi32> loc(#loc33)
    %cast = memref.cast %alloca : memref<16xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    %cast_2 = memref.cast %alloca_0 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc34)
    call @_Z17stream_nested_cpuPKjPjj(%cast, %cast_2, %c16_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc34)
    %cast_3 = memref.cast %alloca_1 : memref<1xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc35)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c16_i32, %true : i32 loc(#loc35)
    %chanOutput_8, %ready_9 = esi.wrap.vr %true, %true : i1 loc(#loc35)
    %4 = handshake.esi_instance @_Z17stream_nested_dsaPKjPjj_esi "_Z17stream_nested_dsaPKjPjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc35)
    %rawOutput, %valid = esi.unwrap.vr %4, %true : i1 loc(#loc35)
    %5 = memref.load %alloca_0[%c0] : memref<1xi32> loc(#loc36)
    %6 = memref.load %alloca_1[%c0] : memref<1xi32> loc(#loc36)
    %7 = arith.cmpi eq, %5, %6 : i32 loc(#loc36)
    %8 = arith.cmpi ne, %5, %6 : i32 loc(#loc44)
    %9 = arith.extui %8 : i1 to i32 loc(#loc41)
    scf.if %7 {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<33xi8> -> index loc(#loc37)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc37)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc37)
      %12 = llvm.call @printf(%11, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32 loc(#loc37)
    } else {
      %intptr = memref.extract_aligned_pointer_as_index %1 : memref<23xi8> -> index loc(#loc45)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc45)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc45)
      %12 = llvm.call @printf(%11, %5, %6) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32 loc(#loc45)
    } loc(#loc41)
    return %9 : i32 loc(#loc38)
  } loc(#loc31)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc39)
  handshake.func @_Z17stream_nested_dsaPKjPjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: i32 loc(fused<#di_subprogram5>[#loc15]), %arg3: i1 loc(fused<#di_subprogram5>[#loc15]), ...) -> i1 attributes {argNames = ["input", "output", "n", "start_token"], loom.annotations = ["loom.accel=stream_nested"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : i1 loc(#loc47)
    %1 = handshake.join %0 : none loc(#loc47)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : index} : index loc(#loc53)
    %5 = arith.cmpi ugt, %arg2, %2 : i32 loc(#loc55)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc53)
    handshake.sink %falseResult : none loc(#loc53)
    %6 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc53)
    %7 = arith.index_cast %2 : i32 to index loc(#loc53)
    %8 = arith.index_cast %arg2 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %7, %6, %8 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.parallel degree=2 schedule=1"], step_op = "<<=", stop_cond = "<"} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %9 = dataflow.carry %willContinue, %3, %falseResult_11 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_1 : i1 loc(#loc53)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %9 : i32 loc(#loc53)
    %10 = dataflow.invariant %afterCond, %trueResult : i1, none -> none loc(#loc53)
    %11 = arith.index_cast %afterValue : index to i32 loc(#loc53)
    %12 = handshake.constant %10 {value = 1 : index} : index loc(#loc57)
    %13 = arith.index_cast %3 : i32 to index loc(#loc57)
    %index_4, %willContinue_5 = dataflow.stream %8, %12, %13 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = ">>=", stop_cond = "!="} loc(#loc57)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc57)
    %14 = dataflow.carry %willContinue_5, %afterValue_0, %falseResult_19 : i1, i32, i32 -> i32 loc(#loc57)
    %afterValue_8, %afterCond_9 = dataflow.gate %14, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc57)
    handshake.sink %afterCond_9 : i1 loc(#loc57)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %14 : i32 loc(#loc57)
    %15 = dataflow.invariant %afterCond_7, %10 : i1, none -> none loc(#loc57)
    %16 = arith.index_cast %afterValue_6 : index to i32 loc(#loc57)
    %17 = dataflow.invariant %afterCond_7, %11 : i1, i32 -> i32 loc(#loc2)
    %18 = arith.addi %16, %17 : i32 loc(#loc2)
    %19 = handshake.constant %15 {value = 1 : index} : index loc(#loc61)
    %index_12, %willContinue_13 = dataflow.stream %7, %19, %8 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = "<<=", stop_cond = "<="} loc(#loc61)
    %afterValue_14, %afterCond_15 = dataflow.gate %index_12, %willContinue_13 : index, i1 -> index, i1 loc(#loc61)
    %20 = dataflow.carry %willContinue_13, %afterValue_8, %27 : i1, i32, i32 -> i32 loc(#loc61)
    %afterValue_16, %afterCond_17 = dataflow.gate %20, %willContinue_13 : i32, i1 -> i32, i1 loc(#loc61)
    handshake.sink %afterCond_17 : i1 loc(#loc61)
    %trueResult_18, %falseResult_19 = handshake.cond_br %willContinue_13, %20 : i32 loc(#loc61)
    %21 = arith.index_cast %afterValue_14 : index to i32 loc(#loc61)
    %22 = dataflow.invariant %afterCond_15, %18 : i1, i32 -> i32 loc(#loc64)
    %23 = arith.addi %22, %21 : i32 loc(#loc64)
    %24 = arith.remui %23, %arg2 : i32 loc(#loc64)
    %25 = arith.extui %24 : i32 to i64 loc(#loc65)
    %26 = arith.index_cast %25 : i64 to index loc(#loc65)
    %dataResult, %addressResults = handshake.load [%26] %31#0, %35 : index, i32 loc(#loc65)
    %27 = arith.addi %dataResult, %afterValue_16 : i32 loc(#loc65)
    %28 = handshake.constant %1 {value = 1 : index} : index loc(#loc53)
    %29 = arith.select %5, %28, %4 : index loc(#loc53)
    %30 = handshake.mux %29 [%3, %falseResult_3] : index, i32 loc(#loc53)
    %dataResult_20, %addressResult = handshake.store [%4] %30, %1 : index, i32 loc(#loc48)
    %31:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc47)
    %32 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_20, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc47)
    %33 = dataflow.carry %willContinue, %trueResult, %trueResult_25 : i1, none, none -> none loc(#loc53)
    %34 = dataflow.carry %willContinue_5, %33, %trueResult_23 : i1, none, none -> none loc(#loc57)
    %35 = dataflow.carry %willContinue_13, %34, %trueResult_21 : i1, none, none -> none loc(#loc61)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_13, %31#1 : none loc(#loc61)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_5, %falseResult_22 : none loc(#loc57)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %falseResult_24 : none loc(#loc53)
    %36 = handshake.mux %29 [%falseResult, %falseResult_26] : index, none loc(#loc53)
    %37 = handshake.join %36, %32 : none, none loc(#loc47)
    %38 = handshake.constant %37 {value = true} : i1 loc(#loc47)
    handshake.return %38 : i1 loc(#loc47)
  } loc(#loc47)
  handshake.func @_Z17stream_nested_dsaPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: i32 loc(fused<#di_subprogram5>[#loc15]), %arg3: none loc(fused<#di_subprogram5>[#loc15]), ...) -> none attributes {argNames = ["input", "output", "n", "start_token"], loom.annotations = ["loom.accel=stream_nested"], resNames = ["done_token"]} {
    %0 = handshake.join %arg3 : none loc(#loc47)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : index} : index loc(#loc53)
    %4 = arith.cmpi ugt, %arg2, %1 : i32 loc(#loc55)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc53)
    handshake.sink %falseResult : none loc(#loc53)
    %5 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc53)
    %6 = arith.index_cast %1 : i32 to index loc(#loc53)
    %7 = arith.index_cast %arg2 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.parallel degree=2 schedule=1"], step_op = "<<=", stop_cond = "<"} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %8 = dataflow.carry %willContinue, %2, %falseResult_11 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_1 : i1 loc(#loc53)
    %trueResult_2, %falseResult_3 = handshake.cond_br %willContinue, %8 : i32 loc(#loc53)
    %9 = dataflow.invariant %afterCond, %trueResult : i1, none -> none loc(#loc53)
    %10 = arith.index_cast %afterValue : index to i32 loc(#loc53)
    %11 = handshake.constant %9 {value = 1 : index} : index loc(#loc57)
    %12 = arith.index_cast %2 : i32 to index loc(#loc57)
    %index_4, %willContinue_5 = dataflow.stream %7, %11, %12 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = ">>=", stop_cond = "!="} loc(#loc57)
    %afterValue_6, %afterCond_7 = dataflow.gate %index_4, %willContinue_5 : index, i1 -> index, i1 loc(#loc57)
    %13 = dataflow.carry %willContinue_5, %afterValue_0, %falseResult_19 : i1, i32, i32 -> i32 loc(#loc57)
    %afterValue_8, %afterCond_9 = dataflow.gate %13, %willContinue_5 : i32, i1 -> i32, i1 loc(#loc57)
    handshake.sink %afterCond_9 : i1 loc(#loc57)
    %trueResult_10, %falseResult_11 = handshake.cond_br %willContinue_5, %13 : i32 loc(#loc57)
    %14 = dataflow.invariant %afterCond_7, %9 : i1, none -> none loc(#loc57)
    %15 = arith.index_cast %afterValue_6 : index to i32 loc(#loc57)
    %16 = dataflow.invariant %afterCond_7, %10 : i1, i32 -> i32 loc(#loc2)
    %17 = arith.addi %15, %16 : i32 loc(#loc2)
    %18 = handshake.constant %14 {value = 1 : index} : index loc(#loc61)
    %index_12, %willContinue_13 = dataflow.stream %6, %18, %7 {loom.annotations = ["loom.loop.tripcount typical=8 avg=8 min=1 max=16", "loom.loop.unroll factor=2"], step_op = "<<=", stop_cond = "<="} loc(#loc61)
    %afterValue_14, %afterCond_15 = dataflow.gate %index_12, %willContinue_13 : index, i1 -> index, i1 loc(#loc61)
    %19 = dataflow.carry %willContinue_13, %afterValue_8, %26 : i1, i32, i32 -> i32 loc(#loc61)
    %afterValue_16, %afterCond_17 = dataflow.gate %19, %willContinue_13 : i32, i1 -> i32, i1 loc(#loc61)
    handshake.sink %afterCond_17 : i1 loc(#loc61)
    %trueResult_18, %falseResult_19 = handshake.cond_br %willContinue_13, %19 : i32 loc(#loc61)
    %20 = arith.index_cast %afterValue_14 : index to i32 loc(#loc61)
    %21 = dataflow.invariant %afterCond_15, %17 : i1, i32 -> i32 loc(#loc64)
    %22 = arith.addi %21, %20 : i32 loc(#loc64)
    %23 = arith.remui %22, %arg2 : i32 loc(#loc64)
    %24 = arith.extui %23 : i32 to i64 loc(#loc65)
    %25 = arith.index_cast %24 : i64 to index loc(#loc65)
    %dataResult, %addressResults = handshake.load [%25] %30#0, %34 : index, i32 loc(#loc65)
    %26 = arith.addi %dataResult, %afterValue_16 : i32 loc(#loc65)
    %27 = handshake.constant %0 {value = 1 : index} : index loc(#loc53)
    %28 = arith.select %4, %27, %3 : index loc(#loc53)
    %29 = handshake.mux %28 [%2, %falseResult_3] : index, i32 loc(#loc53)
    %dataResult_20, %addressResult = handshake.store [%3] %29, %0 : index, i32 loc(#loc48)
    %30:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc47)
    %31 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_20, %addressResult) {id = 1 : i32} : (i32, index) -> none loc(#loc47)
    %32 = dataflow.carry %willContinue, %trueResult, %trueResult_25 : i1, none, none -> none loc(#loc53)
    %33 = dataflow.carry %willContinue_5, %32, %trueResult_23 : i1, none, none -> none loc(#loc57)
    %34 = dataflow.carry %willContinue_13, %33, %trueResult_21 : i1, none, none -> none loc(#loc61)
    %trueResult_21, %falseResult_22 = handshake.cond_br %willContinue_13, %30#1 : none loc(#loc61)
    %trueResult_23, %falseResult_24 = handshake.cond_br %willContinue_5, %falseResult_22 : none loc(#loc57)
    %trueResult_25, %falseResult_26 = handshake.cond_br %willContinue, %falseResult_24 : none loc(#loc53)
    %35 = handshake.mux %28 [%falseResult, %falseResult_26] : index, none loc(#loc53)
    %36 = handshake.join %35, %31 : none, none loc(#loc47)
    handshake.return %36 : none loc(#loc49)
  } loc(#loc47)
  func.func @_Z17stream_nested_cpuPKjPjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram6>[#loc23]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram6>[#loc23]), %arg2: i32 loc(fused<#di_subprogram6>[#loc23])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %0 = arith.cmpi ugt, %arg2, %c1_i32 : i32 loc(#loc56)
    %1 = scf.if %0 -> (i32) {
      %2:2 = scf.while (%arg3 = %c0_i32, %arg4 = %c1_i32) : (i32, i32) -> (i32, i32) {
        %3:2 = scf.while (%arg5 = %arg3, %arg6 = %arg2) : (i32, i32) -> (i32, i32) {
          %6 = arith.addi %arg6, %arg4 : i32 loc(#loc2)
          %7:2 = scf.while (%arg7 = %arg5, %arg8 = %c1_i32) : (i32, i32) -> (i32, i32) {
            %10 = arith.addi %6, %arg8 : i32 loc(#loc66)
            %11 = arith.remui %10, %arg2 : i32 loc(#loc66)
            %12 = arith.extui %11 : i32 to i64 loc(#loc67)
            %13 = arith.index_cast %12 : i64 to index loc(#loc67)
            %14 = memref.load %arg0[%13] : memref<?xi32, strided<[1], offset: ?>> loc(#loc67)
            %15 = arith.addi %14, %arg7 : i32 loc(#loc67)
            %16 = arith.shli %arg8, %c1_i32 : i32 loc(#loc63)
            %17 = arith.cmpi ule, %16, %arg2 : i32 loc(#loc68)
            scf.condition(%17) %15, %16 : i32, i32 loc(#loc62)
          } do {
          ^bb0(%arg7: i32 loc(fused<#di_lexical_block36>[#loc26]), %arg8: i32 loc(fused<#di_lexical_block36>[#loc26])):
            scf.yield %arg7, %arg8 : i32, i32 loc(#loc62)
          } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = "<<=", stop_cond = "<="}} loc(#loc62)
          %8 = arith.shrui %arg6, %c1_i32 : i32 loc(#loc59)
          %9 = arith.cmpi ne, %8, %c0_i32 : i32 loc(#loc60)
          scf.condition(%9) %7#0, %8 : i32, i32 loc(#loc58)
        } do {
        ^bb0(%arg5: i32 loc(fused<#di_lexical_block30>[#loc25]), %arg6: i32 loc(fused<#di_lexical_block30>[#loc25])):
          scf.yield %arg5, %arg6 : i32, i32 loc(#loc58)
        } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = ">>=", stop_cond = "!="}} loc(#loc58)
        %4 = arith.shli %arg4, %c1_i32 : i32 loc(#loc56)
        %5 = arith.cmpi ult, %4, %arg2 : i32 loc(#loc56)
        scf.condition(%5) %3#0, %4 : i32, i32 loc(#loc54)
      } do {
      ^bb0(%arg3: i32 loc(fused<#di_lexical_block24>[#loc24]), %arg4: i32 loc(fused<#di_lexical_block24>[#loc24])):
        scf.yield %arg3, %arg4 : i32, i32 loc(#loc54)
      } attributes {loom.stream = {cmp_on_update = true, iv = 1 : i64, step_op = "<<=", stop_cond = "<"}} loc(#loc54)
      scf.yield %2#0 : i32 loc(#loc54)
    } else {
      scf.yield %c0_i32 : i32 loc(#loc54)
    } loc(#loc54)
    memref.store %1, %arg1[%c0] : memref<?xi32, strided<[1], offset: ?>> loc(#loc51)
    return loc(#loc52)
  } loc(#loc50)
} loc(#loc)
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file1 = #llvm.di_file<"/usr/include/stdio.h" in "">
#loc = loc("tests/app/stream_nested/main.cpp":0:0)
#loc1 = loc("tests/app/stream_nested/main.cpp":6:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/stream_nested/main.cpp":10:0)
#loc5 = loc("tests/app/stream_nested/main.cpp":12:0)
#loc6 = loc("tests/app/stream_nested/main.cpp":13:0)
#loc7 = loc("tests/app/stream_nested/main.cpp":15:0)
#loc8 = loc("tests/app/stream_nested/main.cpp":16:0)
#loc9 = loc("tests/app/stream_nested/main.cpp":18:0)
#loc10 = loc("tests/app/stream_nested/main.cpp":19:0)
#loc11 = loc("tests/app/stream_nested/main.cpp":23:0)
#loc12 = loc("tests/app/stream_nested/main.cpp":20:0)
#loc13 = loc("tests/app/stream_nested/main.cpp":25:0)
#loc14 = loc("/usr/include/stdio.h":363:0)
#loc16 = loc("tests/app/stream_nested/stream_nested.cpp":27:0)
#loc17 = loc("tests/app/stream_nested/stream_nested.cpp":31:0)
#loc18 = loc("tests/app/stream_nested/stream_nested.cpp":35:0)
#loc19 = loc("tests/app/stream_nested/stream_nested.cpp":36:0)
#loc20 = loc("tests/app/stream_nested/stream_nested.cpp":37:0)
#loc21 = loc("tests/app/stream_nested/stream_nested.cpp":41:0)
#loc22 = loc("tests/app/stream_nested/stream_nested.cpp":42:0)
#loc27 = loc("tests/app/stream_nested/stream_nested.cpp":12:0)
#loc28 = loc("tests/app/stream_nested/stream_nested.cpp":13:0)
#loc29 = loc("tests/app/stream_nested/stream_nested.cpp":17:0)
#loc30 = loc("tests/app/stream_nested/stream_nested.cpp":18:0)
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type3>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type, #di_derived_type5, #di_null_type>
#di_subprogram4 = #llvm.di_subprogram<scope = #di_file1, name = "printf", file = #di_file1, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type1>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 19>
#loc31 = loc(fused<#di_subprogram3>[#loc1])
#loc32 = loc(fused<#di_subprogram3>[#loc5])
#loc33 = loc(fused<#di_subprogram3>[#loc6])
#loc34 = loc(fused<#di_subprogram3>[#loc7])
#loc35 = loc(fused<#di_subprogram3>[#loc8])
#loc36 = loc(fused<#di_subprogram3>[#loc9])
#loc37 = loc(fused<#di_subprogram3>[#loc11])
#loc38 = loc(fused<#di_subprogram3>[#loc13])
#loc39 = loc(fused<#di_subprogram4>[#loc14])
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 9>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 19>
#loc41 = loc(fused<#di_lexical_block12>[#loc10])
#loc42 = loc(fused<#di_lexical_block15>[#loc4])
#loc43 = loc(fused<#di_lexical_block15>[#loc3])
#loc44 = loc(fused[#loc41, #loc36])
#loc45 = loc(fused<#di_lexical_block16>[#loc12])
#loc46 = loc(fused[#loc40, #loc43])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file2, line = 27>
#loc48 = loc(fused<#di_subprogram5>[#loc21])
#loc49 = loc(fused<#di_subprogram5>[#loc22])
#loc51 = loc(fused<#di_subprogram6>[#loc29])
#loc52 = loc(fused<#di_subprogram6>[#loc30])
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file2, line = 27>
#loc53 = loc(fused<#di_lexical_block23>[#loc16])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file2, line = 27>
#loc55 = loc(fused<#di_lexical_block25>[#loc16])
#loc56 = loc(fused<#di_lexical_block26>[#loc24])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file2, line = 31>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file2, line = 31>
#loc57 = loc(fused<#di_lexical_block29>[#loc17])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file2, line = 31>
#loc59 = loc(fused<#di_lexical_block32>[#loc25])
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file2, line = 35>
#loc60 = loc(fused[#loc58, #loc59])
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file2, line = 35>
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file2, line = 11>
#loc61 = loc(fused<#di_lexical_block35>[#loc18])
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file2, line = 35>
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file2, line = 11>
#loc63 = loc(fused<#di_lexical_block38>[#loc26])
#loc64 = loc(fused<#di_lexical_block39>[#loc19])
#loc65 = loc(fused<#di_lexical_block39>[#loc20])
#loc66 = loc(fused<#di_lexical_block40>[#loc27])
#loc67 = loc(fused<#di_lexical_block40>[#loc28])
#loc68 = loc(fused[#loc62, #loc63])
