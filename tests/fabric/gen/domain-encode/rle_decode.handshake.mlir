#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned long", sizeInBits = 64, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/rle_decode/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/rle_decode/rle_decode.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc5 = loc("tests/app/rle_decode/main.cpp":14:0)
#loc11 = loc("tests/app/rle_decode/main.cpp":29:0)
#loc17 = loc("tests/app/rle_decode/rle_decode.cpp":31:0)
#loc25 = loc("tests/app/rle_decode/rle_decode.cpp":12:0)
#loc26 = loc("tests/app/rle_decode/rle_decode.cpp":18:0)
#loc29 = loc("tests/app/rle_decode/rle_decode.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 14>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 37>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 18>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "__vla_expr0", type = #di_basic_type2, flags = Artificial>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "__vla_expr1", type = #di_basic_type2, flags = Artificial>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block2, file = #di_file1, line = 37>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 18>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 224, elements = #llvm.di_subrange<count = 7 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, elements = #llvm.di_subrange<count = #di_local_variable>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, elements = #llvm.di_subrange<count = #di_local_variable1>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 37>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 18>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "output_size", file = #di_file, line = 13, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 14, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 29, type = #di_derived_type1>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram1, name = "write_idx", file = #di_file1, line = 35, type = #di_derived_type1>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 37, type = #di_derived_type1>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram2, name = "write_idx", file = #di_file1, line = 16, type = #di_derived_type1>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 18, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 41>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file1, line = 22>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "encoded_length", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "values", file = #di_file, line = 9, type = #di_composite_type>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram, name = "counts", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_output", file = #di_file, line = 19, type = #di_composite_type1>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_output", file = #di_file, line = 20, type = #di_composite_type2>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram1, name = "encoded_length", file = #di_file1, line = 34, arg = 4, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "value", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block6, name = "count", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram2, name = "encoded_length", file = #di_file1, line = 15, arg = 4, type = #di_derived_type2>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "value", file = #di_file1, line = 19, type = #di_derived_type1>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "count", file = #di_file1, line = 20, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output_data", file = #di_file1, line = 33, arg = 3, type = #di_derived_type5>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "j", file = #di_file1, line = 41, type = #di_derived_type1>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_data", file = #di_file1, line = 14, arg = 3, type = #di_derived_type5>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "j", file = #di_file1, line = 22, type = #di_derived_type1>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable2, #di_local_variable3, #di_local_variable, #di_local_variable12, #di_local_variable1, #di_local_variable13, #di_local_variable4>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 14>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 29>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_values", file = #di_file1, line = 31, arg = 1, type = #di_derived_type6>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input_counts", file = #di_file1, line = 32, arg = 2, type = #di_derived_type6>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_values", file = #di_file1, line = 12, arg = 1, type = #di_derived_type6>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram2, name = "input_counts", file = #di_file1, line = 13, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "rle_decode_dsa", linkageName = "_Z14rle_decode_dsaPKjS0_Pjj", file = #di_file1, line = 31, scopeLine = 34, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable20, #di_local_variable14, #di_local_variable5, #di_local_variable6, #di_local_variable15, #di_local_variable16, #di_local_variable21>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "rle_decode_cpu", linkageName = "_Z14rle_decode_cpuPKjS0_Pjj", file = #di_file1, line = 12, scopeLine = 15, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable26, #di_local_variable27, #di_local_variable22, #di_local_variable17, #di_local_variable7, #di_local_variable8, #di_local_variable18, #di_local_variable19, #di_local_variable23>
#loc42 = loc(fused<#di_lexical_block10>[#loc5])
#loc43 = loc(fused<#di_lexical_block11>[#loc11])
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 18>
#loc46 = loc(fused<#di_subprogram4>[#loc17])
#loc48 = loc(fused<#di_subprogram5>[#loc25])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 18>
#loc54 = loc(fused<#di_lexical_block17>[#loc26])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 18>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 22>
#loc66 = loc(fused<#di_lexical_block25>[#loc29])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @__const.main.values : memref<7xi32> = dense<[1, 2, 3, 4, 5, 6, 7]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.counts : memref<7xi32> = dense<[3, 2, 4, 5, 1, 3, 2]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @str : memref<19xi8> = dense<[114, 108, 101, 95, 100, 101, 99, 111, 100, 101, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<19xi8> = dense<[114, 108, 101, 95, 100, 101, 99, 111, 100, 101, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<36xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 114, 108, 101, 95, 100, 101, 99, 111, 100, 101, 47, 114, 108, 101, 95, 100, 101, 99, 111, 100, 101, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc2)
    %0 = seq.const_clock  low loc(#loc33)
    %c2_i32 = arith.constant 2 : i32 loc(#loc33)
    %1 = ub.poison : i32 loc(#loc33)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c7 = arith.constant 7 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %2 = memref.get_global @__const.main.values : memref<7xi32> loc(#loc2)
    %3 = memref.get_global @__const.main.counts : memref<7xi32> loc(#loc2)
    %4 = memref.get_global @str : memref<19xi8> loc(#loc2)
    %5 = memref.get_global @str.2 : memref<19xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<7xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<7xi32> loc(#loc2)
    scf.for %arg0 = %c0 to %c7 step %c1 {
      %13 = memref.load %2[%arg0] : memref<7xi32> loc(#loc34)
      memref.store %13, %alloca[%arg0] : memref<7xi32> loc(#loc34)
    } loc(#loc34)
    scf.for %arg0 = %c0 to %c7 step %c1 {
      %13 = memref.load %3[%arg0] : memref<7xi32> loc(#loc35)
      memref.store %13, %alloca_0[%arg0] : memref<7xi32> loc(#loc35)
    } loc(#loc35)
    %6:2 = scf.while (%arg0 = %c0_i64, %arg1 = %c0_i32) : (i64, i32) -> (i64, i32) {
      %13 = arith.index_cast %arg0 : i64 to index loc(#loc50)
      %14 = memref.load %alloca_0[%13] : memref<7xi32> loc(#loc50)
      %15 = arith.addi %14, %arg1 : i32 loc(#loc50)
      %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc44)
      %17 = arith.cmpi ne, %16, %c7_i64 : i64 loc(#loc51)
      scf.condition(%17) %16, %15 : i64, i32 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc5]), %arg1: i32 loc(fused<#di_lexical_block10>[#loc5])):
      scf.yield %arg0, %arg1 : i64, i32 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %7 = arith.extui %6#1 : i32 to i64 loc(#loc36)
    %8 = arith.index_cast %7 : i64 to index loc(#loc36)
    %alloca_1 = memref.alloca(%8) : memref<?xi32> loc(#loc36)
    %alloca_2 = memref.alloca(%8) : memref<?xi32> loc(#loc37)
    %cast = memref.cast %alloca : memref<7xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_3 = memref.cast %alloca_0 : memref<7xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    %cast_4 = memref.cast %alloca_1 : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc38)
    call @_Z14rle_decode_cpuPKjS0_Pjj(%cast, %cast_3, %cast_4, %c7_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32) -> () loc(#loc38)
    %cast_5 = memref.cast %alloca_2 : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c7_i32, %true : i32 loc(#loc39)
    %chanOutput_12, %ready_13 = esi.wrap.vr %true, %true : i1 loc(#loc39)
    %9 = handshake.esi_instance @_Z14rle_decode_dsaPKjS0_Pjj_esi "_Z14rle_decode_dsaPKjS0_Pjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc39)
    %rawOutput, %valid = esi.unwrap.vr %9, %true : i1 loc(#loc39)
    %10 = arith.cmpi eq, %6#1, %c0_i32 : i32 loc(#loc45)
    %11:2 = scf.if %10 -> (i1, i32) {
      scf.yield %false, %c0_i32 : i1, i32 loc(#loc43)
    } else {
      %13:2 = scf.while (%arg0 = %c0_i32) : (i32) -> (i32, i32) {
        %16 = arith.extui %arg0 : i32 to i64 loc(#loc55)
        %17 = arith.index_cast %16 : i64 to index loc(#loc55)
        %18 = memref.load %alloca_1[%17] : memref<?xi32> loc(#loc55)
        %19 = memref.load %alloca_2[%17] : memref<?xi32> loc(#loc55)
        %20 = arith.cmpi eq, %18, %19 : i32 loc(#loc55)
        %21:3 = scf.if %20 -> (i32, i32, i32) {
          %23 = arith.addi %arg0, %c1_i32 : i32 loc(#loc45)
          %24 = arith.cmpi eq, %23, %6#1 : i32 loc(#loc45)
          %25 = arith.extui %24 : i1 to i32 loc(#loc43)
          %26 = arith.cmpi ne, %23, %6#1 : i32 loc(#loc52)
          %27 = arith.extui %26 : i1 to i32 loc(#loc43)
          scf.yield %23, %25, %27 : i32, i32, i32 loc(#loc55)
        } else {
          scf.yield %1, %c2_i32, %c0_i32 : i32, i32, i32 loc(#loc55)
        } loc(#loc55)
        %22 = arith.trunci %21#2 : i32 to i1 loc(#loc43)
        scf.condition(%22) %21#0, %21#1 : i32, i32 loc(#loc43)
      } do {
      ^bb0(%arg0: i32 loc(fused<#di_lexical_block11>[#loc11]), %arg1: i32 loc(fused<#di_lexical_block11>[#loc11])):
        scf.yield %arg0 : i32 loc(#loc43)
      } loc(#loc43)
      %14 = arith.index_castui %13#1 : i32 to index loc(#loc43)
      %15:2 = scf.index_switch %14 -> i1, i32 
      case 1 {
        scf.yield %false, %c0_i32 : i1, i32 loc(#loc43)
      }
      default {
        %intptr = memref.extract_aligned_pointer_as_index %4 : memref<19xi8> -> index loc(#loc58)
        %16 = arith.index_cast %intptr : index to i64 loc(#loc58)
        %17 = llvm.inttoptr %16 : i64 to !llvm.ptr loc(#loc58)
        %18 = llvm.call @puts(%17) : (!llvm.ptr) -> i32 loc(#loc58)
        scf.yield %true, %c1_i32 : i1, i32 loc(#loc59)
      } loc(#loc43)
      scf.yield %15#0, %15#1 : i1, i32 loc(#loc43)
    } loc(#loc43)
    %12 = arith.select %11#0, %11#1, %c0_i32 : i32 loc(#loc2)
    scf.if %11#0 {
    } else {
      %intptr = memref.extract_aligned_pointer_as_index %5 : memref<19xi8> -> index loc(#loc40)
      %13 = arith.index_cast %intptr : index to i64 loc(#loc40)
      %14 = llvm.inttoptr %13 : i64 to !llvm.ptr loc(#loc40)
      %15 = llvm.call @puts(%14) : (!llvm.ptr) -> i32 loc(#loc40)
    } loc(#loc2)
    return %12 : i32 loc(#loc41)
  } loc(#loc33)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.stackrestore.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z14rle_decode_dsaPKjS0_Pjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg3: i32 loc(fused<#di_subprogram4>[#loc17]), %arg4: i1 loc(fused<#di_subprogram4>[#loc17]), ...) -> i1 attributes {argNames = ["input_values", "input_counts", "output_data", "encoded_length", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc46)
    %1 = handshake.join %0 : none loc(#loc46)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %5, %1 : none loc(#loc53)
    %6 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %7 = arith.index_cast %4 : i64 to index loc(#loc53)
    %8 = arith.index_cast %arg3 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %7, %6, %8 {step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %9 = dataflow.carry %willContinue, %3, %22 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_0, %afterCond_1 = dataflow.gate %9, %willContinue : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_1 : i1 loc(#loc53)
    %10 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %23#0, %26 : index, i32 loc(#loc60)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %24#0, %38 : index, i32 loc(#loc61)
    %11 = arith.cmpi eq, %dataResult_2, %3 : i32 loc(#loc67)
    %12 = arith.addi %afterValue_0, %dataResult_2 : i32 loc(#loc65)
    %13 = handshake.constant %1 {value = true} : i1 loc(#loc65)
    %14 = dataflow.carry %13, %afterValue_0, %trueResult_5 : i1, i32, i32 -> i32 loc(#loc65)
    %15 = arith.extui %14 : i32 to i64 loc(#loc69)
    %16 = arith.index_cast %15 : i64 to index loc(#loc69)
    %dataResult_4, %addressResult = handshake.store [%16] %dataResult, %32 : index, i32 loc(#loc69)
    %17 = arith.addi %14, %2 : i32 loc(#loc70)
    %18 = arith.cmpi ne, %17, %12 : i32 loc(#loc71)
    %trueResult_5, %falseResult_6 = handshake.cond_br %18, %17 : i32 loc(#loc65)
    %19 = handshake.constant %10 {value = 0 : index} : index loc(#loc65)
    %20 = handshake.constant %10 {value = 1 : index} : index loc(#loc65)
    %21 = arith.select %11, %20, %19 : index loc(#loc65)
    %22 = handshake.mux %21 [%falseResult_6, %afterValue_0] : index, i32 loc(#loc65)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %24:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %25 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc46)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc53)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %23#1 : none loc(#loc53)
    %27 = handshake.constant %1 {value = 0 : index} : index loc(#loc53)
    %28 = handshake.constant %1 {value = 1 : index} : index loc(#loc53)
    %29 = arith.select %5, %28, %27 : index loc(#loc53)
    %30 = handshake.mux %29 [%falseResult_8, %trueResult] : index, none loc(#loc53)
    %31 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %11, %31 : none loc(#loc65)
    %32 = dataflow.carry %18, %falseResult_10, %trueResult_11 : i1, none, none -> none loc(#loc65)
    %trueResult_11, %falseResult_12 = handshake.cond_br %18, %25 : none loc(#loc65)
    %33 = handshake.constant %31 {value = 0 : index} : index loc(#loc65)
    %34 = handshake.constant %31 {value = 1 : index} : index loc(#loc65)
    %35 = arith.select %11, %34, %33 : index loc(#loc65)
    %36 = handshake.mux %35 [%falseResult_12, %trueResult_9] : index, none loc(#loc65)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %36 : none loc(#loc53)
    %37 = handshake.mux %29 [%falseResult_14, %trueResult] : index, none loc(#loc53)
    %38 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %24#1 : none loc(#loc53)
    %39 = handshake.mux %29 [%falseResult_16, %trueResult] : index, none loc(#loc53)
    %40 = handshake.join %30, %37, %39 : none, none, none loc(#loc46)
    %41 = handshake.constant %40 {value = true} : i1 loc(#loc46)
    handshake.return %41 : i1 loc(#loc46)
  } loc(#loc46)
  handshake.func @_Z14rle_decode_dsaPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc17]), %arg3: i32 loc(fused<#di_subprogram4>[#loc17]), %arg4: none loc(fused<#di_subprogram4>[#loc17]), ...) -> none attributes {argNames = ["input_values", "input_counts", "output_data", "encoded_length", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc46)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc56)
    %trueResult, %falseResult = handshake.cond_br %4, %0 : none loc(#loc53)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc53)
    %6 = arith.index_cast %3 : i64 to index loc(#loc53)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc53)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc53)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc53)
    %8 = dataflow.carry %willContinue, %2, %21 : i1, i32, i32 -> i32 loc(#loc53)
    %afterValue_0, %afterCond_1 = dataflow.gate %8, %willContinue : i32, i1 -> i32, i1 loc(#loc53)
    handshake.sink %afterCond_1 : i1 loc(#loc53)
    %9 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc53)
    %dataResult, %addressResults = handshake.load [%afterValue] %22#0, %25 : index, i32 loc(#loc60)
    %dataResult_2, %addressResults_3 = handshake.load [%afterValue] %23#0, %37 : index, i32 loc(#loc61)
    %10 = arith.cmpi eq, %dataResult_2, %2 : i32 loc(#loc67)
    %11 = arith.addi %afterValue_0, %dataResult_2 : i32 loc(#loc65)
    %12 = handshake.constant %0 {value = true} : i1 loc(#loc65)
    %13 = dataflow.carry %12, %afterValue_0, %trueResult_5 : i1, i32, i32 -> i32 loc(#loc65)
    %14 = arith.extui %13 : i32 to i64 loc(#loc69)
    %15 = arith.index_cast %14 : i64 to index loc(#loc69)
    %dataResult_4, %addressResult = handshake.store [%15] %dataResult, %31 : index, i32 loc(#loc69)
    %16 = arith.addi %13, %1 : i32 loc(#loc70)
    %17 = arith.cmpi ne, %16, %11 : i32 loc(#loc71)
    %trueResult_5, %falseResult_6 = handshake.cond_br %17, %16 : i32 loc(#loc65)
    %18 = handshake.constant %9 {value = 0 : index} : index loc(#loc65)
    %19 = handshake.constant %9 {value = 1 : index} : index loc(#loc65)
    %20 = arith.select %10, %19, %18 : index loc(#loc65)
    %21 = handshake.mux %20 [%falseResult_6, %afterValue_0] : index, i32 loc(#loc65)
    %22:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none) loc(#loc46)
    %23:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_3) {id = 1 : i32} : (index) -> (i32, none) loc(#loc46)
    %24 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult) {id = 2 : i32} : (i32, index) -> none loc(#loc46)
    %25 = dataflow.carry %willContinue, %falseResult, %trueResult_7 : i1, none, none -> none loc(#loc53)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %22#1 : none loc(#loc53)
    %26 = handshake.constant %0 {value = 0 : index} : index loc(#loc53)
    %27 = handshake.constant %0 {value = 1 : index} : index loc(#loc53)
    %28 = arith.select %4, %27, %26 : index loc(#loc53)
    %29 = handshake.mux %28 [%falseResult_8, %trueResult] : index, none loc(#loc53)
    %30 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc53)
    %trueResult_9, %falseResult_10 = handshake.cond_br %10, %30 : none loc(#loc65)
    %31 = dataflow.carry %17, %falseResult_10, %trueResult_11 : i1, none, none -> none loc(#loc65)
    %trueResult_11, %falseResult_12 = handshake.cond_br %17, %24 : none loc(#loc65)
    %32 = handshake.constant %30 {value = 0 : index} : index loc(#loc65)
    %33 = handshake.constant %30 {value = 1 : index} : index loc(#loc65)
    %34 = arith.select %10, %33, %32 : index loc(#loc65)
    %35 = handshake.mux %34 [%falseResult_12, %trueResult_9] : index, none loc(#loc65)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %35 : none loc(#loc53)
    %36 = handshake.mux %28 [%falseResult_14, %trueResult] : index, none loc(#loc53)
    %37 = dataflow.carry %willContinue, %falseResult, %trueResult_15 : i1, none, none -> none loc(#loc53)
    %trueResult_15, %falseResult_16 = handshake.cond_br %willContinue, %23#1 : none loc(#loc53)
    %38 = handshake.mux %28 [%falseResult_16, %trueResult] : index, none loc(#loc53)
    %39 = handshake.join %29, %36, %38 : none, none, none loc(#loc46)
    handshake.return %39 : none loc(#loc47)
  } loc(#loc46)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z14rle_decode_cpuPKjS0_Pjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc25]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc25]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc25]), %arg3: i32 loc(fused<#di_subprogram5>[#loc25])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc57)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc57)
      %2:2 = scf.while (%arg4 = %c0_i64, %arg5 = %c0_i32) : (i64, i32) -> (i64, i32) {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc62)
        %4 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc62)
        %5 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc63)
        %6 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc68)
        %7 = scf.if %6 -> (i32) {
          scf.yield %arg5 : i32 loc(#loc66)
        } else {
          %10 = arith.addi %arg5, %5 : i32 loc(#loc66)
          %11 = scf.while (%arg6 = %arg5) : (i32) -> i32 {
            %12 = arith.extui %arg6 : i32 to i64 loc(#loc72)
            %13 = arith.index_cast %12 : i64 to index loc(#loc72)
            memref.store %4, %arg2[%13] : memref<?xi32, strided<[1], offset: ?>> loc(#loc72)
            %14 = arith.addi %arg6, %c1_i32 : i32 loc(#loc73)
            %15 = arith.cmpi ne, %14, %10 : i32 loc(#loc74)
            scf.condition(%15) %14 : i32 loc(#loc66)
          } do {
          ^bb0(%arg6: i32 loc(fused<#di_lexical_block25>[#loc29])):
            scf.yield %arg6 : i32 loc(#loc66)
          } loc(#loc66)
          scf.yield %11 : i32 loc(#loc66)
        } loc(#loc66)
        %8 = arith.addi %arg4, %c1_i64 : i64 loc(#loc57)
        %9 = arith.cmpi ne, %8, %1 : i64 loc(#loc64)
        scf.condition(%9) %8, %7 : i64, i32 loc(#loc54)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block17>[#loc26]), %arg5: i32 loc(fused<#di_lexical_block17>[#loc26])):
        scf.yield %arg4, %arg5 : i64, i32 loc(#loc54)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc54)
    } loc(#loc54)
    return loc(#loc49)
  } loc(#loc48)
} loc(#loc)
#loc = loc("tests/app/rle_decode/main.cpp":0:0)
#loc1 = loc("tests/app/rle_decode/main.cpp":5:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/rle_decode/main.cpp":9:0)
#loc4 = loc("tests/app/rle_decode/main.cpp":10:0)
#loc6 = loc("tests/app/rle_decode/main.cpp":15:0)
#loc7 = loc("tests/app/rle_decode/main.cpp":19:0)
#loc8 = loc("tests/app/rle_decode/main.cpp":20:0)
#loc9 = loc("tests/app/rle_decode/main.cpp":23:0)
#loc10 = loc("tests/app/rle_decode/main.cpp":26:0)
#loc12 = loc("tests/app/rle_decode/main.cpp":30:0)
#loc13 = loc("tests/app/rle_decode/main.cpp":31:0)
#loc14 = loc("tests/app/rle_decode/main.cpp":32:0)
#loc15 = loc("tests/app/rle_decode/main.cpp":36:0)
#loc16 = loc("tests/app/rle_decode/main.cpp":38:0)
#loc18 = loc("tests/app/rle_decode/rle_decode.cpp":37:0)
#loc19 = loc("tests/app/rle_decode/rle_decode.cpp":38:0)
#loc20 = loc("tests/app/rle_decode/rle_decode.cpp":39:0)
#loc21 = loc("tests/app/rle_decode/rle_decode.cpp":41:0)
#loc22 = loc("tests/app/rle_decode/rle_decode.cpp":42:0)
#loc23 = loc("tests/app/rle_decode/rle_decode.cpp":43:0)
#loc24 = loc("tests/app/rle_decode/rle_decode.cpp":46:0)
#loc27 = loc("tests/app/rle_decode/rle_decode.cpp":19:0)
#loc28 = loc("tests/app/rle_decode/rle_decode.cpp":20:0)
#loc30 = loc("tests/app/rle_decode/rle_decode.cpp":23:0)
#loc31 = loc("tests/app/rle_decode/rle_decode.cpp":24:0)
#loc32 = loc("tests/app/rle_decode/rle_decode.cpp":27:0)
#loc33 = loc(fused<#di_subprogram3>[#loc1])
#loc34 = loc(fused<#di_subprogram3>[#loc3])
#loc35 = loc(fused<#di_subprogram3>[#loc4])
#loc36 = loc(fused<#di_subprogram3>[#loc7])
#loc37 = loc(fused<#di_subprogram3>[#loc8])
#loc38 = loc(fused<#di_subprogram3>[#loc9])
#loc39 = loc(fused<#di_subprogram3>[#loc10])
#loc40 = loc(fused<#di_subprogram3>[#loc15])
#loc41 = loc(fused<#di_subprogram3>[#loc16])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 14>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 29>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 14>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 29>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 37>
#loc44 = loc(fused<#di_lexical_block12>[#loc5])
#loc45 = loc(fused<#di_lexical_block13>[#loc11])
#loc47 = loc(fused<#di_subprogram4>[#loc24])
#loc49 = loc(fused<#di_subprogram5>[#loc32])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file, line = 30>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 37>
#loc50 = loc(fused<#di_lexical_block14>[#loc6])
#loc51 = loc(fused[#loc42, #loc44])
#loc52 = loc(fused[#loc43, #loc45])
#loc53 = loc(fused<#di_lexical_block16>[#loc18])
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 30>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 37>
#loc55 = loc(fused<#di_lexical_block18>[#loc12])
#loc56 = loc(fused<#di_lexical_block19>[#loc18])
#loc57 = loc(fused<#di_lexical_block20>[#loc26])
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 41>
#loc58 = loc(fused<#di_lexical_block21>[#loc13])
#loc59 = loc(fused<#di_lexical_block21>[#loc14])
#loc60 = loc(fused<#di_lexical_block22>[#loc19])
#loc61 = loc(fused<#di_lexical_block22>[#loc20])
#loc62 = loc(fused<#di_lexical_block23>[#loc27])
#loc63 = loc(fused<#di_lexical_block23>[#loc28])
#loc64 = loc(fused[#loc54, #loc57])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 41>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 22>
#loc65 = loc(fused<#di_lexical_block24>[#loc21])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 41>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 22>
#loc67 = loc(fused<#di_lexical_block26>[#loc21])
#loc68 = loc(fused<#di_lexical_block27>[#loc29])
#loc69 = loc(fused<#di_lexical_block28>[#loc22])
#loc70 = loc(fused<#di_lexical_block28>[#loc23])
#loc71 = loc(fused[#loc65, #loc67])
#loc72 = loc(fused<#di_lexical_block29>[#loc30])
#loc73 = loc(fused<#di_lexical_block29>[#loc31])
#loc74 = loc(fused[#loc66, #loc68])
