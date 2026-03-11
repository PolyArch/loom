#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_file = #llvm.di_file<"tests/app/scatter_add/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/scatter_add/scatter_add.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc3 = loc("tests/app/scatter_add/main.cpp":18:0)
#loc10 = loc("tests/app/scatter_add/main.cpp":35:0)
#loc16 = loc("tests/app/scatter_add/scatter_add.cpp":33:0)
#loc22 = loc("tests/app/scatter_add/scatter_add.cpp":17:0)
#loc23 = loc("tests/app/scatter_add/scatter_add.cpp":22:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 18>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 23>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 35>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file1, line = 38>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 22>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file1, line = 38>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file1, line = 22>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 32768, elements = #llvm.di_subrange<count = 1024 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type1, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file1, line = 38>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file1, line = 22>
#di_local_variable = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 18, type = #di_derived_type1>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 23, type = #di_derived_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 35, type = #di_derived_type1>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 38, type = #di_derived_type1>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 22, type = #di_derived_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 6, type = #di_derived_type2>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "dst_size", file = #di_file, line = 7, type = #di_derived_type2>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "src", file = #di_file, line = 10, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "indices", file = #di_file, line = 11, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram, name = "expect_dst", file = #di_file, line = 14, type = #di_composite_type1>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram, name = "calculated_dst", file = #di_file, line = 15, type = #di_composite_type1>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_subprogram1, name = "N", file = #di_file1, line = 36, arg = 4, type = #di_derived_type2>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_subprogram1, name = "dst_size", file = #di_file1, line = 37, arg = 5, type = #di_derived_type2>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "idx", file = #di_file1, line = 39, type = #di_derived_type1>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram2, name = "N", file = #di_file1, line = 20, arg = 4, type = #di_derived_type2>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram2, name = "dst_size", file = #di_file1, line = 21, arg = 5, type = #di_derived_type2>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "idx", file = #di_file1, line = 23, type = #di_derived_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram1, name = "dst", file = #di_file1, line = 35, arg = 3, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram2, name = "dst", file = #di_file1, line = 19, arg = 3, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[5]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 5, scopeLine = 5, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable, #di_local_variable1, #di_local_variable2>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 18>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 35>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram1, name = "src", file = #di_file1, line = 33, arg = 1, type = #di_derived_type6>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram1, name = "indices", file = #di_file1, line = 34, arg = 2, type = #di_derived_type6>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "src", file = #di_file1, line = 17, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram2, name = "indices", file = #di_file1, line = 18, arg = 2, type = #di_derived_type6>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type5, #di_derived_type2, #di_derived_type2>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[6]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "scatter_add_dsa", linkageName = "_Z15scatter_add_dsaPKjS0_Pjjj", file = #di_file1, line = 33, scopeLine = 37, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable19, #di_local_variable20, #di_local_variable17, #di_local_variable11, #di_local_variable12, #di_local_variable3, #di_local_variable13>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "scatter_add_cpu", linkageName = "_Z15scatter_add_cpuPKjS0_Pjjj", file = #di_file1, line = 17, scopeLine = 21, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable22, #di_local_variable18, #di_local_variable14, #di_local_variable15, #di_local_variable4, #di_local_variable16>
#loc33 = loc(fused<#di_lexical_block9>[#loc3])
#loc34 = loc(fused<#di_lexical_block11>[#loc10])
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file1, line = 22>
#loc37 = loc(fused<#di_subprogram4>[#loc16])
#loc39 = loc(fused<#di_subprogram5>[#loc22])
#loc48 = loc(fused<#di_lexical_block19>[#loc23])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @str : memref<20xi8> = dense<[115, 99, 97, 116, 116, 101, 114, 95, 97, 100, 100, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<20xi8> = dense<[115, 99, 97, 116, 116, 101, 114, 95, 97, 100, 100, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<19xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 56, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<38xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 115, 99, 97, 116, 116, 101, 114, 95, 97, 100, 100, 47, 115, 99, 97, 116, 116, 101, 114, 95, 97, 100, 100, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc28)
    %false = arith.constant false loc(#loc28)
    %0 = seq.const_clock  low loc(#loc28)
    %c2_i32 = arith.constant 2 : i32 loc(#loc28)
    %1 = ub.poison : i64 loc(#loc28)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c256 = arith.constant 256 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c7_i32 = arith.constant 7 : i32 loc(#loc2)
    %c255_i32 = arith.constant 255 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc2)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %c256_i32 = arith.constant 256 : i32 loc(#loc2)
    %2 = memref.get_global @str : memref<20xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<20xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<1024xi32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<256xi32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<256xi32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc41)
      %11 = arith.remui %10, %c10_i32 : i32 loc(#loc41)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc41)
      memref.store %11, %alloca[%12] : memref<1024xi32> loc(#loc41)
      %13 = arith.muli %10, %c7_i32 : i32 loc(#loc42)
      %14 = arith.andi %13, %c255_i32 : i32 loc(#loc42)
      memref.store %14, %alloca_0[%12] : memref<1024xi32> loc(#loc42)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc35)
      %16 = arith.cmpi ne, %15, %c1024_i64 : i64 loc(#loc43)
      scf.condition(%16) %15 : i64 loc(#loc33)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc3])):
      scf.yield %arg0 : i64 loc(#loc33)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc33)
    scf.for %arg0 = %c0 to %c256 step %c1 {
      memref.store %c0_i32, %alloca_1[%arg0] : memref<256xi32> loc(#loc44)
    } loc(#loc44)
    scf.for %arg0 = %c0 to %c256 step %c1 {
      memref.store %c0_i32, %alloca_2[%arg0] : memref<256xi32> loc(#loc45)
    } loc(#loc45)
    %cast = memref.cast %alloca : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_3 = memref.cast %alloca_0 : memref<1024xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    %cast_4 = memref.cast %alloca_1 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc29)
    call @_Z15scatter_add_cpuPKjS0_Pjjj(%cast, %cast_3, %cast_4, %c1024_i32, %c256_i32) : (memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, memref<?xi32, strided<[1], offset: ?>>, i32, i32) -> () loc(#loc29)
    %cast_5 = memref.cast %alloca_2 : memref<256xi32> to memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_6, %ready_7 = esi.wrap.vr %cast_3, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_8, %ready_9 = esi.wrap.vr %cast_5, %true : memref<?xi32, strided<[1], offset: ?>> loc(#loc30)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c1024_i32, %true : i32 loc(#loc30)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c256_i32, %true : i32 loc(#loc30)
    %chanOutput_14, %ready_15 = esi.wrap.vr %true, %true : i1 loc(#loc30)
    %5 = handshake.esi_instance @_Z15scatter_add_dsaPKjS0_Pjjj_esi "_Z15scatter_add_dsaPKjS0_Pjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14) : (!esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<memref<?xi32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc30)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc30)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      %11 = memref.load %alloca_1[%10] : memref<256xi32> loc(#loc49)
      %12 = memref.load %alloca_2[%10] : memref<256xi32> loc(#loc49)
      %13 = arith.cmpi eq, %11, %12 : i32 loc(#loc49)
      %14:3 = scf.if %13 -> (i64, i32, i32) {
        %16 = arith.addi %arg0, %c1_i64 : i64 loc(#loc36)
        %17 = arith.cmpi eq, %16, %c256_i64 : i64 loc(#loc36)
        %18 = arith.extui %17 : i1 to i32 loc(#loc34)
        %19 = arith.cmpi ne, %16, %c256_i64 : i64 loc(#loc46)
        %20 = arith.extui %19 : i1 to i32 loc(#loc34)
        scf.yield %16, %18, %20 : i64, i32, i32 loc(#loc49)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc49)
      } loc(#loc49)
      %15 = arith.trunci %14#2 : i32 to i1 loc(#loc34)
      scf.condition(%15) %14#0, %13, %14#1 : i64, i1, i32 loc(#loc34)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc10]), %arg1: i1 loc(fused<#di_lexical_block11>[#loc10]), %arg2: i32 loc(fused<#di_lexical_block11>[#loc10])):
      scf.yield %arg0 : i64 loc(#loc34)
    } loc(#loc34)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc34)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc34)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<20xi8> -> index loc(#loc52)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc52)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc52)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc52)
      scf.yield %c1_i32 : i32 loc(#loc53)
    } loc(#loc34)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<20xi8> -> index loc(#loc31)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc31)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc31)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc31)
    } loc(#loc2)
    return %9 : i32 loc(#loc32)
  } loc(#loc28)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.memset.p0.i64(memref<?xi8, strided<[1], offset: ?>>, i8, i64, i1) loc(#loc2)
  handshake.func @_Z15scatter_add_dsaPKjS0_Pjjj_esi(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg3: i32 loc(fused<#di_subprogram4>[#loc16]), %arg4: i32 loc(fused<#di_subprogram4>[#loc16]), %arg5: i1 loc(fused<#di_subprogram4>[#loc16]), ...) -> i1 attributes {argNames = ["src", "indices", "dst", "N", "dst_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : i1 loc(#loc37)
    %1 = handshake.join %0 : none loc(#loc37)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc50)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc47)
    %5 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %6 = arith.index_cast %3 : i64 to index loc(#loc47)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %6, %5, %7 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %dataResult, %addressResults = handshake.load [%afterValue] %13#0, %15 : index, i32 loc(#loc54)
    %8 = arith.cmpi ult, %dataResult, %arg4 : i32 loc(#loc57)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %12#0, %trueResult_11 : index, i32 loc(#loc59)
    %9 = arith.extui %dataResult : i32 to i64 loc(#loc59)
    %10 = arith.index_cast %9 : i64 to index loc(#loc59)
    %dataResult_2, %addressResults_3 = handshake.load [%10] %14#0, %trueResult_7 : index, i32 loc(#loc59)
    %11 = arith.addi %dataResult_2, %dataResult_0 : i32 loc(#loc59)
    %dataResult_4, %addressResult = handshake.store [%10] %11, %trueResult_7 : index, i32 loc(#loc59)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %13:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc37)
    %14:3 = handshake.extmemory[ld = 1, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult, %addressResults_3) {id = 2 : i32} : (i32, index, index) -> (i32, none, none) loc(#loc37)
    %15 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc47)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %13#1 : none loc(#loc47)
    %16 = handshake.constant %1 {value = 0 : index} : index loc(#loc47)
    %17 = handshake.constant %1 {value = 1 : index} : index loc(#loc47)
    %18 = arith.select %4, %17, %16 : index loc(#loc47)
    %19 = handshake.mux %18 [%falseResult_6, %trueResult] : index, none loc(#loc47)
    %20 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc47)
    %trueResult_7, %falseResult_8 = handshake.cond_br %8, %20 : none loc(#loc57)
    %21 = handshake.join %14#2, %14#1 : none, none loc(#loc59)
    %22 = handshake.constant %20 {value = 0 : index} : index loc(#loc57)
    %23 = handshake.constant %20 {value = 1 : index} : index loc(#loc57)
    %24 = arith.select %8, %23, %22 : index loc(#loc57)
    %25 = handshake.mux %24 [%falseResult_8, %21] : index, none loc(#loc57)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %25 : none loc(#loc47)
    %26 = handshake.mux %18 [%falseResult_10, %trueResult] : index, none loc(#loc47)
    %27 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc47)
    %trueResult_11, %falseResult_12 = handshake.cond_br %8, %27 : none loc(#loc57)
    %28 = handshake.constant %27 {value = 0 : index} : index loc(#loc57)
    %29 = handshake.constant %27 {value = 1 : index} : index loc(#loc57)
    %30 = arith.select %8, %29, %28 : index loc(#loc57)
    %31 = handshake.mux %30 [%falseResult_12, %12#1] : index, none loc(#loc57)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %31 : none loc(#loc47)
    %32 = handshake.mux %18 [%falseResult_14, %trueResult] : index, none loc(#loc47)
    %33 = handshake.join %19, %26, %32 : none, none, none loc(#loc37)
    %34 = handshake.constant %33 {value = true} : i1 loc(#loc37)
    handshake.return %34 : i1 loc(#loc37)
  } loc(#loc37)
  handshake.func @_Z15scatter_add_dsaPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg1: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg2: memref<?xi32, strided<[1], offset: ?>> loc(fused<#di_subprogram4>[#loc16]), %arg3: i32 loc(fused<#di_subprogram4>[#loc16]), %arg4: i32 loc(fused<#di_subprogram4>[#loc16]), %arg5: none loc(fused<#di_subprogram4>[#loc16]), ...) -> none attributes {argNames = ["src", "indices", "dst", "N", "dst_size", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg5 : none loc(#loc37)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi eq, %arg3, %1 : i32 loc(#loc50)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc47)
    %4 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc47)
    %5 = arith.index_cast %2 : i64 to index loc(#loc47)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc47)
    %index, %willContinue = dataflow.stream %5, %4, %6 {step_op = "+=", stop_cond = "!="} loc(#loc47)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc47)
    %dataResult, %addressResults = handshake.load [%afterValue] %12#0, %14 : index, i32 loc(#loc54)
    %7 = arith.cmpi ult, %dataResult, %arg4 : i32 loc(#loc57)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %11#0, %trueResult_11 : index, i32 loc(#loc59)
    %8 = arith.extui %dataResult : i32 to i64 loc(#loc59)
    %9 = arith.index_cast %8 : i64 to index loc(#loc59)
    %dataResult_2, %addressResults_3 = handshake.load [%9] %13#0, %trueResult_7 : index, i32 loc(#loc59)
    %10 = arith.addi %dataResult_2, %dataResult_0 : i32 loc(#loc59)
    %dataResult_4, %addressResult = handshake.store [%9] %10, %trueResult_7 : index, i32 loc(#loc59)
    %11:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults_1) {id = 0 : i32} : (index) -> (i32, none) loc(#loc37)
    %12:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xi32, strided<[1], offset: ?>>) (%addressResults) {id = 1 : i32} : (index) -> (i32, none) loc(#loc37)
    %13:3 = handshake.extmemory[ld = 1, st = 1] (%arg2 : memref<?xi32, strided<[1], offset: ?>>) (%dataResult_4, %addressResult, %addressResults_3) {id = 2 : i32} : (i32, index, index) -> (i32, none, none) loc(#loc37)
    %14 = dataflow.carry %willContinue, %falseResult, %trueResult_5 : i1, none, none -> none loc(#loc47)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %12#1 : none loc(#loc47)
    %15 = handshake.constant %0 {value = 0 : index} : index loc(#loc47)
    %16 = handshake.constant %0 {value = 1 : index} : index loc(#loc47)
    %17 = arith.select %3, %16, %15 : index loc(#loc47)
    %18 = handshake.mux %17 [%falseResult_6, %trueResult] : index, none loc(#loc47)
    %19 = dataflow.carry %willContinue, %falseResult, %trueResult_9 : i1, none, none -> none loc(#loc47)
    %trueResult_7, %falseResult_8 = handshake.cond_br %7, %19 : none loc(#loc57)
    %20 = handshake.join %13#2, %13#1 : none, none loc(#loc59)
    %21 = handshake.constant %19 {value = 0 : index} : index loc(#loc57)
    %22 = handshake.constant %19 {value = 1 : index} : index loc(#loc57)
    %23 = arith.select %7, %22, %21 : index loc(#loc57)
    %24 = handshake.mux %23 [%falseResult_8, %20] : index, none loc(#loc57)
    %trueResult_9, %falseResult_10 = handshake.cond_br %willContinue, %24 : none loc(#loc47)
    %25 = handshake.mux %17 [%falseResult_10, %trueResult] : index, none loc(#loc47)
    %26 = dataflow.carry %willContinue, %falseResult, %trueResult_13 : i1, none, none -> none loc(#loc47)
    %trueResult_11, %falseResult_12 = handshake.cond_br %7, %26 : none loc(#loc57)
    %27 = handshake.constant %26 {value = 0 : index} : index loc(#loc57)
    %28 = handshake.constant %26 {value = 1 : index} : index loc(#loc57)
    %29 = arith.select %7, %28, %27 : index loc(#loc57)
    %30 = handshake.mux %29 [%falseResult_12, %11#1] : index, none loc(#loc57)
    %trueResult_13, %falseResult_14 = handshake.cond_br %willContinue, %30 : none loc(#loc47)
    %31 = handshake.mux %17 [%falseResult_14, %trueResult] : index, none loc(#loc47)
    %32 = handshake.join %18, %25, %31 : none, none, none loc(#loc37)
    handshake.return %32 : none loc(#loc38)
  } loc(#loc37)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @_Z15scatter_add_cpuPKjS0_Pjjj(%arg0: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg1: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg2: memref<?xi32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram5>[#loc22]), %arg3: i32 loc(fused<#di_subprogram5>[#loc22]), %arg4: i32 loc(fused<#di_subprogram5>[#loc22])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc51)
    scf.if %0 {
    } else {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc51)
      %2 = scf.while (%arg5 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg5 : i64 to index loc(#loc55)
        %4 = memref.load %arg1[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc55)
        %5 = arith.cmpi ult, %4, %arg4 : i32 loc(#loc58)
        scf.if %5 {
          %8 = memref.load %arg0[%3] : memref<?xi32, strided<[1], offset: ?>> loc(#loc60)
          %9 = arith.extui %4 : i32 to i64 loc(#loc60)
          %10 = arith.index_cast %9 : i64 to index loc(#loc60)
          %11 = memref.load %arg2[%10] : memref<?xi32, strided<[1], offset: ?>> loc(#loc60)
          %12 = arith.addi %11, %8 : i32 loc(#loc60)
          memref.store %12, %arg2[%10] : memref<?xi32, strided<[1], offset: ?>> loc(#loc60)
        } loc(#loc58)
        %6 = arith.addi %arg5, %c1_i64 : i64 loc(#loc51)
        %7 = arith.cmpi ne, %6, %1 : i64 loc(#loc56)
        scf.condition(%7) %6 : i64 loc(#loc48)
      } do {
      ^bb0(%arg5: i64 loc(fused<#di_lexical_block19>[#loc23])):
        scf.yield %arg5 : i64 loc(#loc48)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc48)
    } loc(#loc48)
    return loc(#loc40)
  } loc(#loc39)
} loc(#loc)
#loc = loc("tests/app/scatter_add/main.cpp":0:0)
#loc1 = loc("tests/app/scatter_add/main.cpp":5:0)
#loc2 = loc(unknown)
#loc4 = loc("tests/app/scatter_add/main.cpp":19:0)
#loc5 = loc("tests/app/scatter_add/main.cpp":20:0)
#loc6 = loc("tests/app/scatter_add/main.cpp":24:0)
#loc7 = loc("tests/app/scatter_add/main.cpp":25:0)
#loc8 = loc("tests/app/scatter_add/main.cpp":29:0)
#loc9 = loc("tests/app/scatter_add/main.cpp":32:0)
#loc11 = loc("tests/app/scatter_add/main.cpp":36:0)
#loc12 = loc("tests/app/scatter_add/main.cpp":37:0)
#loc13 = loc("tests/app/scatter_add/main.cpp":38:0)
#loc14 = loc("tests/app/scatter_add/main.cpp":42:0)
#loc15 = loc("tests/app/scatter_add/main.cpp":44:0)
#loc17 = loc("tests/app/scatter_add/scatter_add.cpp":38:0)
#loc18 = loc("tests/app/scatter_add/scatter_add.cpp":39:0)
#loc19 = loc("tests/app/scatter_add/scatter_add.cpp":40:0)
#loc20 = loc("tests/app/scatter_add/scatter_add.cpp":41:0)
#loc21 = loc("tests/app/scatter_add/scatter_add.cpp":44:0)
#loc24 = loc("tests/app/scatter_add/scatter_add.cpp":23:0)
#loc25 = loc("tests/app/scatter_add/scatter_add.cpp":24:0)
#loc26 = loc("tests/app/scatter_add/scatter_add.cpp":25:0)
#loc27 = loc("tests/app/scatter_add/scatter_add.cpp":28:0)
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file, line = 23>
#loc28 = loc(fused<#di_subprogram3>[#loc1])
#loc29 = loc(fused<#di_subprogram3>[#loc8])
#loc30 = loc(fused<#di_subprogram3>[#loc9])
#loc31 = loc(fused<#di_subprogram3>[#loc14])
#loc32 = loc(fused<#di_subprogram3>[#loc15])
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 18>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 23>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 35>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 18>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 23>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 35>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file1, line = 38>
#loc35 = loc(fused<#di_lexical_block12>[#loc3])
#loc36 = loc(fused<#di_lexical_block14>[#loc10])
#loc38 = loc(fused<#di_subprogram4>[#loc21])
#loc40 = loc(fused<#di_subprogram5>[#loc27])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 36>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 38>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 22>
#loc41 = loc(fused<#di_lexical_block15>[#loc4])
#loc42 = loc(fused<#di_lexical_block15>[#loc5])
#loc43 = loc(fused[#loc33, #loc35])
#loc44 = loc(fused<#di_lexical_block16>[#loc6])
#loc45 = loc(fused<#di_lexical_block16>[#loc7])
#loc46 = loc(fused[#loc34, #loc36])
#loc47 = loc(fused<#di_lexical_block18>[#loc17])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 36>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 38>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 22>
#loc49 = loc(fused<#di_lexical_block20>[#loc11])
#loc50 = loc(fused<#di_lexical_block21>[#loc17])
#loc51 = loc(fused<#di_lexical_block22>[#loc23])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file1, line = 40>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file1, line = 24>
#loc52 = loc(fused<#di_lexical_block23>[#loc12])
#loc53 = loc(fused<#di_lexical_block23>[#loc13])
#loc54 = loc(fused<#di_lexical_block24>[#loc18])
#loc55 = loc(fused<#di_lexical_block25>[#loc24])
#loc56 = loc(fused[#loc48, #loc51])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 40>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 24>
#loc57 = loc(fused<#di_lexical_block26>[#loc19])
#loc58 = loc(fused<#di_lexical_block27>[#loc25])
#loc59 = loc(fused<#di_lexical_block28>[#loc20])
#loc60 = loc(fused<#di_lexical_block29>[#loc26])
