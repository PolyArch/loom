#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "bool", sizeInBits = 8, encoding = DW_ATE_boolean>
#di_file = #llvm.di_file<"tests/app/vecadd/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file3 = #llvm.di_file<"tests/app/vecadd/vecadd.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc12 = loc("tests/app/vecadd/main.cpp":25:0)
#loc15 = loc("tests/app/vecadd/main.cpp":31:0)
#loc19 = loc("tests/app/vecadd/main.cpp":37:0)
#loc22 = loc("tests/app/vecadd/main.cpp":44:0)
#loc33 = loc("tests/app/vecadd/vecadd.cpp":19:0)
#loc37 = loc("tests/app/vecadd/vecadd.cpp":11:0)
#loc38 = loc("tests/app/vecadd/vecadd.cpp":12:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file3, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type1, sizeInBits = 256, elements = #llvm.di_subrange<count = 8 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type1>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type1, sizeInBits = 64>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 25>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 31>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 37>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 44>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file3, line = 23>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file3, line = 12>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "passed", file = #di_file, line = 43, type = #di_basic_type2>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "n", file = #di_file3, line = 20, arg = 4, type = #di_basic_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram3, name = "n", file = #di_file3, line = 11, arg = 4, type = #di_basic_type>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type2, sizeInBits = 64>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block3, file = #di_file, line = 44>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "N", file = #di_file, line = 12, type = #di_derived_type>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file, line = 13, type = #di_composite_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "b", file = #di_file, line = 14, type = #di_composite_type>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram, name = "c_cpu", file = #di_file, line = 15, type = #di_composite_type>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram, name = "c_accel", file = #di_file, line = 16, type = #di_composite_type>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_lexical_block, name = "i", file = #di_file, line = 25, type = #di_basic_type>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "i", file = #di_file, line = 31, type = #di_basic_type>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file, line = 37, type = #di_basic_type>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file, line = 44, type = #di_basic_type>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file3, line = 23, type = #di_basic_type>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram3, name = "c", file = #di_file3, line = 11, arg = 3, type = #di_derived_type3>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_lexical_block5, name = "i", file = #di_file3, line = 12, type = #di_basic_type>
#di_derived_type8 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type5>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 44>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram2, name = "c", file = #di_file3, line = 20, arg = 3, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram3, name = "a", file = #di_file3, line = 11, arg = 1, type = #di_derived_type5>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram3, name = "b", file = #di_file3, line = 11, arg = 2, type = #di_derived_type5>
#di_subroutine_type2 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type5, #di_derived_type5, #di_derived_type3, #di_basic_type>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_lexical_block7, name = "expected", file = #di_file, line = 45, type = #di_basic_type1>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram2, name = "a", file = #di_file3, line = 19, arg = 1, type = #di_derived_type8>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram2, name = "b", file = #di_file3, line = 19, arg = 2, type = #di_derived_type8>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file3, name = "vecadd_cpu", linkageName = "_Z10vecadd_cpuPKfS0_Pfi", file = #di_file3, line = 11, scopeLine = 11, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type2, retainedNodes = #di_local_variable17, #di_local_variable18, #di_local_variable14, #di_local_variable3, #di_local_variable15>
#di_subroutine_type4 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type8, #di_derived_type8, #di_derived_type6, #di_basic_type>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file3, line = 12>
#di_subprogram6 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "main", file = #di_file, line = 11, scopeLine = 11, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable, #di_local_variable12, #di_local_variable19>
#di_subprogram8 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit1, scope = #di_file3, name = "vecadd_dsa", linkageName = "_Z10vecadd_dsaPKfS0_Pfi", file = #di_file3, line = 19, scopeLine = 20, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type4, retainedNodes = #di_local_variable20, #di_local_variable21, #di_local_variable16, #di_local_variable2, #di_local_variable13>
#loc42 = loc(fused<#di_subprogram5>[#loc37])
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 25>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 31>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 37>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 44>
#loc60 = loc(fused<#di_subprogram8>[#loc33])
#loc62 = loc(fused<#di_lexical_block8>[#loc38])
#loc63 = loc(fused<#di_lexical_block9>[#loc12])
#loc64 = loc(fused<#di_lexical_block10>[#loc15])
#loc65 = loc(fused<#di_lexical_block11>[#loc19])
#loc66 = loc(fused<#di_lexical_block12>[#loc22])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @__const.main.a : memref<8xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @__const.main.b : memref<8xf32> = dense<[5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00]> {alignment = 16 : i64} loc(#loc)
  memref.global constant @".str.1" : memref<6xi8> = dense<[97, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.2" : memref<7xi8> = dense<[37, 46, 49, 102, 37, 115, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.3" : memref<3xi8> = dense<[44, 32, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.4" : memref<1xi8> = dense<0> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.6" : memref<6xi8> = dense<[98, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.7" : memref<14xi8> = dense<[99, 32, 61, 32, 97, 32, 43, 32, 98, 32, 61, 32, 91, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str.8" : memref<55xi8> = dense<[69, 82, 82, 79, 82, 58, 32, 99, 91, 37, 100, 93, 32, 61, 32, 37, 46, 49, 102, 32, 40, 99, 112, 117, 41, 32, 37, 46, 49, 102, 32, 40, 97, 99, 99, 101, 108, 41, 44, 32, 101, 120, 112, 101, 99, 116, 101, 100, 32, 37, 46, 49, 102, 10, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str : memref<25xi8> = dense<[86, 101, 99, 116, 111, 114, 32, 65, 100, 100, 105, 116, 105, 111, 110, 32, 82, 101, 115, 117, 108, 116, 115, 58, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.13 : memref<2xi8> = dense<[93, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.14 : memref<32xi8> = dense<[70, 65, 73, 76, 69, 68, 58, 32, 83, 111, 109, 101, 32, 114, 101, 115, 117, 108, 116, 115, 32, 105, 110, 99, 111, 114, 114, 101, 99, 116, 33, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.15 : memref<29xi8> = dense<[80, 65, 83, 83, 69, 68, 58, 32, 65, 108, 108, 32, 114, 101, 115, 117, 108, 116, 115, 32, 99, 111, 114, 114, 101, 99, 116, 33, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @".str" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @".str.1.18" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 118, 101, 99, 97, 100, 100, 47, 118, 101, 99, 97, 100, 100, 46, 99, 112, 112, 0]> loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc2)
    %false = arith.constant false loc(#loc44)
    %0 = seq.const_clock  low loc(#loc44)
    %c0_i32 = arith.constant 0 : i32 loc(#loc44)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c8 = arith.constant 8 : index loc(#loc2)
    %c1 = arith.constant 1 : index loc(#loc2)
    %cst_0 = arith.constant 9.99999974E-6 : f32 loc(#loc2)
    %c1_i8 = arith.constant 1 : i8 loc(#loc2)
    %c8_i64 = arith.constant 8 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c7_i64 = arith.constant 7 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c0_i8 = arith.constant 0 : i8 loc(#loc2)
    %c0 = arith.constant 0 : index loc(#loc2)
    %1 = memref.get_global @__const.main.a : memref<8xf32> loc(#loc2)
    %2 = memref.get_global @__const.main.b : memref<8xf32> loc(#loc2)
    %3 = memref.get_global @str : memref<25xi8> loc(#loc2)
    %4 = memref.get_global @".str.1" : memref<6xi8> loc(#loc2)
    %5 = memref.get_global @".str.4" : memref<1xi8> loc(#loc2)
    %6 = memref.get_global @".str.3" : memref<3xi8> loc(#loc2)
    %7 = memref.get_global @".str.2" : memref<7xi8> loc(#loc2)
    %8 = memref.get_global @str.13 : memref<2xi8> loc(#loc2)
    %9 = memref.get_global @".str.6" : memref<6xi8> loc(#loc2)
    %10 = memref.get_global @".str.7" : memref<14xi8> loc(#loc2)
    %11 = memref.get_global @".str.8" : memref<55xi8> loc(#loc2)
    %12 = memref.get_global @str.15 : memref<29xi8> loc(#loc2)
    %13 = memref.get_global @str.14 : memref<32xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<8xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<8xf32> loc(#loc2)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %45 = memref.load %1[%arg0] : memref<8xf32> loc(#loc45)
      memref.store %45, %alloca[%arg0] : memref<8xf32> loc(#loc45)
    } loc(#loc45)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %45 = memref.load %2[%arg0] : memref<8xf32> loc(#loc46)
      memref.store %45, %alloca_1[%arg0] : memref<8xf32> loc(#loc46)
    } loc(#loc46)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      memref.store %cst, %alloca_2[%arg0] : memref<8xf32> loc(#loc47)
    } loc(#loc47)
    scf.for %arg0 = %c0 to %c8 step %c1 {
      memref.store %cst, %alloca_3[%arg0] : memref<8xf32> loc(#loc48)
    } loc(#loc48)
    %cast = memref.cast %alloca : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    %cast_4 = memref.cast %alloca_1 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    %cast_5 = memref.cast %alloca_2 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc49)
    call @_Z10vecadd_cpuPKfS0_Pfi(%cast, %cast_4, %cast_5, %c8_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32) -> () loc(#loc49)
    %cast_6 = memref.cast %alloca_3 : memref<8xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_7, %ready_8 = esi.wrap.vr %cast_4, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_9, %ready_10 = esi.wrap.vr %cast_6, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc50)
    %chanOutput_11, %ready_12 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc50)
    %chanOutput_13, %ready_14 = esi.wrap.vr %true, %true : i1 loc(#loc50)
    %14 = handshake.esi_instance @_Z10vecadd_dsaPKfS0_Pfi_esi "_Z10vecadd_dsaPKfS0_Pfi_inst0" clk %0 rst %false(%chanOutput, %chanOutput_7, %chanOutput_9, %chanOutput_11, %chanOutput_13) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc50)
    %rawOutput, %valid = esi.unwrap.vr %14, %true : i1 loc(#loc50)
    %intptr = memref.extract_aligned_pointer_as_index %3 : memref<25xi8> -> index loc(#loc51)
    %15 = arith.index_cast %intptr : index to i64 loc(#loc51)
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr loc(#loc51)
    %17 = llvm.call @puts(%16) : (!llvm.ptr) -> i32 loc(#loc51)
    %intptr_15 = memref.extract_aligned_pointer_as_index %4 : memref<6xi8> -> index loc(#loc52)
    %18 = arith.index_cast %intptr_15 : index to i64 loc(#loc52)
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr loc(#loc52)
    %20 = llvm.call @printf(%19) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc52)
    %cast_16 = memref.cast %5 : memref<1xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc78)
    %cast_17 = memref.cast %6 : memref<3xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc78)
    %intptr_18 = memref.extract_aligned_pointer_as_index %7 : memref<7xi8> -> index loc(#loc78)
    %21 = arith.index_cast %intptr_18 : index to i64 loc(#loc78)
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr loc(#loc78)
    %23 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %45 = arith.index_cast %arg0 : i64 to index loc(#loc78)
      %46 = memref.load %alloca[%45] : memref<8xf32> loc(#loc78)
      %47 = arith.extf %46 : f32 to f64 loc(#loc78)
      %48 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc78)
      %49 = arith.select %48, %cast_16, %cast_17 : memref<?xi8, strided<[1], offset: ?>> loc(#loc78)
      %intptr_25 = memref.extract_aligned_pointer_as_index %49 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc78)
      %50 = arith.index_cast %intptr_25 : index to i64 loc(#loc78)
      %51 = llvm.inttoptr %50 : i64 to !llvm.ptr loc(#loc78)
      %52 = llvm.call @printf(%22, %47, %51) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc78)
      %53 = arith.addi %arg0, %c1_i64 : i64 loc(#loc71)
      %54 = arith.cmpi ne, %53, %c8_i64 : i64 loc(#loc79)
      scf.condition(%54) %53 : i64 loc(#loc63)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block9>[#loc12])):
      scf.yield %arg0 : i64 loc(#loc63)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc63)
    %intptr_19 = memref.extract_aligned_pointer_as_index %8 : memref<2xi8> -> index loc(#loc53)
    %24 = arith.index_cast %intptr_19 : index to i64 loc(#loc53)
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr loc(#loc53)
    %26 = llvm.call @puts(%25) : (!llvm.ptr) -> i32 loc(#loc53)
    %intptr_20 = memref.extract_aligned_pointer_as_index %9 : memref<6xi8> -> index loc(#loc54)
    %27 = arith.index_cast %intptr_20 : index to i64 loc(#loc54)
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr loc(#loc54)
    %29 = llvm.call @printf(%28) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc54)
    %30 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %45 = arith.index_cast %arg0 : i64 to index loc(#loc80)
      %46 = memref.load %alloca_1[%45] : memref<8xf32> loc(#loc80)
      %47 = arith.extf %46 : f32 to f64 loc(#loc80)
      %48 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc80)
      %49 = arith.select %48, %cast_16, %cast_17 : memref<?xi8, strided<[1], offset: ?>> loc(#loc80)
      %intptr_25 = memref.extract_aligned_pointer_as_index %49 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc80)
      %50 = arith.index_cast %intptr_25 : index to i64 loc(#loc80)
      %51 = llvm.inttoptr %50 : i64 to !llvm.ptr loc(#loc80)
      %52 = llvm.call @printf(%22, %47, %51) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc80)
      %53 = arith.addi %arg0, %c1_i64 : i64 loc(#loc72)
      %54 = arith.cmpi ne, %53, %c8_i64 : i64 loc(#loc81)
      scf.condition(%54) %53 : i64 loc(#loc64)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block10>[#loc15])):
      scf.yield %arg0 : i64 loc(#loc64)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc64)
    %31 = llvm.call @puts(%25) : (!llvm.ptr) -> i32 loc(#loc55)
    %intptr_21 = memref.extract_aligned_pointer_as_index %10 : memref<14xi8> -> index loc(#loc56)
    %32 = arith.index_cast %intptr_21 : index to i64 loc(#loc56)
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr loc(#loc56)
    %34 = llvm.call @printf(%33) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32 loc(#loc56)
    %35 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %45 = arith.index_cast %arg0 : i64 to index loc(#loc82)
      %46 = memref.load %alloca_3[%45] : memref<8xf32> loc(#loc82)
      %47 = arith.extf %46 : f32 to f64 loc(#loc82)
      %48 = arith.cmpi eq, %arg0, %c7_i64 : i64 loc(#loc82)
      %49 = arith.select %48, %cast_16, %cast_17 : memref<?xi8, strided<[1], offset: ?>> loc(#loc82)
      %intptr_25 = memref.extract_aligned_pointer_as_index %49 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc82)
      %50 = arith.index_cast %intptr_25 : index to i64 loc(#loc82)
      %51 = llvm.inttoptr %50 : i64 to !llvm.ptr loc(#loc82)
      %52 = llvm.call @printf(%22, %47, %51) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, !llvm.ptr) -> i32 loc(#loc82)
      %53 = arith.addi %arg0, %c1_i64 : i64 loc(#loc73)
      %54 = arith.cmpi ne, %53, %c8_i64 : i64 loc(#loc83)
      scf.condition(%54) %53 : i64 loc(#loc65)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block11>[#loc19])):
      scf.yield %arg0 : i64 loc(#loc65)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc65)
    %36 = llvm.call @puts(%25) : (!llvm.ptr) -> i32 loc(#loc57)
    %37:2 = scf.while (%arg0 = %c0_i64, %arg1 = %c1_i8) : (i64, i8) -> (i64, i8) {
      %45 = arith.index_cast %arg0 : i64 to index loc(#loc84)
      %46 = memref.load %alloca[%45] : memref<8xf32> loc(#loc84)
      %47 = memref.load %alloca_1[%45] : memref<8xf32> loc(#loc84)
      %48 = arith.addf %46, %47 : f32 loc(#loc84)
      %49 = memref.load %alloca_2[%45] : memref<8xf32> loc(#loc87)
      %50 = arith.subf %49, %48 : f32 loc(#loc87)
      %51 = math.absf %50 : f32 loc(#loc89)
      %52 = arith.cmpf ogt, %51, %cst_0 : f32 loc(#loc87)
      %53 = scf.if %52 -> (i32) {
        scf.yield %c0_i32 : i32 loc(#loc87)
      } else {
        %58 = memref.load %alloca_3[%45] : memref<8xf32> loc(#loc88)
        %59 = arith.subf %58, %48 : f32 loc(#loc88)
        %60 = math.absf %59 : f32 loc(#loc90)
        %61 = arith.cmpf ogt, %60, %cst_0 : f32 loc(#loc88)
        %62 = arith.xori %61, %true : i1 loc(#loc87)
        %63 = arith.extui %62 : i1 to i32 loc(#loc87)
        scf.yield %63 : i32 loc(#loc87)
      } loc(#loc87)
      %54 = arith.index_castui %53 : i32 to index loc(#loc87)
      %55 = scf.index_switch %54 -> i8 
      case 0 {
        %58 = arith.extf %49 : f32 to f64 loc(#loc91)
        %59 = memref.load %alloca_3[%45] : memref<8xf32> loc(#loc91)
        %60 = arith.extf %59 : f32 to f64 loc(#loc91)
        %61 = arith.extf %48 : f32 to f64 loc(#loc91)
        %62 = arith.trunci %arg0 : i64 to i32 loc(#loc92)
        %intptr_25 = memref.extract_aligned_pointer_as_index %11 : memref<55xi8> -> index loc(#loc92)
        %63 = arith.index_cast %intptr_25 : index to i64 loc(#loc92)
        %64 = llvm.inttoptr %63 : i64 to !llvm.ptr loc(#loc92)
        %65 = llvm.call @printf(%64, %62, %58, %60, %61) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, f64, f64, f64) -> i32 loc(#loc92)
        scf.yield %c0_i8 : i8 loc(#loc93)
      }
      default {
        scf.yield %arg1 : i8 loc(#loc87)
      } loc(#loc87)
      %56 = arith.addi %arg0, %c1_i64 : i64 loc(#loc74)
      %57 = arith.cmpi ne, %56, %c8_i64 : i64 loc(#loc85)
      scf.condition(%57) %56, %55 : i64, i8 loc(#loc66)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block12>[#loc22]), %arg1: i8 loc(fused<#di_lexical_block12>[#loc22])):
      scf.yield %arg0, %arg1 : i64, i8 loc(#loc66)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc66)
    %38 = arith.trunci %37#1 : i8 to i1 loc(#loc67)
    %cast_22 = memref.cast %12 : memref<29xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc68)
    %cast_23 = memref.cast %13 : memref<32xi8> to memref<?xi8, strided<[1], offset: ?>> loc(#loc68)
    %39 = arith.select %38, %cast_22, %cast_23 : memref<?xi8, strided<[1], offset: ?>> loc(#loc68)
    %40 = arith.xori %38, %true : i1 loc(#loc68)
    %41 = arith.extui %40 : i1 to i32 loc(#loc68)
    %intptr_24 = memref.extract_aligned_pointer_as_index %39 : memref<?xi8, strided<[1], offset: ?>> -> index loc(#loc68)
    %42 = arith.index_cast %intptr_24 : index to i64 loc(#loc68)
    %43 = llvm.inttoptr %42 : i64 to !llvm.ptr loc(#loc68)
    %44 = llvm.call @puts(%43) : (!llvm.ptr) -> i32 loc(#loc68)
    return %41 : i32 loc(#loc58)
  } loc(#loc44)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.memcpy.p0.p0.i64(memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, memref<?xi8, strided<[1], offset: ?>> {loom.noalias}, i64, i1) loc(#loc2)
  func.func private @llvm.memset.p0.i64(memref<?xi8, strided<[1], offset: ?>>, i8, i64, i1) loc(#loc2)
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} loc(#loc59)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  handshake.func @_Z10vecadd_dsaPKfS0_Pfi_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg3: i32 loc(fused<#di_subprogram8>[#loc33]), %arg4: i1 loc(fused<#di_subprogram8>[#loc33]), ...) -> i1 attributes {argNames = ["a", "b", "c", "n", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : i1 loc(#loc60)
    %1 = handshake.join %0 : none loc(#loc60)
    %2 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.cmpi sgt, %arg3, %2 : i32 loc(#loc75)
    %trueResult, %falseResult = handshake.cond_br %4, %1 : none loc(#loc69)
    handshake.sink %falseResult : none loc(#loc69)
    %5 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc69)
    %6 = arith.index_cast %3 : i64 to index loc(#loc69)
    %7 = arith.index_cast %arg3 : i32 to index loc(#loc69)
    %index, %willContinue = dataflow.stream %6, %5, %7 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=0 max=0"], step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc69)
    %dataResult, %addressResults = handshake.load [%afterValue] %9#0, %12 : index, f32 loc(#loc86)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %10#0, %19 : index, f32 loc(#loc86)
    %8 = arith.addf %dataResult, %dataResult_0 : f32 loc(#loc86)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %8, %17 : index, f32 loc(#loc86)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc60)
    %10:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc60)
    %11 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc60)
    %12 = dataflow.carry %willContinue, %trueResult, %trueResult_3 : i1, none, none -> none loc(#loc69)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %9#1 : none loc(#loc69)
    %13 = handshake.constant %1 {value = 0 : index} : index loc(#loc69)
    %14 = handshake.constant %1 {value = 1 : index} : index loc(#loc69)
    %15 = arith.select %4, %14, %13 : index loc(#loc69)
    %16 = handshake.mux %15 [%falseResult, %falseResult_4] : index, none loc(#loc69)
    %17 = dataflow.carry %willContinue, %trueResult, %trueResult_5 : i1, none, none -> none loc(#loc69)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %11 : none loc(#loc69)
    %18 = handshake.mux %15 [%falseResult, %falseResult_6] : index, none loc(#loc69)
    %19 = dataflow.carry %willContinue, %trueResult, %trueResult_7 : i1, none, none -> none loc(#loc69)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %10#1 : none loc(#loc69)
    %20 = handshake.mux %15 [%falseResult, %falseResult_8] : index, none loc(#loc69)
    %21 = handshake.join %16, %18, %20 : none, none, none loc(#loc60)
    %22 = handshake.constant %21 {value = true} : i1 loc(#loc60)
    handshake.return %22 : i1 loc(#loc60)
  } loc(#loc60)
  handshake.func @_Z10vecadd_dsaPKfS0_Pfi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram8>[#loc33]), %arg3: i32 loc(fused<#di_subprogram8>[#loc33]), %arg4: none loc(fused<#di_subprogram8>[#loc33]), ...) -> none attributes {argNames = ["a", "b", "c", "n", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg4 : none loc(#loc60)
    %1 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %3 = arith.cmpi sgt, %arg3, %1 : i32 loc(#loc75)
    %trueResult, %falseResult = handshake.cond_br %3, %0 : none loc(#loc69)
    handshake.sink %falseResult : none loc(#loc69)
    %4 = handshake.constant %trueResult {value = 1 : index} : index loc(#loc69)
    %5 = arith.index_cast %2 : i64 to index loc(#loc69)
    %6 = arith.index_cast %arg3 : i32 to index loc(#loc69)
    %index, %willContinue = dataflow.stream %5, %4, %6 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1", "loom.loop.tripcount typical=256 avg=256 min=0 max=0"], step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc69)
    %dataResult, %addressResults = handshake.load [%afterValue] %8#0, %11 : index, f32 loc(#loc86)
    %dataResult_0, %addressResults_1 = handshake.load [%afterValue] %9#0, %18 : index, f32 loc(#loc86)
    %7 = arith.addf %dataResult, %dataResult_0 : f32 loc(#loc86)
    %dataResult_2, %addressResult = handshake.store [%afterValue] %7, %16 : index, f32 loc(#loc86)
    %8:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc60)
    %9:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_1) {id = 1 : i32} : (index) -> (f32, none) loc(#loc60)
    %10 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_2, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc60)
    %11 = dataflow.carry %willContinue, %trueResult, %trueResult_3 : i1, none, none -> none loc(#loc69)
    %trueResult_3, %falseResult_4 = handshake.cond_br %willContinue, %8#1 : none loc(#loc69)
    %12 = handshake.constant %0 {value = 0 : index} : index loc(#loc69)
    %13 = handshake.constant %0 {value = 1 : index} : index loc(#loc69)
    %14 = arith.select %3, %13, %12 : index loc(#loc69)
    %15 = handshake.mux %14 [%falseResult, %falseResult_4] : index, none loc(#loc69)
    %16 = dataflow.carry %willContinue, %trueResult, %trueResult_5 : i1, none, none -> none loc(#loc69)
    %trueResult_5, %falseResult_6 = handshake.cond_br %willContinue, %10 : none loc(#loc69)
    %17 = handshake.mux %14 [%falseResult, %falseResult_6] : index, none loc(#loc69)
    %18 = dataflow.carry %willContinue, %trueResult, %trueResult_7 : i1, none, none -> none loc(#loc69)
    %trueResult_7, %falseResult_8 = handshake.cond_br %willContinue, %9#1 : none loc(#loc69)
    %19 = handshake.mux %14 [%falseResult, %falseResult_8] : index, none loc(#loc69)
    %20 = handshake.join %15, %17, %19 : none, none, none loc(#loc60)
    handshake.return %20 : none loc(#loc61)
  } loc(#loc60)
  func.func @_Z10vecadd_cpuPKfS0_Pfi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc37]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc37]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc37]), %arg3: i32 loc(fused<#di_subprogram5>[#loc37])) {
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.cmpi sgt, %arg3, %c0_i32 : i32 loc(#loc70)
    scf.if %0 {
      %1 = arith.extui %arg3 : i32 to i64 loc(#loc70)
      %2 = scf.while (%arg4 = %c0_i64) : (i64) -> i64 {
        %3 = arith.index_cast %arg4 : i64 to index loc(#loc76)
        %4 = memref.load %arg0[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc76)
        %5 = memref.load %arg1[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc76)
        %6 = arith.addf %4, %5 : f32 loc(#loc76)
        memref.store %6, %arg2[%3] : memref<?xf32, strided<[1], offset: ?>> loc(#loc76)
        %7 = arith.addi %arg4, %c1_i64 : i64 loc(#loc70)
        %8 = arith.cmpi ne, %7, %1 : i64 loc(#loc77)
        scf.condition(%8) %7 : i64 loc(#loc62)
      } do {
      ^bb0(%arg4: i64 loc(fused<#di_lexical_block8>[#loc38])):
        scf.yield %arg4 : i64 loc(#loc62)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc62)
    } loc(#loc62)
    return loc(#loc43)
  } loc(#loc42)
} loc(#loc)
#di_basic_type3 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file1 = #llvm.di_file<"/usr/lib/gcc/x86_64-redhat-linux/14/../../../../include/c++/14/cmath" in "">
#di_file2 = #llvm.di_file<"/usr/include/stdio.h" in "">
#di_namespace = #llvm.di_namespace<name = "std", exportSymbols = false>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[8]<>, isRecSelf = true>
#loc = loc("tests/app/vecadd/main.cpp":0:0)
#loc1 = loc("tests/app/vecadd/main.cpp":11:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/vecadd/main.cpp":13:0)
#loc4 = loc("tests/app/vecadd/main.cpp":14:0)
#loc5 = loc("tests/app/vecadd/main.cpp":15:0)
#loc6 = loc("tests/app/vecadd/main.cpp":16:0)
#loc7 = loc("tests/app/vecadd/main.cpp":19:0)
#loc8 = loc("tests/app/vecadd/main.cpp":20:0)
#loc9 = loc("tests/app/vecadd/main.cpp":23:0)
#loc10 = loc("tests/app/vecadd/main.cpp":24:0)
#loc11 = loc("tests/app/vecadd/main.cpp":26:0)
#loc13 = loc("tests/app/vecadd/main.cpp":28:0)
#loc14 = loc("tests/app/vecadd/main.cpp":30:0)
#loc16 = loc("tests/app/vecadd/main.cpp":32:0)
#loc17 = loc("tests/app/vecadd/main.cpp":34:0)
#loc18 = loc("tests/app/vecadd/main.cpp":36:0)
#loc20 = loc("tests/app/vecadd/main.cpp":38:0)
#loc21 = loc("tests/app/vecadd/main.cpp":40:0)
#loc23 = loc("tests/app/vecadd/main.cpp":45:0)
#loc24 = loc("tests/app/vecadd/main.cpp":46:0)
#loc25 = loc("/usr/lib/gcc/x86_64-redhat-linux/14/../../../../include/c++/14/cmath":239:0)
#loc26 = loc("tests/app/vecadd/main.cpp":47:0)
#loc27 = loc("tests/app/vecadd/main.cpp":49:0)
#loc28 = loc("tests/app/vecadd/main.cpp":48:0)
#loc29 = loc("tests/app/vecadd/main.cpp":51:0)
#loc30 = loc("tests/app/vecadd/main.cpp":54:0)
#loc31 = loc("tests/app/vecadd/main.cpp":61:0)
#loc32 = loc("/usr/include/stdio.h":363:0)
#loc34 = loc("tests/app/vecadd/vecadd.cpp":23:0)
#loc35 = loc("tests/app/vecadd/vecadd.cpp":24:0)
#loc36 = loc("tests/app/vecadd/vecadd.cpp":26:0)
#loc39 = loc("tests/app/vecadd/vecadd.cpp":13:0)
#loc40 = loc("tests/app/vecadd/vecadd.cpp":15:0)
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type3>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram1, name = "__x", file = #di_file1, line = 238, arg = 1, type = #di_basic_type1>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_basic_type1, #di_basic_type1>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type1, sizeInBits = 64>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[8]<>, id = distinct[9]<>, compileUnit = #di_compile_unit, scope = #di_namespace, name = "fabs", linkageName = "_ZSt4fabsf", file = #di_file1, line = 238, scopeLine = 239, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable1>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type4>
#loc41 = loc(fused<#di_subprogram4>[#loc25])
#di_subroutine_type3 = #llvm.di_subroutine_type<types = #di_basic_type, #di_derived_type7, #di_null_type>
#di_subprogram7 = #llvm.di_subprogram<scope = #di_file2, name = "printf", file = #di_file2, line = 363, subprogramFlags = Optimized, type = #di_subroutine_type3>
#loc43 = loc(fused<#di_subprogram5>[#loc40])
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_subprogram6, file = #di_file, line = 54>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram8, file = #di_file3, line = 23>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file3, line = 12>
#loc44 = loc(fused<#di_subprogram6>[#loc1])
#loc45 = loc(fused<#di_subprogram6>[#loc3])
#loc46 = loc(fused<#di_subprogram6>[#loc4])
#loc47 = loc(fused<#di_subprogram6>[#loc5])
#loc48 = loc(fused<#di_subprogram6>[#loc6])
#loc49 = loc(fused<#di_subprogram6>[#loc7])
#loc50 = loc(fused<#di_subprogram6>[#loc8])
#loc51 = loc(fused<#di_subprogram6>[#loc9])
#loc52 = loc(fused<#di_subprogram6>[#loc10])
#loc53 = loc(fused<#di_subprogram6>[#loc13])
#loc54 = loc(fused<#di_subprogram6>[#loc14])
#loc55 = loc(fused<#di_subprogram6>[#loc17])
#loc56 = loc(fused<#di_subprogram6>[#loc18])
#loc57 = loc(fused<#di_subprogram6>[#loc21])
#loc58 = loc(fused<#di_subprogram6>[#loc31])
#loc59 = loc(fused<#di_subprogram7>[#loc32])
#loc61 = loc(fused<#di_subprogram8>[#loc36])
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 25>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 31>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 37>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 44>
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file3, line = 23>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file3, line = 12>
#loc67 = loc(fused<#di_lexical_block13>[#loc30])
#loc68 = loc(fused<#di_lexical_block13>[#loc])
#loc69 = loc(fused<#di_lexical_block14>[#loc34])
#loc70 = loc(fused<#di_lexical_block15>[#loc38])
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 25>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 31>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 37>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 44>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file3, line = 23>
#loc71 = loc(fused<#di_lexical_block16>[#loc12])
#loc72 = loc(fused<#di_lexical_block17>[#loc15])
#loc73 = loc(fused<#di_lexical_block18>[#loc19])
#loc74 = loc(fused<#di_lexical_block19>[#loc22])
#loc75 = loc(fused<#di_lexical_block20>[#loc34])
#loc76 = loc(fused<#di_lexical_block21>[#loc39])
#loc77 = loc(fused[#loc62, #loc70])
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 46>
#loc78 = loc(fused<#di_lexical_block22>[#loc11])
#loc79 = loc(fused[#loc63, #loc71])
#loc80 = loc(fused<#di_lexical_block23>[#loc16])
#loc81 = loc(fused[#loc64, #loc72])
#loc82 = loc(fused<#di_lexical_block24>[#loc20])
#loc83 = loc(fused[#loc65, #loc73])
#loc84 = loc(fused<#di_lexical_block25>[#loc23])
#loc85 = loc(fused[#loc66, #loc74])
#loc86 = loc(fused<#di_lexical_block26>[#loc35])
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 47>
#loc87 = loc(fused<#di_lexical_block27>[#loc24])
#loc88 = loc(fused<#di_lexical_block27>[#loc26])
#loc89 = loc(callsite(#loc41 at #loc87))
#loc90 = loc(callsite(#loc41 at #loc88))
#loc91 = loc(fused<#di_lexical_block28>[#loc27])
#loc92 = loc(fused<#di_lexical_block28>[#loc28])
#loc93 = loc(fused<#di_lexical_block28>[#loc29])
