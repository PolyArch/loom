#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/im2col/im2col.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/im2col/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/im2col/im2col.cpp":17:0)
#loc5 = loc("tests/app/im2col/im2col.cpp":29:0)
#loc6 = loc("tests/app/im2col/im2col.cpp":30:0)
#loc7 = loc("tests/app/im2col/im2col.cpp":31:0)
#loc8 = loc("tests/app/im2col/im2col.cpp":34:0)
#loc10 = loc("tests/app/im2col/im2col.cpp":35:0)
#loc14 = loc("tests/app/im2col/im2col.cpp":50:0)
#loc27 = loc("tests/app/im2col/main.cpp":27:0)
#loc31 = loc("tests/app/im2col/main.cpp":38:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 6144, elements = #llvm.di_subrange<count = 192 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 31104, elements = #llvm.di_subrange<count = 972 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 63>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 27>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 38>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 29>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 63>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 20, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 23, type = #di_composite_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 24, type = #di_composite_type1>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block4, file = #di_file, line = 29>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 63>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 18, arg = 2, type = #di_derived_type4>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "OH", file = #di_file, line = 26, type = #di_derived_type5>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "OW", file = #di_file, line = 27, type = #di_derived_type5>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_lexical_block, name = "c", file = #di_file, line = 29, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 51, arg = 2, type = #di_derived_type4>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "OH", file = #di_file, line = 59, type = #di_derived_type5>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "OW", file = #di_file, line = 60, type = #di_derived_type5>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "c", file = #di_file, line = 63, type = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 27, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 38, type = #di_derived_type5>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 30>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 65>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 17, arg = 1, type = #di_derived_type6>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_subprogram, name = "C", file = #di_file, line = 19, arg = 3, type = #di_derived_type7>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "H", file = #di_file, line = 20, arg = 4, type = #di_derived_type7>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "W", file = #di_file, line = 21, arg = 5, type = #di_derived_type7>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "KH", file = #di_file, line = 22, arg = 6, type = #di_derived_type7>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "KW", file = #di_file, line = 23, arg = 7, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "stride_h", file = #di_file, line = 24, arg = 8, type = #di_derived_type7>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "stride_w", file = #di_file, line = 25, arg = 9, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 50, arg = 1, type = #di_derived_type6>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram1, name = "C", file = #di_file, line = 52, arg = 3, type = #di_derived_type7>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram1, name = "H", file = #di_file, line = 53, arg = 4, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram1, name = "W", file = #di_file, line = 54, arg = 5, type = #di_derived_type7>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "KH", file = #di_file, line = 55, arg = 6, type = #di_derived_type7>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram1, name = "KW", file = #di_file, line = 56, arg = 7, type = #di_derived_type7>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "stride_h", file = #di_file, line = 57, arg = 8, type = #di_derived_type7>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram1, name = "stride_w", file = #di_file, line = 58, arg = 9, type = #di_derived_type7>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram2, name = "C", file = #di_file1, line = 7, type = #di_derived_type7>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram2, name = "H", file = #di_file1, line = 8, type = #di_derived_type7>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_subprogram2, name = "W", file = #di_file1, line = 9, type = #di_derived_type7>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram2, name = "KH", file = #di_file1, line = 10, type = #di_derived_type7>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram2, name = "KW", file = #di_file1, line = 11, type = #di_derived_type7>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_subprogram2, name = "stride_h", file = #di_file1, line = 12, type = #di_derived_type7>
#di_local_variable35 = #llvm.di_local_variable<scope = #di_subprogram2, name = "stride_w", file = #di_file1, line = 13, type = #di_derived_type7>
#di_local_variable36 = #llvm.di_local_variable<scope = #di_subprogram2, name = "OH", file = #di_file1, line = 14, type = #di_derived_type7>
#di_local_variable37 = #llvm.di_local_variable<scope = #di_subprogram2, name = "OW", file = #di_file1, line = 15, type = #di_derived_type7>
#di_local_variable38 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_rows", file = #di_file1, line = 16, type = #di_derived_type7>
#di_local_variable39 = #llvm.di_local_variable<scope = #di_subprogram2, name = "output_cols", file = #di_file1, line = 17, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type4, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 30>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 65>
#di_local_variable40 = #llvm.di_local_variable<scope = #di_lexical_block8, name = "kh", file = #di_file, line = 30, type = #di_derived_type5>
#di_local_variable41 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "kh", file = #di_file, line = 65, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable29, #di_local_variable30, #di_local_variable31, #di_local_variable32, #di_local_variable33, #di_local_variable34, #di_local_variable35, #di_local_variable36, #di_local_variable37, #di_local_variable38, #di_local_variable39, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable11, #di_local_variable12>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 30>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 65>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 27>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 38>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 31>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 67>
#loc42 = loc(fused<#di_lexical_block14>[#loc27])
#loc43 = loc(fused<#di_lexical_block15>[#loc31])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file, line = 31>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file, line = 67>
#di_local_variable42 = #llvm.di_local_variable<scope = #di_lexical_block16, name = "kw", file = #di_file, line = 31, type = #di_derived_type5>
#di_local_variable43 = #llvm.di_local_variable<scope = #di_lexical_block17, name = "kw", file = #di_file, line = 67, type = #di_derived_type5>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file, line = 31>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file, line = 67>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 34>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block25, file = #di_file, line = 70>
#di_local_variable44 = #llvm.di_local_variable<scope = #di_lexical_block24, name = "row", file = #di_file, line = 32, type = #di_derived_type5>
#di_local_variable45 = #llvm.di_local_variable<scope = #di_lexical_block25, name = "row", file = #di_file, line = 68, type = #di_derived_type5>
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file, line = 34>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 70>
#di_local_variable46 = #llvm.di_local_variable<scope = #di_lexical_block27, name = "oh", file = #di_file, line = 34, type = #di_derived_type5>
#di_local_variable47 = #llvm.di_local_variable<scope = #di_lexical_block28, name = "oh", file = #di_file, line = 70, type = #di_derived_type5>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file, line = 34>
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 70>
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 35>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block33, file = #di_file, line = 71>
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 35>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 71>
#di_local_variable48 = #llvm.di_local_variable<scope = #di_lexical_block34, name = "ow", file = #di_file, line = 35, type = #di_derived_type5>
#di_local_variable49 = #llvm.di_local_variable<scope = #di_lexical_block35, name = "ow", file = #di_file, line = 71, type = #di_derived_type5>
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 35>
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file, line = 71>
#di_local_variable50 = #llvm.di_local_variable<scope = #di_lexical_block38, name = "h", file = #di_file, line = 36, type = #di_derived_type5>
#di_local_variable51 = #llvm.di_local_variable<scope = #di_lexical_block38, name = "w", file = #di_file, line = 37, type = #di_derived_type5>
#di_local_variable52 = #llvm.di_local_variable<scope = #di_lexical_block38, name = "col", file = #di_file, line = 38, type = #di_derived_type5>
#di_local_variable53 = #llvm.di_local_variable<scope = #di_lexical_block39, name = "h", file = #di_file, line = 72, type = #di_derived_type5>
#di_local_variable54 = #llvm.di_local_variable<scope = #di_lexical_block39, name = "w", file = #di_file, line = 73, type = #di_derived_type5>
#di_local_variable55 = #llvm.di_local_variable<scope = #di_lexical_block39, name = "col", file = #di_file, line = 74, type = #di_derived_type5>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "im2col_cpu", linkageName = "_Z10im2col_cpuPKfPfjjjjjjj", file = #di_file, line = 17, scopeLine = 25, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable13, #di_local_variable3, #di_local_variable14, #di_local_variable15, #di_local_variable16, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable4, #di_local_variable5, #di_local_variable6, #di_local_variable40, #di_local_variable42, #di_local_variable44, #di_local_variable46, #di_local_variable48, #di_local_variable50, #di_local_variable51, #di_local_variable52>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "im2col_dsa", linkageName = "_Z10im2col_dsaPKfPfjjjjjjj", file = #di_file, line = 50, scopeLine = 58, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable21, #di_local_variable7, #di_local_variable22, #di_local_variable23, #di_local_variable24, #di_local_variable25, #di_local_variable26, #di_local_variable27, #di_local_variable28, #di_local_variable8, #di_local_variable9, #di_local_variable10, #di_local_variable41, #di_local_variable43, #di_local_variable45, #di_local_variable47, #di_local_variable49, #di_local_variable53, #di_local_variable54, #di_local_variable55>
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 29>
#loc52 = loc(fused<#di_subprogram4>[#loc1])
#loc56 = loc(fused<#di_subprogram5>[#loc14])
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file, line = 29>
#loc60 = loc(fused<#di_lexical_block40>[#loc5])
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_lexical_block42, file = #di_file, line = 29>
#di_lexical_block46 = #llvm.di_lexical_block<scope = #di_lexical_block44, file = #di_file, line = 30>
#di_lexical_block48 = #llvm.di_lexical_block<scope = #di_lexical_block46, file = #di_file, line = 30>
#loc65 = loc(fused<#di_lexical_block46>[#loc6])
#di_lexical_block50 = #llvm.di_lexical_block<scope = #di_lexical_block48, file = #di_file, line = 30>
#di_lexical_block52 = #llvm.di_lexical_block<scope = #di_lexical_block50, file = #di_file, line = 31>
#di_lexical_block54 = #llvm.di_lexical_block<scope = #di_lexical_block52, file = #di_file, line = 31>
#loc69 = loc(fused<#di_lexical_block52>[#loc7])
#di_lexical_block56 = #llvm.di_lexical_block<scope = #di_lexical_block54, file = #di_file, line = 31>
#di_lexical_block58 = #llvm.di_lexical_block<scope = #di_lexical_block56, file = #di_file, line = 34>
#di_lexical_block60 = #llvm.di_lexical_block<scope = #di_lexical_block58, file = #di_file, line = 34>
#loc75 = loc(fused<#di_lexical_block58>[#loc8])
#di_lexical_block62 = #llvm.di_lexical_block<scope = #di_lexical_block60, file = #di_file, line = 34>
#di_lexical_block64 = #llvm.di_lexical_block<scope = #di_lexical_block62, file = #di_file, line = 35>
#loc78 = loc(fused<#di_lexical_block64>[#loc10])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 52, 44, 98, 108, 111, 99, 107, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<28xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 105, 109, 50, 99, 111, 108, 47, 105, 109, 50, 99, 111, 108, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<15xi8> = dense<[105, 109, 50, 99, 111, 108, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<15xi8> = dense<[105, 109, 50, 99, 111, 108, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z10im2col_cpuPKfPfjjjjjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: i32 loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: i32 loc(fused<#di_subprogram4>[#loc1]), %arg5: i32 loc(fused<#di_subprogram4>[#loc1]), %arg6: i32 loc(fused<#di_subprogram4>[#loc1]), %arg7: i32 loc(fused<#di_subprogram4>[#loc1]), %arg8: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.subi %arg3, %arg5 : i32 loc(#loc53)
    %1 = arith.divui %0, %arg7 : i32 loc(#loc53)
    %2 = arith.addi %1, %c1_i32 : i32 loc(#loc53)
    %3 = arith.subi %arg4, %arg6 : i32 loc(#loc54)
    %4 = arith.divui %3, %arg8 : i32 loc(#loc54)
    %5 = arith.addi %4, %c1_i32 : i32 loc(#loc54)
    %6 = arith.cmpi eq, %arg2, %c0_i32 : i32 loc(#loc62)
    scf.if %6 {
    } else {
      %7 = arith.cmpi eq, %arg5, %c0_i32 : i32 loc(#loc2)
      %8 = arith.cmpi eq, %arg6, %c0_i32 : i32 loc(#loc2)
      %9 = arith.cmpi eq, %2, %c0_i32 : i32 loc(#loc2)
      %10 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc2)
      %11 = arith.extui %5 : i32 to i64 loc(#loc60)
      %12 = scf.while (%arg9 = %c0_i32) : (i32) -> i32 {
        scf.if %7 {
        } else {
          %15 = arith.muli %arg9, %arg5 : i32 loc(#loc2)
          %16 = arith.muli %arg9, %arg3 : i32 loc(#loc2)
          %17 = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
            scf.if %8 {
            } else {
              %20 = arith.addi %arg10, %15 : i32 loc(#loc2)
              %21 = arith.muli %20, %arg6 : i32 loc(#loc2)
              %22 = arith.addi %arg10, %16 : i32 loc(#loc2)
              %23 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
                scf.if %9 {
                } else {
                  %26 = arith.addi %arg11, %21 : i32 loc(#loc72)
                  %27 = arith.muli %26, %2 : i32 loc(#loc2)
                  %28 = scf.while (%arg12 = %c0_i32) : (i32) -> i32 {
                    scf.if %10 {
                    } else {
                      %31 = arith.muli %arg12, %arg7 : i32 loc(#loc2)
                      %32 = arith.addi %22, %31 : i32 loc(#loc2)
                      %33 = arith.muli %32, %arg4 : i32 loc(#loc2)
                      %34 = arith.addi %33, %arg11 : i32 loc(#loc2)
                      %35 = arith.addi %arg12, %27 : i32 loc(#loc2)
                      %36 = arith.muli %35, %5 : i32 loc(#loc2)
                      %37 = scf.while (%arg13 = %c0_i64) : (i64) -> i64 {
                        %38 = arith.trunci %arg13 : i64 to i32 loc(#loc81)
                        %39 = arith.muli %arg8, %38 : i32 loc(#loc81)
                        %40 = arith.addi %34, %39 : i32 loc(#loc82)
                        %41 = arith.extui %40 : i32 to i64 loc(#loc82)
                        %42 = arith.index_cast %41 : i64 to index loc(#loc82)
                        %43 = memref.load %arg0[%42] : memref<?xf32, strided<[1], offset: ?>> loc(#loc82)
                        %44 = arith.addi %36, %38 : i32 loc(#loc82)
                        %45 = arith.extui %44 : i32 to i64 loc(#loc82)
                        %46 = arith.index_cast %45 : i64 to index loc(#loc82)
                        memref.store %43, %arg1[%46] : memref<?xf32, strided<[1], offset: ?>> loc(#loc82)
                        %47 = arith.addi %arg13, %c1_i64 : i64 loc(#loc80)
                        %48 = arith.cmpi ult, %47, %11 : i64 loc(#loc80)
                        scf.condition(%48) %47 : i64 loc(#loc78)
                      } do {
                      ^bb0(%arg13: i64 loc(fused<#di_lexical_block64>[#loc10])):
                        scf.yield %arg13 : i64 loc(#loc78)
                      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc78)
                    } loc(#loc78)
                    %29 = arith.addi %arg12, %c1_i32 : i32 loc(#loc77)
                    %30 = arith.cmpi ult, %29, %2 : i32 loc(#loc77)
                    scf.condition(%30) %29 : i32 loc(#loc75)
                  } do {
                  ^bb0(%arg12: i32 loc(fused<#di_lexical_block58>[#loc8])):
                    scf.yield %arg12 : i32 loc(#loc75)
                  } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc75)
                } loc(#loc75)
                %24 = arith.addi %arg11, %c1_i32 : i32 loc(#loc71)
                %25 = arith.cmpi ne, %24, %arg6 : i32 loc(#loc73)
                scf.condition(%25) %24 : i32 loc(#loc69)
              } do {
              ^bb0(%arg11: i32 loc(fused<#di_lexical_block52>[#loc7])):
                scf.yield %arg11 : i32 loc(#loc69)
              } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc69)
            } loc(#loc69)
            %18 = arith.addi %arg10, %c1_i32 : i32 loc(#loc67)
            %19 = arith.cmpi ne, %18, %arg5 : i32 loc(#loc68)
            scf.condition(%19) %18 : i32 loc(#loc65)
          } do {
          ^bb0(%arg10: i32 loc(fused<#di_lexical_block46>[#loc6])):
            scf.yield %arg10 : i32 loc(#loc65)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc65)
        } loc(#loc65)
        %13 = arith.addi %arg9, %c1_i32 : i32 loc(#loc62)
        %14 = arith.cmpi ne, %13, %arg2 : i32 loc(#loc64)
        scf.condition(%14) %13 : i32 loc(#loc60)
      } do {
      ^bb0(%arg9: i32 loc(fused<#di_lexical_block40>[#loc5])):
        scf.yield %arg9 : i32 loc(#loc60)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc60)
    } loc(#loc60)
    return loc(#loc55)
  } loc(#loc52)
  handshake.func @_Z10im2col_dsaPKfPfjjjjjjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg2: i32 loc(fused<#di_subprogram5>[#loc14]), %arg3: i32 loc(fused<#di_subprogram5>[#loc14]), %arg4: i32 loc(fused<#di_subprogram5>[#loc14]), %arg5: i32 loc(fused<#di_subprogram5>[#loc14]), %arg6: i32 loc(fused<#di_subprogram5>[#loc14]), %arg7: i32 loc(fused<#di_subprogram5>[#loc14]), %arg8: i32 loc(fused<#di_subprogram5>[#loc14]), %arg9: i1 loc(fused<#di_subprogram5>[#loc14]), ...) -> i1 attributes {argNames = ["input", "output", "C", "H", "W", "KH", "KW", "stride_h", "stride_w", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg9 : i1 loc(#loc56)
    %1 = handshake.join %0 : none loc(#loc56)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.subi %arg3, %arg5 : i32 loc(#loc57)
    %6 = arith.divui %5, %arg7 : i32 loc(#loc57)
    %7 = arith.addi %6, %2 : i32 loc(#loc57)
    %8 = arith.subi %arg4, %arg6 : i32 loc(#loc58)
    %9 = arith.divui %8, %arg8 : i32 loc(#loc58)
    %10 = arith.addi %9, %2 : i32 loc(#loc58)
    %11 = arith.cmpi eq, %arg2, %3 : i32 loc(#loc63)
    %trueResult, %falseResult = handshake.cond_br %11, %1 : none loc(#loc61)
    %12 = arith.cmpi eq, %arg5, %3 : i32 loc(#loc2)
    %13 = arith.cmpi eq, %arg6, %3 : i32 loc(#loc2)
    %14 = arith.cmpi eq, %7, %3 : i32 loc(#loc2)
    %15 = arith.cmpi eq, %10, %3 : i32 loc(#loc2)
    %16 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc61)
    %17 = arith.index_cast %3 : i32 to index loc(#loc61)
    %18 = arith.index_cast %arg2 : i32 to index loc(#loc61)
    %index, %willContinue = dataflow.stream %17, %16, %18 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc61)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc61)
    %19 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc61)
    %20 = arith.index_cast %afterValue : index to i32 loc(#loc61)
    %21 = dataflow.invariant %afterCond, %12 : i1, i1 -> i1 loc(#loc66)
    %trueResult_0, %falseResult_1 = handshake.cond_br %21, %19 : none loc(#loc66)
    %22 = arith.muli %20, %arg5 : i32 loc(#loc2)
    %23 = arith.muli %20, %arg3 : i32 loc(#loc2)
    %24 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc66)
    %25 = arith.index_cast %arg5 : i32 to index loc(#loc66)
    %index_2, %willContinue_3 = dataflow.stream %17, %24, %25 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "!="} loc(#loc66)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc66)
    %26 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc66)
    %27 = arith.index_cast %afterValue_4 : index to i32 loc(#loc66)
    %28 = dataflow.invariant %afterCond, %13 : i1, i1 -> i1 loc(#loc70)
    %trueResult_6, %falseResult_7 = handshake.cond_br %28, %26 : none loc(#loc70)
    %29 = dataflow.invariant %afterCond_5, %22 : i1, i32 -> i32 loc(#loc2)
    %30 = arith.addi %27, %29 : i32 loc(#loc2)
    %31 = arith.muli %30, %arg6 : i32 loc(#loc2)
    %32 = dataflow.invariant %afterCond_5, %23 : i1, i32 -> i32 loc(#loc2)
    %33 = arith.addi %27, %32 : i32 loc(#loc2)
    %34 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc70)
    %35 = arith.index_cast %arg6 : i32 to index loc(#loc70)
    %index_8, %willContinue_9 = dataflow.stream %17, %34, %35 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "!="} loc(#loc70)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc70)
    %36 = dataflow.invariant %afterCond_11, %falseResult_7 : i1, none -> none loc(#loc70)
    %37 = arith.index_cast %afterValue_10 : index to i32 loc(#loc70)
    %38 = dataflow.invariant %afterCond, %14 : i1, i1 -> i1 loc(#loc76)
    %trueResult_12, %falseResult_13 = handshake.cond_br %38, %36 : none loc(#loc76)
    %39 = dataflow.invariant %afterCond_11, %31 : i1, i32 -> i32 loc(#loc74)
    %40 = arith.addi %37, %39 : i32 loc(#loc74)
    %41 = arith.muli %40, %7 : i32 loc(#loc2)
    %42 = handshake.constant %falseResult_13 {value = 1 : index} : index loc(#loc76)
    %43 = arith.index_cast %7 : i32 to index loc(#loc76)
    %index_14, %willContinue_15 = dataflow.stream %17, %42, %43 {step_op = "+=", stop_cond = "<"} loc(#loc76)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc76)
    %44 = dataflow.invariant %afterCond_17, %falseResult_13 : i1, none -> none loc(#loc76)
    %45 = arith.index_cast %afterValue_16 : index to i32 loc(#loc76)
    %46 = dataflow.invariant %afterCond, %15 : i1, i1 -> i1 loc(#loc79)
    %trueResult_18, %falseResult_19 = handshake.cond_br %46, %44 : none loc(#loc79)
    %47 = arith.muli %45, %arg7 : i32 loc(#loc2)
    %48 = dataflow.invariant %afterCond_11, %33 : i1, i32 -> i32 loc(#loc2)
    %49 = arith.addi %48, %47 : i32 loc(#loc2)
    %50 = arith.muli %49, %arg4 : i32 loc(#loc2)
    %51 = arith.addi %50, %37 : i32 loc(#loc2)
    %52 = dataflow.invariant %afterCond_17, %41 : i1, i32 -> i32 loc(#loc2)
    %53 = arith.addi %45, %52 : i32 loc(#loc2)
    %54 = arith.muli %53, %10 : i32 loc(#loc2)
    %55 = handshake.constant %falseResult_19 {value = 1 : index} : index loc(#loc79)
    %56 = arith.index_cast %4 : i64 to index loc(#loc79)
    %57 = arith.index_cast %10 : i32 to index loc(#loc79)
    %index_20, %willContinue_21 = dataflow.stream %56, %55, %57 {step_op = "+=", stop_cond = "<"} loc(#loc79)
    %afterValue_22, %afterCond_23 = dataflow.gate %index_20, %willContinue_21 : index, i1 -> index, i1 loc(#loc79)
    %58 = arith.index_cast %afterValue_22 : index to i64 loc(#loc79)
    %59 = arith.trunci %58 : i64 to i32 loc(#loc83)
    %60 = arith.muli %arg8, %59 : i32 loc(#loc83)
    %61 = dataflow.invariant %afterCond_23, %51 : i1, i32 -> i32 loc(#loc84)
    %62 = arith.addi %61, %60 : i32 loc(#loc84)
    %63 = arith.extui %62 : i32 to i64 loc(#loc84)
    %64 = arith.index_cast %63 : i64 to index loc(#loc84)
    %dataResult, %addressResults = handshake.load [%64] %69#0, %75 : index, f32 loc(#loc84)
    %65 = dataflow.invariant %afterCond_23, %54 : i1, i32 -> i32 loc(#loc84)
    %66 = arith.addi %65, %59 : i32 loc(#loc84)
    %67 = arith.extui %66 : i32 to i64 loc(#loc84)
    %68 = arith.index_cast %67 : i64 to index loc(#loc84)
    %dataResult_24, %addressResult = handshake.store [%68] %dataResult, %100 : index, f32 loc(#loc84)
    %69:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc56)
    %70 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_24, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc56)
    %71 = dataflow.carry %willContinue, %falseResult, %trueResult_41 : i1, none, none -> none loc(#loc61)
    %trueResult_25, %falseResult_26 = handshake.cond_br %21, %71 : none loc(#loc66)
    %72 = dataflow.carry %willContinue_3, %falseResult_26, %trueResult_39 : i1, none, none -> none loc(#loc66)
    %trueResult_27, %falseResult_28 = handshake.cond_br %28, %72 : none loc(#loc70)
    %73 = dataflow.carry %willContinue_9, %falseResult_28, %trueResult_37 : i1, none, none -> none loc(#loc70)
    %trueResult_29, %falseResult_30 = handshake.cond_br %38, %73 : none loc(#loc76)
    %74 = dataflow.carry %willContinue_15, %falseResult_30, %trueResult_35 : i1, none, none -> none loc(#loc76)
    %trueResult_31, %falseResult_32 = handshake.cond_br %46, %74 : none loc(#loc79)
    %75 = dataflow.carry %willContinue_21, %falseResult_32, %trueResult_33 : i1, none, none -> none loc(#loc79)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_21, %69#1 : none loc(#loc79)
    %76 = handshake.constant %74 {value = 0 : index} : index loc(#loc79)
    %77 = handshake.constant %74 {value = 1 : index} : index loc(#loc79)
    %78 = arith.select %46, %77, %76 : index loc(#loc79)
    %79 = handshake.mux %78 [%falseResult_34, %trueResult_31] : index, none loc(#loc79)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_15, %79 : none loc(#loc76)
    %80 = handshake.constant %73 {value = 0 : index} : index loc(#loc76)
    %81 = handshake.constant %73 {value = 1 : index} : index loc(#loc76)
    %82 = arith.select %38, %81, %80 : index loc(#loc76)
    %83 = handshake.mux %82 [%falseResult_36, %trueResult_29] : index, none loc(#loc76)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_9, %83 : none loc(#loc70)
    %84 = handshake.constant %72 {value = 0 : index} : index loc(#loc70)
    %85 = handshake.constant %72 {value = 1 : index} : index loc(#loc70)
    %86 = arith.select %28, %85, %84 : index loc(#loc70)
    %87 = handshake.mux %86 [%falseResult_38, %trueResult_27] : index, none loc(#loc70)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue_3, %87 : none loc(#loc66)
    %88 = handshake.constant %71 {value = 0 : index} : index loc(#loc66)
    %89 = handshake.constant %71 {value = 1 : index} : index loc(#loc66)
    %90 = arith.select %21, %89, %88 : index loc(#loc66)
    %91 = handshake.mux %90 [%falseResult_40, %trueResult_25] : index, none loc(#loc66)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue, %91 : none loc(#loc61)
    %92 = handshake.constant %1 {value = 0 : index} : index loc(#loc61)
    %93 = handshake.constant %1 {value = 1 : index} : index loc(#loc61)
    %94 = arith.select %11, %93, %92 : index loc(#loc61)
    %95 = handshake.mux %94 [%falseResult_42, %trueResult] : index, none loc(#loc61)
    %96 = dataflow.carry %willContinue, %falseResult, %trueResult_59 : i1, none, none -> none loc(#loc61)
    %trueResult_43, %falseResult_44 = handshake.cond_br %21, %96 : none loc(#loc66)
    %97 = dataflow.carry %willContinue_3, %falseResult_44, %trueResult_57 : i1, none, none -> none loc(#loc66)
    %trueResult_45, %falseResult_46 = handshake.cond_br %28, %97 : none loc(#loc70)
    %98 = dataflow.carry %willContinue_9, %falseResult_46, %trueResult_55 : i1, none, none -> none loc(#loc70)
    %trueResult_47, %falseResult_48 = handshake.cond_br %38, %98 : none loc(#loc76)
    %99 = dataflow.carry %willContinue_15, %falseResult_48, %trueResult_53 : i1, none, none -> none loc(#loc76)
    %trueResult_49, %falseResult_50 = handshake.cond_br %46, %99 : none loc(#loc79)
    %100 = dataflow.carry %willContinue_21, %falseResult_50, %trueResult_51 : i1, none, none -> none loc(#loc79)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue_21, %70 : none loc(#loc79)
    %101 = handshake.constant %99 {value = 0 : index} : index loc(#loc79)
    %102 = handshake.constant %99 {value = 1 : index} : index loc(#loc79)
    %103 = arith.select %46, %102, %101 : index loc(#loc79)
    %104 = handshake.mux %103 [%falseResult_52, %trueResult_49] : index, none loc(#loc79)
    %trueResult_53, %falseResult_54 = handshake.cond_br %willContinue_15, %104 : none loc(#loc76)
    %105 = handshake.constant %98 {value = 0 : index} : index loc(#loc76)
    %106 = handshake.constant %98 {value = 1 : index} : index loc(#loc76)
    %107 = arith.select %38, %106, %105 : index loc(#loc76)
    %108 = handshake.mux %107 [%falseResult_54, %trueResult_47] : index, none loc(#loc76)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue_9, %108 : none loc(#loc70)
    %109 = handshake.constant %97 {value = 0 : index} : index loc(#loc70)
    %110 = handshake.constant %97 {value = 1 : index} : index loc(#loc70)
    %111 = arith.select %28, %110, %109 : index loc(#loc70)
    %112 = handshake.mux %111 [%falseResult_56, %trueResult_45] : index, none loc(#loc70)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue_3, %112 : none loc(#loc66)
    %113 = handshake.constant %96 {value = 0 : index} : index loc(#loc66)
    %114 = handshake.constant %96 {value = 1 : index} : index loc(#loc66)
    %115 = arith.select %21, %114, %113 : index loc(#loc66)
    %116 = handshake.mux %115 [%falseResult_58, %trueResult_43] : index, none loc(#loc66)
    %trueResult_59, %falseResult_60 = handshake.cond_br %willContinue, %116 : none loc(#loc61)
    %117 = handshake.mux %94 [%falseResult_60, %trueResult] : index, none loc(#loc61)
    %118 = handshake.join %95, %117 : none, none loc(#loc56)
    %119 = handshake.constant %118 {value = true} : i1 loc(#loc56)
    handshake.return %119 : i1 loc(#loc56)
  } loc(#loc56)
  handshake.func @_Z10im2col_dsaPKfPfjjjjjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc14]), %arg2: i32 loc(fused<#di_subprogram5>[#loc14]), %arg3: i32 loc(fused<#di_subprogram5>[#loc14]), %arg4: i32 loc(fused<#di_subprogram5>[#loc14]), %arg5: i32 loc(fused<#di_subprogram5>[#loc14]), %arg6: i32 loc(fused<#di_subprogram5>[#loc14]), %arg7: i32 loc(fused<#di_subprogram5>[#loc14]), %arg8: i32 loc(fused<#di_subprogram5>[#loc14]), %arg9: none loc(fused<#di_subprogram5>[#loc14]), ...) -> none attributes {argNames = ["input", "output", "C", "H", "W", "KH", "KW", "stride_h", "stride_w", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg9 : none loc(#loc56)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %4 = arith.subi %arg3, %arg5 : i32 loc(#loc57)
    %5 = arith.divui %4, %arg7 : i32 loc(#loc57)
    %6 = arith.addi %5, %1 : i32 loc(#loc57)
    %7 = arith.subi %arg4, %arg6 : i32 loc(#loc58)
    %8 = arith.divui %7, %arg8 : i32 loc(#loc58)
    %9 = arith.addi %8, %1 : i32 loc(#loc58)
    %10 = arith.cmpi eq, %arg2, %2 : i32 loc(#loc63)
    %trueResult, %falseResult = handshake.cond_br %10, %0 : none loc(#loc61)
    %11 = arith.cmpi eq, %arg5, %2 : i32 loc(#loc2)
    %12 = arith.cmpi eq, %arg6, %2 : i32 loc(#loc2)
    %13 = arith.cmpi eq, %6, %2 : i32 loc(#loc2)
    %14 = arith.cmpi eq, %9, %2 : i32 loc(#loc2)
    %15 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc61)
    %16 = arith.index_cast %2 : i32 to index loc(#loc61)
    %17 = arith.index_cast %arg2 : i32 to index loc(#loc61)
    %index, %willContinue = dataflow.stream %16, %15, %17 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc61)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc61)
    %18 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc61)
    %19 = arith.index_cast %afterValue : index to i32 loc(#loc61)
    %20 = dataflow.invariant %afterCond, %11 : i1, i1 -> i1 loc(#loc66)
    %trueResult_0, %falseResult_1 = handshake.cond_br %20, %18 : none loc(#loc66)
    %21 = arith.muli %19, %arg5 : i32 loc(#loc2)
    %22 = arith.muli %19, %arg3 : i32 loc(#loc2)
    %23 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc66)
    %24 = arith.index_cast %arg5 : i32 to index loc(#loc66)
    %index_2, %willContinue_3 = dataflow.stream %16, %23, %24 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "!="} loc(#loc66)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc66)
    %25 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc66)
    %26 = arith.index_cast %afterValue_4 : index to i32 loc(#loc66)
    %27 = dataflow.invariant %afterCond, %12 : i1, i1 -> i1 loc(#loc70)
    %trueResult_6, %falseResult_7 = handshake.cond_br %27, %25 : none loc(#loc70)
    %28 = dataflow.invariant %afterCond_5, %21 : i1, i32 -> i32 loc(#loc2)
    %29 = arith.addi %26, %28 : i32 loc(#loc2)
    %30 = arith.muli %29, %arg6 : i32 loc(#loc2)
    %31 = dataflow.invariant %afterCond_5, %22 : i1, i32 -> i32 loc(#loc2)
    %32 = arith.addi %26, %31 : i32 loc(#loc2)
    %33 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc70)
    %34 = arith.index_cast %arg6 : i32 to index loc(#loc70)
    %index_8, %willContinue_9 = dataflow.stream %16, %33, %34 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "!="} loc(#loc70)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc70)
    %35 = dataflow.invariant %afterCond_11, %falseResult_7 : i1, none -> none loc(#loc70)
    %36 = arith.index_cast %afterValue_10 : index to i32 loc(#loc70)
    %37 = dataflow.invariant %afterCond, %13 : i1, i1 -> i1 loc(#loc76)
    %trueResult_12, %falseResult_13 = handshake.cond_br %37, %35 : none loc(#loc76)
    %38 = dataflow.invariant %afterCond_11, %30 : i1, i32 -> i32 loc(#loc74)
    %39 = arith.addi %36, %38 : i32 loc(#loc74)
    %40 = arith.muli %39, %6 : i32 loc(#loc2)
    %41 = handshake.constant %falseResult_13 {value = 1 : index} : index loc(#loc76)
    %42 = arith.index_cast %6 : i32 to index loc(#loc76)
    %index_14, %willContinue_15 = dataflow.stream %16, %41, %42 {step_op = "+=", stop_cond = "<"} loc(#loc76)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc76)
    %43 = dataflow.invariant %afterCond_17, %falseResult_13 : i1, none -> none loc(#loc76)
    %44 = arith.index_cast %afterValue_16 : index to i32 loc(#loc76)
    %45 = dataflow.invariant %afterCond, %14 : i1, i1 -> i1 loc(#loc79)
    %trueResult_18, %falseResult_19 = handshake.cond_br %45, %43 : none loc(#loc79)
    %46 = arith.muli %44, %arg7 : i32 loc(#loc2)
    %47 = dataflow.invariant %afterCond_11, %32 : i1, i32 -> i32 loc(#loc2)
    %48 = arith.addi %47, %46 : i32 loc(#loc2)
    %49 = arith.muli %48, %arg4 : i32 loc(#loc2)
    %50 = arith.addi %49, %36 : i32 loc(#loc2)
    %51 = dataflow.invariant %afterCond_17, %40 : i1, i32 -> i32 loc(#loc2)
    %52 = arith.addi %44, %51 : i32 loc(#loc2)
    %53 = arith.muli %52, %9 : i32 loc(#loc2)
    %54 = handshake.constant %falseResult_19 {value = 1 : index} : index loc(#loc79)
    %55 = arith.index_cast %3 : i64 to index loc(#loc79)
    %56 = arith.index_cast %9 : i32 to index loc(#loc79)
    %index_20, %willContinue_21 = dataflow.stream %55, %54, %56 {step_op = "+=", stop_cond = "<"} loc(#loc79)
    %afterValue_22, %afterCond_23 = dataflow.gate %index_20, %willContinue_21 : index, i1 -> index, i1 loc(#loc79)
    %57 = arith.index_cast %afterValue_22 : index to i64 loc(#loc79)
    %58 = arith.trunci %57 : i64 to i32 loc(#loc83)
    %59 = arith.muli %arg8, %58 : i32 loc(#loc83)
    %60 = dataflow.invariant %afterCond_23, %50 : i1, i32 -> i32 loc(#loc84)
    %61 = arith.addi %60, %59 : i32 loc(#loc84)
    %62 = arith.extui %61 : i32 to i64 loc(#loc84)
    %63 = arith.index_cast %62 : i64 to index loc(#loc84)
    %dataResult, %addressResults = handshake.load [%63] %68#0, %74 : index, f32 loc(#loc84)
    %64 = dataflow.invariant %afterCond_23, %53 : i1, i32 -> i32 loc(#loc84)
    %65 = arith.addi %64, %58 : i32 loc(#loc84)
    %66 = arith.extui %65 : i32 to i64 loc(#loc84)
    %67 = arith.index_cast %66 : i64 to index loc(#loc84)
    %dataResult_24, %addressResult = handshake.store [%67] %dataResult, %99 : index, f32 loc(#loc84)
    %68:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc56)
    %69 = handshake.extmemory[ld = 0, st = 1] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_24, %addressResult) {id = 1 : i32} : (f32, index) -> none loc(#loc56)
    %70 = dataflow.carry %willContinue, %falseResult, %trueResult_41 : i1, none, none -> none loc(#loc61)
    %trueResult_25, %falseResult_26 = handshake.cond_br %20, %70 : none loc(#loc66)
    %71 = dataflow.carry %willContinue_3, %falseResult_26, %trueResult_39 : i1, none, none -> none loc(#loc66)
    %trueResult_27, %falseResult_28 = handshake.cond_br %27, %71 : none loc(#loc70)
    %72 = dataflow.carry %willContinue_9, %falseResult_28, %trueResult_37 : i1, none, none -> none loc(#loc70)
    %trueResult_29, %falseResult_30 = handshake.cond_br %37, %72 : none loc(#loc76)
    %73 = dataflow.carry %willContinue_15, %falseResult_30, %trueResult_35 : i1, none, none -> none loc(#loc76)
    %trueResult_31, %falseResult_32 = handshake.cond_br %45, %73 : none loc(#loc79)
    %74 = dataflow.carry %willContinue_21, %falseResult_32, %trueResult_33 : i1, none, none -> none loc(#loc79)
    %trueResult_33, %falseResult_34 = handshake.cond_br %willContinue_21, %68#1 : none loc(#loc79)
    %75 = handshake.constant %73 {value = 0 : index} : index loc(#loc79)
    %76 = handshake.constant %73 {value = 1 : index} : index loc(#loc79)
    %77 = arith.select %45, %76, %75 : index loc(#loc79)
    %78 = handshake.mux %77 [%falseResult_34, %trueResult_31] : index, none loc(#loc79)
    %trueResult_35, %falseResult_36 = handshake.cond_br %willContinue_15, %78 : none loc(#loc76)
    %79 = handshake.constant %72 {value = 0 : index} : index loc(#loc76)
    %80 = handshake.constant %72 {value = 1 : index} : index loc(#loc76)
    %81 = arith.select %37, %80, %79 : index loc(#loc76)
    %82 = handshake.mux %81 [%falseResult_36, %trueResult_29] : index, none loc(#loc76)
    %trueResult_37, %falseResult_38 = handshake.cond_br %willContinue_9, %82 : none loc(#loc70)
    %83 = handshake.constant %71 {value = 0 : index} : index loc(#loc70)
    %84 = handshake.constant %71 {value = 1 : index} : index loc(#loc70)
    %85 = arith.select %27, %84, %83 : index loc(#loc70)
    %86 = handshake.mux %85 [%falseResult_38, %trueResult_27] : index, none loc(#loc70)
    %trueResult_39, %falseResult_40 = handshake.cond_br %willContinue_3, %86 : none loc(#loc66)
    %87 = handshake.constant %70 {value = 0 : index} : index loc(#loc66)
    %88 = handshake.constant %70 {value = 1 : index} : index loc(#loc66)
    %89 = arith.select %20, %88, %87 : index loc(#loc66)
    %90 = handshake.mux %89 [%falseResult_40, %trueResult_25] : index, none loc(#loc66)
    %trueResult_41, %falseResult_42 = handshake.cond_br %willContinue, %90 : none loc(#loc61)
    %91 = handshake.constant %0 {value = 0 : index} : index loc(#loc61)
    %92 = handshake.constant %0 {value = 1 : index} : index loc(#loc61)
    %93 = arith.select %10, %92, %91 : index loc(#loc61)
    %94 = handshake.mux %93 [%falseResult_42, %trueResult] : index, none loc(#loc61)
    %95 = dataflow.carry %willContinue, %falseResult, %trueResult_59 : i1, none, none -> none loc(#loc61)
    %trueResult_43, %falseResult_44 = handshake.cond_br %20, %95 : none loc(#loc66)
    %96 = dataflow.carry %willContinue_3, %falseResult_44, %trueResult_57 : i1, none, none -> none loc(#loc66)
    %trueResult_45, %falseResult_46 = handshake.cond_br %27, %96 : none loc(#loc70)
    %97 = dataflow.carry %willContinue_9, %falseResult_46, %trueResult_55 : i1, none, none -> none loc(#loc70)
    %trueResult_47, %falseResult_48 = handshake.cond_br %37, %97 : none loc(#loc76)
    %98 = dataflow.carry %willContinue_15, %falseResult_48, %trueResult_53 : i1, none, none -> none loc(#loc76)
    %trueResult_49, %falseResult_50 = handshake.cond_br %45, %98 : none loc(#loc79)
    %99 = dataflow.carry %willContinue_21, %falseResult_50, %trueResult_51 : i1, none, none -> none loc(#loc79)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue_21, %69 : none loc(#loc79)
    %100 = handshake.constant %98 {value = 0 : index} : index loc(#loc79)
    %101 = handshake.constant %98 {value = 1 : index} : index loc(#loc79)
    %102 = arith.select %45, %101, %100 : index loc(#loc79)
    %103 = handshake.mux %102 [%falseResult_52, %trueResult_49] : index, none loc(#loc79)
    %trueResult_53, %falseResult_54 = handshake.cond_br %willContinue_15, %103 : none loc(#loc76)
    %104 = handshake.constant %97 {value = 0 : index} : index loc(#loc76)
    %105 = handshake.constant %97 {value = 1 : index} : index loc(#loc76)
    %106 = arith.select %37, %105, %104 : index loc(#loc76)
    %107 = handshake.mux %106 [%falseResult_54, %trueResult_47] : index, none loc(#loc76)
    %trueResult_55, %falseResult_56 = handshake.cond_br %willContinue_9, %107 : none loc(#loc70)
    %108 = handshake.constant %96 {value = 0 : index} : index loc(#loc70)
    %109 = handshake.constant %96 {value = 1 : index} : index loc(#loc70)
    %110 = arith.select %27, %109, %108 : index loc(#loc70)
    %111 = handshake.mux %110 [%falseResult_56, %trueResult_45] : index, none loc(#loc70)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue_3, %111 : none loc(#loc66)
    %112 = handshake.constant %95 {value = 0 : index} : index loc(#loc66)
    %113 = handshake.constant %95 {value = 1 : index} : index loc(#loc66)
    %114 = arith.select %20, %113, %112 : index loc(#loc66)
    %115 = handshake.mux %114 [%falseResult_58, %trueResult_43] : index, none loc(#loc66)
    %trueResult_59, %falseResult_60 = handshake.cond_br %willContinue, %115 : none loc(#loc61)
    %116 = handshake.mux %93 [%falseResult_60, %trueResult] : index, none loc(#loc61)
    %117 = handshake.join %94, %116 : none, none loc(#loc56)
    handshake.return %117 : none loc(#loc59)
  } loc(#loc56)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc37)
    %false = arith.constant false loc(#loc37)
    %0 = seq.const_clock  low loc(#loc37)
    %c2_i32 = arith.constant 2 : i32 loc(#loc37)
    %1 = ub.poison : i64 loc(#loc37)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c972_i64 = arith.constant 972 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c192_i64 = arith.constant 192 : i64 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %cst = arith.constant 9.99999997E-7 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<15xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<15xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<192xf32> loc(#loc2)
    %alloca_0 = memref.alloca() : memref<972xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<972xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %10 = arith.trunci %arg0 : i64 to i32 loc(#loc46)
      %11 = arith.uitofp %10 : i32 to f32 loc(#loc46)
      %12 = arith.index_cast %arg0 : i64 to index loc(#loc46)
      memref.store %11, %alloca[%12] : memref<192xf32> loc(#loc46)
      %13 = arith.addi %arg0, %c1_i64 : i64 loc(#loc44)
      %14 = arith.cmpi ne, %13, %c192_i64 : i64 loc(#loc47)
      scf.condition(%14) %13 : i64 loc(#loc42)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block14>[#loc27])):
      scf.yield %arg0 : i64 loc(#loc42)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc42)
    %cast = memref.cast %alloca : memref<192xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    %cast_2 = memref.cast %alloca_0 : memref<972xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc38)
    call @_Z10im2col_cpuPKfPfjjjjjjj(%cast, %cast_2, %c3_i32, %c8_i32, %c8_i32, %c3_i32, %c3_i32, %c1_i32, %c1_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32, i32, i32, i32, i32, i32) -> () loc(#loc38)
    %cast_3 = memref.cast %alloca_1 : memref<972xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_4, %ready_5 = esi.wrap.vr %cast_3, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc39)
    %chanOutput_6, %ready_7 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc39)
    %chanOutput_8, %ready_9 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc39)
    %chanOutput_10, %ready_11 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc39)
    %chanOutput_12, %ready_13 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc39)
    %chanOutput_14, %ready_15 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc39)
    %chanOutput_16, %ready_17 = esi.wrap.vr %c1_i32, %true : i32 loc(#loc39)
    %chanOutput_18, %ready_19 = esi.wrap.vr %c1_i32, %true : i32 loc(#loc39)
    %chanOutput_20, %ready_21 = esi.wrap.vr %true, %true : i1 loc(#loc39)
    %5 = handshake.esi_instance @_Z10im2col_dsaPKfPfjjjjjjj_esi "_Z10im2col_dsaPKfPfjjjjjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_4, %chanOutput_6, %chanOutput_8, %chanOutput_10, %chanOutput_12, %chanOutput_14, %chanOutput_16, %chanOutput_18, %chanOutput_20) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc39)
    %rawOutput, %valid = esi.unwrap.vr %5, %true : i1 loc(#loc39)
    %6:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %10 = arith.index_cast %arg0 : i64 to index loc(#loc49)
      %11 = memref.load %alloca_0[%10] : memref<972xf32> loc(#loc49)
      %12 = memref.load %alloca_1[%10] : memref<972xf32> loc(#loc49)
      %13 = arith.subf %11, %12 : f32 loc(#loc49)
      %14 = math.absf %13 : f32 loc(#loc49)
      %15 = arith.cmpf ule, %14, %cst : f32 loc(#loc49)
      %16:3 = scf.if %15 -> (i64, i32, i32) {
        %18 = arith.addi %arg0, %c1_i64 : i64 loc(#loc45)
        %19 = arith.cmpi eq, %18, %c972_i64 : i64 loc(#loc45)
        %20 = arith.extui %19 : i1 to i32 loc(#loc43)
        %21 = arith.cmpi ne, %18, %c972_i64 : i64 loc(#loc48)
        %22 = arith.extui %21 : i1 to i32 loc(#loc43)
        scf.yield %18, %20, %22 : i64, i32, i32 loc(#loc49)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc49)
      } loc(#loc49)
      %17 = arith.trunci %16#2 : i32 to i1 loc(#loc43)
      scf.condition(%17) %16#0, %15, %16#1 : i64, i1, i32 loc(#loc43)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc31]), %arg1: i1 loc(fused<#di_lexical_block15>[#loc31]), %arg2: i32 loc(fused<#di_lexical_block15>[#loc31])):
      scf.yield %arg0 : i64 loc(#loc43)
    } loc(#loc43)
    %7 = arith.index_castui %6#2 : i32 to index loc(#loc43)
    %8 = scf.index_switch %7 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc43)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<15xi8> -> index loc(#loc50)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc50)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc50)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc50)
      scf.yield %c1_i32 : i32 loc(#loc51)
    } loc(#loc43)
    %9 = arith.select %6#1, %c0_i32, %8 : i32 loc(#loc2)
    scf.if %6#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<15xi8> -> index loc(#loc40)
      %10 = arith.index_cast %intptr : index to i64 loc(#loc40)
      %11 = llvm.inttoptr %10 : i64 to !llvm.ptr loc(#loc40)
      %12 = llvm.call @puts(%11) : (!llvm.ptr) -> i32 loc(#loc40)
    } loc(#loc2)
    return %9 : i32 loc(#loc41)
  } loc(#loc37)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/im2col/im2col.cpp":0:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/im2col/im2col.cpp":26:0)
#loc4 = loc("tests/app/im2col/im2col.cpp":27:0)
#loc9 = loc("tests/app/im2col/im2col.cpp":32:0)
#loc11 = loc("tests/app/im2col/im2col.cpp":37:0)
#loc12 = loc("tests/app/im2col/im2col.cpp":40:0)
#loc13 = loc("tests/app/im2col/im2col.cpp":46:0)
#loc15 = loc("tests/app/im2col/im2col.cpp":59:0)
#loc16 = loc("tests/app/im2col/im2col.cpp":60:0)
#loc17 = loc("tests/app/im2col/im2col.cpp":63:0)
#loc18 = loc("tests/app/im2col/im2col.cpp":65:0)
#loc19 = loc("tests/app/im2col/im2col.cpp":67:0)
#loc20 = loc("tests/app/im2col/im2col.cpp":70:0)
#loc21 = loc("tests/app/im2col/im2col.cpp":68:0)
#loc22 = loc("tests/app/im2col/im2col.cpp":71:0)
#loc23 = loc("tests/app/im2col/im2col.cpp":73:0)
#loc24 = loc("tests/app/im2col/im2col.cpp":76:0)
#loc25 = loc("tests/app/im2col/im2col.cpp":82:0)
#loc26 = loc("tests/app/im2col/main.cpp":6:0)
#loc28 = loc("tests/app/im2col/main.cpp":28:0)
#loc29 = loc("tests/app/im2col/main.cpp":32:0)
#loc30 = loc("tests/app/im2col/main.cpp":35:0)
#loc32 = loc("tests/app/im2col/main.cpp":39:0)
#loc33 = loc("tests/app/im2col/main.cpp":40:0)
#loc34 = loc("tests/app/im2col/main.cpp":41:0)
#loc35 = loc("tests/app/im2col/main.cpp":45:0)
#loc36 = loc("tests/app/im2col/main.cpp":47:0)
#loc37 = loc(fused<#di_subprogram3>[#loc26])
#loc38 = loc(fused<#di_subprogram3>[#loc29])
#loc39 = loc(fused<#di_subprogram3>[#loc30])
#loc40 = loc(fused<#di_subprogram3>[#loc35])
#loc41 = loc(fused<#di_subprogram3>[#loc36])
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file1, line = 27>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 38>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file1, line = 27>
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file1, line = 38>
#loc44 = loc(fused<#di_lexical_block18>[#loc27])
#loc45 = loc(fused<#di_lexical_block19>[#loc31])
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file1, line = 39>
#loc46 = loc(fused<#di_lexical_block22>[#loc28])
#loc47 = loc(fused[#loc42, #loc44])
#loc48 = loc(fused[#loc43, #loc45])
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block26, file = #di_file1, line = 39>
#loc49 = loc(fused<#di_lexical_block26>[#loc32])
#loc50 = loc(fused<#di_lexical_block29>[#loc33])
#loc51 = loc(fused<#di_lexical_block29>[#loc34])
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 63>
#loc53 = loc(fused<#di_subprogram4>[#loc3])
#loc54 = loc(fused<#di_subprogram4>[#loc4])
#loc55 = loc(fused<#di_subprogram4>[#loc13])
#loc57 = loc(fused<#di_subprogram5>[#loc15])
#loc58 = loc(fused<#di_subprogram5>[#loc16])
#loc59 = loc(fused<#di_subprogram5>[#loc25])
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file, line = 63>
#loc61 = loc(fused<#di_lexical_block41>[#loc17])
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_lexical_block43, file = #di_file, line = 63>
#loc62 = loc(fused<#di_lexical_block42>[#loc5])
#loc63 = loc(fused<#di_lexical_block43>[#loc17])
#di_lexical_block47 = #llvm.di_lexical_block<scope = #di_lexical_block45, file = #di_file, line = 65>
#loc64 = loc(fused[#loc60, #loc62])
#di_lexical_block49 = #llvm.di_lexical_block<scope = #di_lexical_block47, file = #di_file, line = 65>
#loc66 = loc(fused<#di_lexical_block47>[#loc18])
#di_lexical_block51 = #llvm.di_lexical_block<scope = #di_lexical_block49, file = #di_file, line = 65>
#loc67 = loc(fused<#di_lexical_block48>[#loc6])
#di_lexical_block53 = #llvm.di_lexical_block<scope = #di_lexical_block51, file = #di_file, line = 67>
#loc68 = loc(fused[#loc65, #loc67])
#di_lexical_block55 = #llvm.di_lexical_block<scope = #di_lexical_block53, file = #di_file, line = 67>
#loc70 = loc(fused<#di_lexical_block53>[#loc19])
#di_lexical_block57 = #llvm.di_lexical_block<scope = #di_lexical_block55, file = #di_file, line = 67>
#loc71 = loc(fused<#di_lexical_block54>[#loc7])
#di_lexical_block59 = #llvm.di_lexical_block<scope = #di_lexical_block57, file = #di_file, line = 70>
#loc72 = loc(fused<#di_lexical_block56>[#loc9])
#loc73 = loc(fused[#loc69, #loc71])
#loc74 = loc(fused<#di_lexical_block57>[#loc21])
#di_lexical_block61 = #llvm.di_lexical_block<scope = #di_lexical_block59, file = #di_file, line = 70>
#loc76 = loc(fused<#di_lexical_block59>[#loc20])
#di_lexical_block63 = #llvm.di_lexical_block<scope = #di_lexical_block61, file = #di_file, line = 70>
#loc77 = loc(fused<#di_lexical_block60>[#loc8])
#di_lexical_block65 = #llvm.di_lexical_block<scope = #di_lexical_block63, file = #di_file, line = 71>
#di_lexical_block66 = #llvm.di_lexical_block<scope = #di_lexical_block64, file = #di_file, line = 35>
#di_lexical_block67 = #llvm.di_lexical_block<scope = #di_lexical_block65, file = #di_file, line = 71>
#loc79 = loc(fused<#di_lexical_block65>[#loc22])
#di_lexical_block68 = #llvm.di_lexical_block<scope = #di_lexical_block66, file = #di_file, line = 35>
#di_lexical_block69 = #llvm.di_lexical_block<scope = #di_lexical_block67, file = #di_file, line = 71>
#loc80 = loc(fused<#di_lexical_block66>[#loc10])
#loc81 = loc(fused<#di_lexical_block68>[#loc11])
#loc82 = loc(fused<#di_lexical_block68>[#loc12])
#loc83 = loc(fused<#di_lexical_block69>[#loc23])
#loc84 = loc(fused<#di_lexical_block69>[#loc24])
