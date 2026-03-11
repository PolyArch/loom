#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "unsigned int", sizeInBits = 32, encoding = DW_ATE_unsigned>
#di_basic_type2 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"tests/app/depthwise_conv/depthwise_conv.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_file1 = #llvm.di_file<"tests/app/depthwise_conv/main.cpp" in "/home/sihao/github.com/PolyArch/loom">
#di_null_type = #llvm.di_null_type
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[1]<>, isRecSelf = true>
#di_subprogram2 = #llvm.di_subprogram<recId = distinct[2]<>, isRecSelf = true>
#loc1 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":16:0)
#loc5 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":29:0)
#loc6 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":30:0)
#loc7 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":31:0)
#loc8 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":34:0)
#loc9 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":35:0)
#loc15 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":53:0)
#loc29 = loc("tests/app/depthwise_conv/main.cpp":28:0)
#loc31 = loc("tests/app/depthwise_conv/main.cpp":33:0)
#loc35 = loc("tests/app/depthwise_conv/main.cpp":44:0)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file1, producer = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 8192, elements = #llvm.di_subrange<count = 256 : i64>>
#di_composite_type1 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 1152, elements = #llvm.di_subrange<count = 36 : i64>>
#di_composite_type2 = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_basic_type, sizeInBits = 4608, elements = #llvm.di_subrange<count = 144 : i64>>
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_derived_type1 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_basic_type, sizeInBits = 64>
#di_derived_type2 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "__uint32_t", baseType = #di_basic_type1>
#di_lexical_block = #llvm.di_lexical_block<scope = #di_subprogram, file = #di_file, line = 29>
#di_lexical_block1 = #llvm.di_lexical_block<scope = #di_subprogram1, file = #di_file, line = 67>
#di_lexical_block2 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 28>
#di_lexical_block3 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 33>
#di_lexical_block4 = #llvm.di_lexical_block<scope = #di_subprogram2, file = #di_file1, line = 44>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type2>
#di_derived_type3 = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #di_derived_type, sizeInBits = 64>
#di_derived_type4 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type1>
#di_derived_type5 = #llvm.di_derived_type<tag = DW_TAG_typedef, name = "uint32_t", baseType = #di_derived_type2>
#di_lexical_block5 = #llvm.di_lexical_block<scope = #di_lexical_block, file = #di_file, line = 29>
#di_lexical_block6 = #llvm.di_lexical_block<scope = #di_lexical_block1, file = #di_file, line = 67>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram2, name = "input", file = #di_file1, line = 18, type = #di_composite_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram2, name = "kernel", file = #di_file1, line = 21, type = #di_composite_type1>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram2, name = "expect_output", file = #di_file1, line = 24, type = #di_composite_type2>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram2, name = "calculated_output", file = #di_file1, line = 25, type = #di_composite_type2>
#di_derived_type6 = #llvm.di_derived_type<tag = DW_TAG_restrict_type, baseType = #di_derived_type3>
#di_derived_type7 = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_derived_type5>
#di_lexical_block7 = #llvm.di_lexical_block<scope = #di_lexical_block5, file = #di_file, line = 29>
#di_lexical_block8 = #llvm.di_lexical_block<scope = #di_lexical_block6, file = #di_file, line = 67>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, line = 18, arg = 3, type = #di_derived_type4>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "OH", file = #di_file, line = 26, type = #di_derived_type5>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "OW", file = #di_file, line = 27, type = #di_derived_type5>
#di_local_variable7 = #llvm.di_local_variable<scope = #di_lexical_block, name = "c", file = #di_file, line = 29, type = #di_derived_type5>
#di_local_variable8 = #llvm.di_local_variable<scope = #di_subprogram1, name = "output", file = #di_file, line = 55, arg = 3, type = #di_derived_type4>
#di_local_variable9 = #llvm.di_local_variable<scope = #di_subprogram1, name = "OH", file = #di_file, line = 63, type = #di_derived_type5>
#di_local_variable10 = #llvm.di_local_variable<scope = #di_subprogram1, name = "OW", file = #di_file, line = 64, type = #di_derived_type5>
#di_local_variable11 = #llvm.di_local_variable<scope = #di_lexical_block1, name = "c", file = #di_file, line = 67, type = #di_derived_type5>
#di_local_variable12 = #llvm.di_local_variable<scope = #di_lexical_block2, name = "i", file = #di_file1, line = 28, type = #di_derived_type5>
#di_local_variable13 = #llvm.di_local_variable<scope = #di_lexical_block3, name = "i", file = #di_file1, line = 33, type = #di_derived_type5>
#di_local_variable14 = #llvm.di_local_variable<scope = #di_lexical_block4, name = "i", file = #di_file1, line = 44, type = #di_derived_type5>
#di_lexical_block9 = #llvm.di_lexical_block<scope = #di_lexical_block7, file = #di_file, line = 30>
#di_lexical_block10 = #llvm.di_lexical_block<scope = #di_lexical_block8, file = #di_file, line = 69>
#di_local_variable15 = #llvm.di_local_variable<scope = #di_subprogram, name = "input", file = #di_file, line = 16, arg = 1, type = #di_derived_type6>
#di_local_variable16 = #llvm.di_local_variable<scope = #di_subprogram, name = "kernel", file = #di_file, line = 17, arg = 2, type = #di_derived_type6>
#di_local_variable17 = #llvm.di_local_variable<scope = #di_subprogram, name = "C", file = #di_file, line = 19, arg = 4, type = #di_derived_type7>
#di_local_variable18 = #llvm.di_local_variable<scope = #di_subprogram, name = "H", file = #di_file, line = 20, arg = 5, type = #di_derived_type7>
#di_local_variable19 = #llvm.di_local_variable<scope = #di_subprogram, name = "W", file = #di_file, line = 21, arg = 6, type = #di_derived_type7>
#di_local_variable20 = #llvm.di_local_variable<scope = #di_subprogram, name = "KH", file = #di_file, line = 22, arg = 7, type = #di_derived_type7>
#di_local_variable21 = #llvm.di_local_variable<scope = #di_subprogram, name = "KW", file = #di_file, line = 23, arg = 8, type = #di_derived_type7>
#di_local_variable22 = #llvm.di_local_variable<scope = #di_subprogram, name = "stride_h", file = #di_file, line = 24, arg = 9, type = #di_derived_type7>
#di_local_variable23 = #llvm.di_local_variable<scope = #di_subprogram, name = "stride_w", file = #di_file, line = 25, arg = 10, type = #di_derived_type7>
#di_local_variable24 = #llvm.di_local_variable<scope = #di_subprogram1, name = "input", file = #di_file, line = 53, arg = 1, type = #di_derived_type6>
#di_local_variable25 = #llvm.di_local_variable<scope = #di_subprogram1, name = "kernel", file = #di_file, line = 54, arg = 2, type = #di_derived_type6>
#di_local_variable26 = #llvm.di_local_variable<scope = #di_subprogram1, name = "C", file = #di_file, line = 56, arg = 4, type = #di_derived_type7>
#di_local_variable27 = #llvm.di_local_variable<scope = #di_subprogram1, name = "H", file = #di_file, line = 57, arg = 5, type = #di_derived_type7>
#di_local_variable28 = #llvm.di_local_variable<scope = #di_subprogram1, name = "W", file = #di_file, line = 58, arg = 6, type = #di_derived_type7>
#di_local_variable29 = #llvm.di_local_variable<scope = #di_subprogram1, name = "KH", file = #di_file, line = 59, arg = 7, type = #di_derived_type7>
#di_local_variable30 = #llvm.di_local_variable<scope = #di_subprogram1, name = "KW", file = #di_file, line = 60, arg = 8, type = #di_derived_type7>
#di_local_variable31 = #llvm.di_local_variable<scope = #di_subprogram1, name = "stride_h", file = #di_file, line = 61, arg = 9, type = #di_derived_type7>
#di_local_variable32 = #llvm.di_local_variable<scope = #di_subprogram1, name = "stride_w", file = #di_file, line = 62, arg = 10, type = #di_derived_type7>
#di_local_variable33 = #llvm.di_local_variable<scope = #di_subprogram2, name = "C", file = #di_file1, line = 7, type = #di_derived_type7>
#di_local_variable34 = #llvm.di_local_variable<scope = #di_subprogram2, name = "H", file = #di_file1, line = 8, type = #di_derived_type7>
#di_local_variable35 = #llvm.di_local_variable<scope = #di_subprogram2, name = "W", file = #di_file1, line = 9, type = #di_derived_type7>
#di_local_variable36 = #llvm.di_local_variable<scope = #di_subprogram2, name = "KH", file = #di_file1, line = 10, type = #di_derived_type7>
#di_local_variable37 = #llvm.di_local_variable<scope = #di_subprogram2, name = "KW", file = #di_file1, line = 11, type = #di_derived_type7>
#di_local_variable38 = #llvm.di_local_variable<scope = #di_subprogram2, name = "stride_h", file = #di_file1, line = 12, type = #di_derived_type7>
#di_local_variable39 = #llvm.di_local_variable<scope = #di_subprogram2, name = "stride_w", file = #di_file1, line = 13, type = #di_derived_type7>
#di_local_variable40 = #llvm.di_local_variable<scope = #di_subprogram2, name = "OH", file = #di_file1, line = 14, type = #di_derived_type7>
#di_local_variable41 = #llvm.di_local_variable<scope = #di_subprogram2, name = "OW", file = #di_file1, line = 15, type = #di_derived_type7>
#di_subroutine_type1 = #llvm.di_subroutine_type<types = #di_null_type, #di_derived_type6, #di_derived_type6, #di_derived_type4, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7, #di_derived_type7>
#di_lexical_block11 = #llvm.di_lexical_block<scope = #di_lexical_block9, file = #di_file, line = 30>
#di_lexical_block12 = #llvm.di_lexical_block<scope = #di_lexical_block10, file = #di_file, line = 69>
#di_local_variable42 = #llvm.di_local_variable<scope = #di_lexical_block9, name = "oh", file = #di_file, line = 30, type = #di_derived_type5>
#di_local_variable43 = #llvm.di_local_variable<scope = #di_lexical_block10, name = "oh", file = #di_file, line = 69, type = #di_derived_type5>
#di_subprogram3 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[5]<>, compileUnit = #di_compile_unit1, scope = #di_file1, name = "main", file = #di_file1, line = 6, scopeLine = 6, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable33, #di_local_variable34, #di_local_variable35, #di_local_variable36, #di_local_variable37, #di_local_variable38, #di_local_variable39, #di_local_variable40, #di_local_variable41, #di_local_variable, #di_local_variable1, #di_local_variable2, #di_local_variable3, #di_local_variable12, #di_local_variable13, #di_local_variable14>
#di_lexical_block13 = #llvm.di_lexical_block<scope = #di_lexical_block11, file = #di_file, line = 30>
#di_lexical_block14 = #llvm.di_lexical_block<scope = #di_lexical_block12, file = #di_file, line = 69>
#di_lexical_block15 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 28>
#di_lexical_block16 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 33>
#di_lexical_block17 = #llvm.di_lexical_block<scope = #di_subprogram3, file = #di_file1, line = 44>
#di_lexical_block18 = #llvm.di_lexical_block<scope = #di_lexical_block13, file = #di_file, line = 31>
#di_lexical_block19 = #llvm.di_lexical_block<scope = #di_lexical_block14, file = #di_file, line = 71>
#loc46 = loc(fused<#di_lexical_block15>[#loc29])
#loc47 = loc(fused<#di_lexical_block16>[#loc31])
#loc48 = loc(fused<#di_lexical_block17>[#loc35])
#di_lexical_block23 = #llvm.di_lexical_block<scope = #di_lexical_block18, file = #di_file, line = 31>
#di_lexical_block24 = #llvm.di_lexical_block<scope = #di_lexical_block19, file = #di_file, line = 71>
#di_local_variable44 = #llvm.di_local_variable<scope = #di_lexical_block18, name = "ow", file = #di_file, line = 31, type = #di_derived_type5>
#di_local_variable45 = #llvm.di_local_variable<scope = #di_lexical_block19, name = "ow", file = #di_file, line = 71, type = #di_derived_type5>
#di_lexical_block28 = #llvm.di_lexical_block<scope = #di_lexical_block23, file = #di_file, line = 31>
#di_lexical_block29 = #llvm.di_lexical_block<scope = #di_lexical_block24, file = #di_file, line = 71>
#di_lexical_block31 = #llvm.di_lexical_block<scope = #di_lexical_block28, file = #di_file, line = 34>
#di_lexical_block32 = #llvm.di_lexical_block<scope = #di_lexical_block29, file = #di_file, line = 74>
#di_local_variable46 = #llvm.di_local_variable<scope = #di_lexical_block28, name = "sum", file = #di_file, line = 32, type = #di_basic_type>
#di_local_variable47 = #llvm.di_local_variable<scope = #di_lexical_block29, name = "sum", file = #di_file, line = 72, type = #di_basic_type>
#di_lexical_block34 = #llvm.di_lexical_block<scope = #di_lexical_block31, file = #di_file, line = 34>
#di_lexical_block35 = #llvm.di_lexical_block<scope = #di_lexical_block32, file = #di_file, line = 74>
#di_local_variable48 = #llvm.di_local_variable<scope = #di_lexical_block31, name = "kh", file = #di_file, line = 34, type = #di_derived_type5>
#di_local_variable49 = #llvm.di_local_variable<scope = #di_lexical_block32, name = "kh", file = #di_file, line = 74, type = #di_derived_type5>
#di_lexical_block36 = #llvm.di_lexical_block<scope = #di_lexical_block34, file = #di_file, line = 34>
#di_lexical_block37 = #llvm.di_lexical_block<scope = #di_lexical_block35, file = #di_file, line = 74>
#di_lexical_block38 = #llvm.di_lexical_block<scope = #di_lexical_block36, file = #di_file, line = 35>
#di_lexical_block39 = #llvm.di_lexical_block<scope = #di_lexical_block37, file = #di_file, line = 75>
#di_lexical_block40 = #llvm.di_lexical_block<scope = #di_lexical_block38, file = #di_file, line = 35>
#di_lexical_block41 = #llvm.di_lexical_block<scope = #di_lexical_block39, file = #di_file, line = 75>
#di_local_variable50 = #llvm.di_local_variable<scope = #di_lexical_block38, name = "kw", file = #di_file, line = 35, type = #di_derived_type5>
#di_local_variable51 = #llvm.di_local_variable<scope = #di_lexical_block39, name = "kw", file = #di_file, line = 75, type = #di_derived_type5>
#di_lexical_block42 = #llvm.di_lexical_block<scope = #di_lexical_block40, file = #di_file, line = 35>
#di_lexical_block43 = #llvm.di_lexical_block<scope = #di_lexical_block41, file = #di_file, line = 75>
#di_local_variable52 = #llvm.di_local_variable<scope = #di_lexical_block42, name = "h", file = #di_file, line = 36, type = #di_derived_type5>
#di_local_variable53 = #llvm.di_local_variable<scope = #di_lexical_block42, name = "w", file = #di_file, line = 37, type = #di_derived_type5>
#di_local_variable54 = #llvm.di_local_variable<scope = #di_lexical_block42, name = "input_val", file = #di_file, line = 39, type = #di_basic_type>
#di_local_variable55 = #llvm.di_local_variable<scope = #di_lexical_block42, name = "kernel_val", file = #di_file, line = 40, type = #di_basic_type>
#di_local_variable56 = #llvm.di_local_variable<scope = #di_lexical_block43, name = "h", file = #di_file, line = 76, type = #di_derived_type5>
#di_local_variable57 = #llvm.di_local_variable<scope = #di_lexical_block43, name = "w", file = #di_file, line = 77, type = #di_derived_type5>
#di_local_variable58 = #llvm.di_local_variable<scope = #di_lexical_block43, name = "input_val", file = #di_file, line = 79, type = #di_basic_type>
#di_local_variable59 = #llvm.di_local_variable<scope = #di_lexical_block43, name = "kernel_val", file = #di_file, line = 80, type = #di_basic_type>
#di_subprogram4 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "depthwise_conv_cpu", linkageName = "_Z18depthwise_conv_cpuPKfS0_Pfjjjjjjj", file = #di_file, line = 16, scopeLine = 25, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable15, #di_local_variable16, #di_local_variable4, #di_local_variable17, #di_local_variable18, #di_local_variable19, #di_local_variable20, #di_local_variable21, #di_local_variable22, #di_local_variable23, #di_local_variable5, #di_local_variable6, #di_local_variable7, #di_local_variable42, #di_local_variable44, #di_local_variable46, #di_local_variable48, #di_local_variable50, #di_local_variable52, #di_local_variable53, #di_local_variable54, #di_local_variable55>
#di_subprogram5 = #llvm.di_subprogram<recId = distinct[1]<>, id = distinct[7]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "depthwise_conv_dsa", linkageName = "_Z18depthwise_conv_dsaPKfS0_Pfjjjjjjj", file = #di_file, line = 53, scopeLine = 62, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type1, retainedNodes = #di_local_variable24, #di_local_variable25, #di_local_variable8, #di_local_variable26, #di_local_variable27, #di_local_variable28, #di_local_variable29, #di_local_variable30, #di_local_variable31, #di_local_variable32, #di_local_variable9, #di_local_variable10, #di_local_variable11, #di_local_variable43, #di_local_variable45, #di_local_variable47, #di_local_variable49, #di_local_variable51, #di_local_variable56, #di_local_variable57, #di_local_variable58, #di_local_variable59>
#di_lexical_block44 = #llvm.di_lexical_block<scope = #di_subprogram4, file = #di_file, line = 29>
#loc60 = loc(fused<#di_subprogram4>[#loc1])
#loc64 = loc(fused<#di_subprogram5>[#loc15])
#di_lexical_block46 = #llvm.di_lexical_block<scope = #di_lexical_block44, file = #di_file, line = 29>
#loc68 = loc(fused<#di_lexical_block44>[#loc5])
#di_lexical_block48 = #llvm.di_lexical_block<scope = #di_lexical_block46, file = #di_file, line = 29>
#di_lexical_block50 = #llvm.di_lexical_block<scope = #di_lexical_block48, file = #di_file, line = 30>
#di_lexical_block52 = #llvm.di_lexical_block<scope = #di_lexical_block50, file = #di_file, line = 30>
#loc73 = loc(fused<#di_lexical_block50>[#loc6])
#di_lexical_block54 = #llvm.di_lexical_block<scope = #di_lexical_block52, file = #di_file, line = 30>
#di_lexical_block56 = #llvm.di_lexical_block<scope = #di_lexical_block54, file = #di_file, line = 31>
#di_lexical_block58 = #llvm.di_lexical_block<scope = #di_lexical_block56, file = #di_file, line = 31>
#loc76 = loc(fused<#di_lexical_block56>[#loc7])
#di_lexical_block60 = #llvm.di_lexical_block<scope = #di_lexical_block58, file = #di_file, line = 31>
#di_lexical_block62 = #llvm.di_lexical_block<scope = #di_lexical_block60, file = #di_file, line = 34>
#di_lexical_block64 = #llvm.di_lexical_block<scope = #di_lexical_block62, file = #di_file, line = 34>
#loc81 = loc(fused<#di_lexical_block62>[#loc8])
#di_lexical_block66 = #llvm.di_lexical_block<scope = #di_lexical_block64, file = #di_file, line = 34>
#di_lexical_block68 = #llvm.di_lexical_block<scope = #di_lexical_block66, file = #di_file, line = 35>
#loc85 = loc(fused<#di_lexical_block68>[#loc9])
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 23.0.0git (https://github.com/llvm/llvm-project.git b7c1a6f8b447fba6fff47d309eb7ba1bc22e8c53)", llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  memref.global constant @".str" : memref<25xi8> = dense<[108, 111, 111, 109, 46, 109, 101, 109, 111, 114, 121, 95, 98, 97, 110, 107, 61, 52, 44, 98, 108, 111, 99, 107, 0]> loc(#loc)
  memref.global constant @".str.1" : memref<44xi8> = dense<[116, 101, 115, 116, 115, 47, 97, 112, 112, 47, 100, 101, 112, 116, 104, 119, 105, 115, 101, 95, 99, 111, 110, 118, 47, 100, 101, 112, 116, 104, 119, 105, 115, 101, 95, 99, 111, 110, 118, 46, 99, 112, 112, 0]> loc(#loc)
  memref.global constant @".str.2" : memref<12xi8> = dense<[108, 111, 111, 109, 46, 115, 116, 114, 101, 97, 109, 0]> loc(#loc)
  memref.global constant @".str.3" : memref<11xi8> = dense<[108, 111, 111, 109, 46, 97, 99, 99, 101, 108, 0]> loc(#loc)
  memref.global constant @str : memref<23xi8> = dense<[100, 101, 112, 116, 104, 119, 105, 115, 101, 95, 99, 111, 110, 118, 58, 32, 70, 65, 73, 76, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  memref.global constant @str.2 : memref<23xi8> = dense<[100, 101, 112, 116, 104, 119, 105, 115, 101, 95, 99, 111, 110, 118, 58, 32, 80, 65, 83, 83, 69, 68, 0]> {alignment = 1 : i64} loc(#loc)
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>] loc(#loc)
  func.func @_Z18depthwise_conv_cpuPKfS0_Pfjjjjjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg1: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg2: memref<?xf32, strided<[1], offset: ?>> {loom.noalias} loc(fused<#di_subprogram4>[#loc1]), %arg3: i32 loc(fused<#di_subprogram4>[#loc1]), %arg4: i32 loc(fused<#di_subprogram4>[#loc1]), %arg5: i32 loc(fused<#di_subprogram4>[#loc1]), %arg6: i32 loc(fused<#di_subprogram4>[#loc1]), %arg7: i32 loc(fused<#di_subprogram4>[#loc1]), %arg8: i32 loc(fused<#di_subprogram4>[#loc1]), %arg9: i32 loc(fused<#di_subprogram4>[#loc1])) {
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %0 = arith.subi %arg4, %arg6 : i32 loc(#loc61)
    %1 = arith.divui %0, %arg8 : i32 loc(#loc61)
    %2 = arith.addi %1, %c1_i32 : i32 loc(#loc61)
    %3 = arith.subi %arg5, %arg7 : i32 loc(#loc62)
    %4 = arith.divui %3, %arg9 : i32 loc(#loc62)
    %5 = arith.addi %4, %c1_i32 : i32 loc(#loc62)
    %6 = arith.cmpi eq, %arg3, %c0_i32 : i32 loc(#loc70)
    scf.if %6 {
    } else {
      %7 = arith.cmpi eq, %2, %c0_i32 : i32 loc(#loc2)
      %8 = arith.cmpi eq, %5, %c0_i32 : i32 loc(#loc2)
      %9 = arith.cmpi eq, %arg6, %c0_i32 : i32 loc(#loc2)
      %10 = arith.cmpi eq, %arg7, %c0_i32 : i32 loc(#loc2)
      %11 = arith.extui %5 : i32 to i64 loc(#loc68)
      %12 = arith.extui %arg7 : i32 to i64 loc(#loc2)
      %13 = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
        scf.if %7 {
        } else {
          %16 = arith.muli %arg10, %arg4 : i32 loc(#loc2)
          %17 = arith.muli %arg10, %arg6 : i32 loc(#loc2)
          %18 = arith.muli %arg10, %2 : i32 loc(#loc2)
          %19 = scf.while (%arg11 = %c0_i32) : (i32) -> i32 {
            scf.if %8 {
            } else {
              %22 = arith.muli %arg11, %arg8 : i32 loc(#loc2)
              %23 = arith.addi %22, %16 : i32 loc(#loc2)
              %24 = arith.addi %arg11, %18 : i32 loc(#loc2)
              %25 = arith.muli %24, %5 : i32 loc(#loc2)
              %26 = scf.while (%arg12 = %c0_i64) : (i64) -> i64 {
                %27 = scf.if %9 -> (f32) {
                  scf.yield %cst : f32 loc(#loc81)
                } else {
                  %34 = arith.trunci %arg12 : i64 to i32 loc(#loc2)
                  %35 = arith.muli %arg9, %34 : i32 loc(#loc2)
                  %36:2 = scf.while (%arg13 = %c0_i32, %arg14 = %cst) : (i32, f32) -> (i32, f32) {
                    %37 = scf.if %10 -> (f32) {
                      scf.yield %arg14 : f32 loc(#loc85)
                    } else {
                      %40 = arith.addi %23, %arg13 : i32 loc(#loc2)
                      %41 = arith.muli %40, %arg5 : i32 loc(#loc2)
                      %42 = arith.addi %41, %35 : i32 loc(#loc2)
                      %43 = arith.addi %arg13, %17 : i32 loc(#loc2)
                      %44 = arith.muli %43, %arg7 : i32 loc(#loc2)
                      %45:2 = scf.while (%arg15 = %c0_i64, %arg16 = %arg14) : (i64, f32) -> (i64, f32) {
                        %46 = arith.trunci %arg15 : i64 to i32 loc(#loc88)
                        %47 = arith.addi %42, %46 : i32 loc(#loc88)
                        %48 = arith.extui %47 : i32 to i64 loc(#loc88)
                        %49 = arith.index_cast %48 : i64 to index loc(#loc88)
                        %50 = memref.load %arg0[%49] : memref<?xf32, strided<[1], offset: ?>> loc(#loc88)
                        %51 = arith.addi %44, %46 : i32 loc(#loc89)
                        %52 = arith.extui %51 : i32 to i64 loc(#loc89)
                        %53 = arith.index_cast %52 : i64 to index loc(#loc89)
                        %54 = memref.load %arg1[%53] : memref<?xf32, strided<[1], offset: ?>> loc(#loc89)
                        %55 = math.fma %50, %54, %arg16 : f32 loc(#loc90)
                        %56 = arith.addi %arg15, %c1_i64 : i64 loc(#loc87)
                        %57 = arith.cmpi ne, %56, %12 : i64 loc(#loc91)
                        scf.condition(%57) %56, %55 : i64, f32 loc(#loc85)
                      } do {
                      ^bb0(%arg15: i64 loc(fused<#di_lexical_block68>[#loc9]), %arg16: f32 loc(fused<#di_lexical_block68>[#loc9])):
                        scf.yield %arg15, %arg16 : i64, f32 loc(#loc85)
                      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc85)
                      scf.yield %45#1 : f32 loc(#loc85)
                    } loc(#loc85)
                    %38 = arith.addi %arg13, %c1_i32 : i32 loc(#loc83)
                    %39 = arith.cmpi ne, %38, %arg6 : i32 loc(#loc84)
                    scf.condition(%39) %38, %37 : i32, f32 loc(#loc81)
                  } do {
                  ^bb0(%arg13: i32 loc(fused<#di_lexical_block62>[#loc8]), %arg14: f32 loc(fused<#di_lexical_block62>[#loc8])):
                    scf.yield %arg13, %arg14 : i32, f32 loc(#loc81)
                  } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc81)
                  scf.yield %36#1 : f32 loc(#loc81)
                } loc(#loc81)
                %28 = arith.trunci %arg12 : i64 to i32 loc(#loc79)
                %29 = arith.addi %25, %28 : i32 loc(#loc79)
                %30 = arith.extui %29 : i32 to i64 loc(#loc79)
                %31 = arith.index_cast %30 : i64 to index loc(#loc79)
                memref.store %27, %arg2[%31] : memref<?xf32, strided<[1], offset: ?>> loc(#loc79)
                %32 = arith.addi %arg12, %c1_i64 : i64 loc(#loc78)
                %33 = arith.cmpi ult, %32, %11 : i64 loc(#loc78)
                scf.condition(%33) %32 : i64 loc(#loc76)
              } do {
              ^bb0(%arg12: i64 loc(fused<#di_lexical_block56>[#loc7])):
                scf.yield %arg12 : i64 loc(#loc76)
              } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc76)
            } loc(#loc76)
            %20 = arith.addi %arg11, %c1_i32 : i32 loc(#loc75)
            %21 = arith.cmpi ult, %20, %2 : i32 loc(#loc75)
            scf.condition(%21) %20 : i32 loc(#loc73)
          } do {
          ^bb0(%arg11: i32 loc(fused<#di_lexical_block50>[#loc6])):
            scf.yield %arg11 : i32 loc(#loc73)
          } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "<"}} loc(#loc73)
        } loc(#loc73)
        %14 = arith.addi %arg10, %c1_i32 : i32 loc(#loc70)
        %15 = arith.cmpi ne, %14, %arg3 : i32 loc(#loc72)
        scf.condition(%15) %14 : i32 loc(#loc68)
      } do {
      ^bb0(%arg10: i32 loc(fused<#di_lexical_block44>[#loc5])):
        scf.yield %arg10 : i32 loc(#loc68)
      } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc68)
    } loc(#loc68)
    return loc(#loc63)
  } loc(#loc60)
  func.func private @llvm.fmuladd.f32(f32, f32, f32) -> f32 loc(#loc2)
  handshake.func @_Z18depthwise_conv_dsaPKfS0_Pfjjjjjjj_esi(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg3: i32 loc(fused<#di_subprogram5>[#loc15]), %arg4: i32 loc(fused<#di_subprogram5>[#loc15]), %arg5: i32 loc(fused<#di_subprogram5>[#loc15]), %arg6: i32 loc(fused<#di_subprogram5>[#loc15]), %arg7: i32 loc(fused<#di_subprogram5>[#loc15]), %arg8: i32 loc(fused<#di_subprogram5>[#loc15]), %arg9: i32 loc(fused<#di_subprogram5>[#loc15]), %arg10: i1 loc(fused<#di_subprogram5>[#loc15]), ...) -> i1 attributes {argNames = ["input", "kernel", "output", "C", "H", "W", "KH", "KW", "stride_h", "stride_w", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg10 : i1 loc(#loc64)
    %1 = handshake.join %0 : none loc(#loc64)
    %2 = handshake.constant %1 {value = 1 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %1 {value = 0 : i32} : i32 loc(#loc2)
    %4 = handshake.constant %1 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %5 = handshake.constant %1 {value = 0 : i64} : i64 loc(#loc2)
    %6 = arith.subi %arg4, %arg6 : i32 loc(#loc65)
    %7 = arith.divui %6, %arg8 : i32 loc(#loc65)
    %8 = arith.addi %7, %2 : i32 loc(#loc65)
    %9 = arith.subi %arg5, %arg7 : i32 loc(#loc66)
    %10 = arith.divui %9, %arg9 : i32 loc(#loc66)
    %11 = arith.addi %10, %2 : i32 loc(#loc66)
    %12 = arith.cmpi eq, %arg3, %3 : i32 loc(#loc71)
    %trueResult, %falseResult = handshake.cond_br %12, %1 : none loc(#loc69)
    %13 = arith.cmpi eq, %8, %3 : i32 loc(#loc2)
    %14 = arith.cmpi eq, %11, %3 : i32 loc(#loc2)
    %15 = arith.cmpi eq, %arg6, %3 : i32 loc(#loc2)
    %16 = arith.cmpi eq, %arg7, %3 : i32 loc(#loc2)
    %17 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc69)
    %18 = arith.index_cast %3 : i32 to index loc(#loc69)
    %19 = arith.index_cast %arg3 : i32 to index loc(#loc69)
    %index, %willContinue = dataflow.stream %18, %17, %19 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc69)
    %20 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc69)
    %21 = arith.index_cast %afterValue : index to i32 loc(#loc69)
    %22 = dataflow.invariant %afterCond, %13 : i1, i1 -> i1 loc(#loc74)
    %trueResult_0, %falseResult_1 = handshake.cond_br %22, %20 : none loc(#loc74)
    %23 = arith.muli %21, %arg4 : i32 loc(#loc2)
    %24 = arith.muli %21, %arg6 : i32 loc(#loc2)
    %25 = arith.muli %21, %8 : i32 loc(#loc2)
    %26 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc74)
    %27 = arith.index_cast %8 : i32 to index loc(#loc74)
    %index_2, %willContinue_3 = dataflow.stream %18, %26, %27 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "<"} loc(#loc74)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc74)
    %28 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc74)
    %29 = arith.index_cast %afterValue_4 : index to i32 loc(#loc74)
    %30 = dataflow.invariant %afterCond, %14 : i1, i1 -> i1 loc(#loc77)
    %trueResult_6, %falseResult_7 = handshake.cond_br %30, %28 : none loc(#loc77)
    %31 = arith.muli %29, %arg8 : i32 loc(#loc2)
    %32 = dataflow.invariant %afterCond_5, %23 : i1, i32 -> i32 loc(#loc2)
    %33 = arith.addi %31, %32 : i32 loc(#loc2)
    %34 = dataflow.invariant %afterCond_5, %25 : i1, i32 -> i32 loc(#loc2)
    %35 = arith.addi %29, %34 : i32 loc(#loc2)
    %36 = arith.muli %35, %11 : i32 loc(#loc2)
    %37 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc77)
    %38 = arith.index_cast %5 : i64 to index loc(#loc77)
    %39 = arith.index_cast %11 : i32 to index loc(#loc77)
    %index_8, %willContinue_9 = dataflow.stream %38, %37, %39 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "<"} loc(#loc77)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc77)
    %40 = dataflow.invariant %afterCond_11, %falseResult_7 : i1, none -> none loc(#loc77)
    %41 = arith.index_cast %afterValue_10 : index to i64 loc(#loc77)
    %42 = dataflow.invariant %afterCond, %15 : i1, i1 -> i1 loc(#loc82)
    %trueResult_12, %falseResult_13 = handshake.cond_br %42, %40 : none loc(#loc82)
    %43 = arith.trunci %41 : i64 to i32 loc(#loc80)
    %44 = arith.muli %arg9, %43 : i32 loc(#loc2)
    %45 = handshake.constant %falseResult_13 {value = 1 : index} : index loc(#loc82)
    %46 = arith.index_cast %arg6 : i32 to index loc(#loc82)
    %index_14, %willContinue_15 = dataflow.stream %18, %45, %46 {step_op = "+=", stop_cond = "!="} loc(#loc82)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc82)
    %47 = dataflow.carry %willContinue_15, %4, %76 : i1, f32, f32 -> f32 loc(#loc82)
    %afterValue_18, %afterCond_19 = dataflow.gate %47, %willContinue_15 : f32, i1 -> f32, i1 loc(#loc82)
    handshake.sink %afterCond_19 : i1 loc(#loc82)
    %trueResult_20, %falseResult_21 = handshake.cond_br %willContinue_15, %47 : f32 loc(#loc82)
    %48 = dataflow.invariant %afterCond_17, %falseResult_13 : i1, none -> none loc(#loc82)
    %49 = arith.index_cast %afterValue_16 : index to i32 loc(#loc82)
    %50 = dataflow.invariant %afterCond, %16 : i1, i1 -> i1 loc(#loc86)
    %trueResult_22, %falseResult_23 = handshake.cond_br %50, %48 : none loc(#loc86)
    %51 = dataflow.invariant %afterCond_11, %33 : i1, i32 -> i32 loc(#loc2)
    %52 = arith.addi %51, %49 : i32 loc(#loc2)
    %53 = arith.muli %52, %arg5 : i32 loc(#loc2)
    %54 = dataflow.invariant %afterCond_17, %44 : i1, i32 -> i32 loc(#loc2)
    %55 = arith.addi %53, %54 : i32 loc(#loc2)
    %56 = dataflow.invariant %afterCond_5, %24 : i1, i32 -> i32 loc(#loc2)
    %57 = arith.addi %49, %56 : i32 loc(#loc2)
    %58 = arith.muli %57, %arg7 : i32 loc(#loc2)
    %59 = handshake.constant %falseResult_23 {value = 1 : index} : index loc(#loc86)
    %60 = arith.index_cast %arg7 : i32 to index loc(#loc86)
    %index_24, %willContinue_25 = dataflow.stream %38, %59, %60 {step_op = "+=", stop_cond = "!="} loc(#loc86)
    %afterValue_26, %afterCond_27 = dataflow.gate %index_24, %willContinue_25 : index, i1 -> index, i1 loc(#loc86)
    %61 = dataflow.carry %willContinue_25, %afterValue_18, %72 : i1, f32, f32 -> f32 loc(#loc86)
    %afterValue_28, %afterCond_29 = dataflow.gate %61, %willContinue_25 : f32, i1 -> f32, i1 loc(#loc86)
    handshake.sink %afterCond_29 : i1 loc(#loc86)
    %trueResult_30, %falseResult_31 = handshake.cond_br %willContinue_25, %61 : f32 loc(#loc86)
    %62 = arith.index_cast %afterValue_26 : index to i64 loc(#loc86)
    %63 = arith.trunci %62 : i64 to i32 loc(#loc92)
    %64 = dataflow.invariant %afterCond_27, %55 : i1, i32 -> i32 loc(#loc92)
    %65 = arith.addi %64, %63 : i32 loc(#loc92)
    %66 = arith.extui %65 : i32 to i64 loc(#loc92)
    %67 = arith.index_cast %66 : i64 to index loc(#loc92)
    %dataResult, %addressResults = handshake.load [%67] %85#0, %92 : index, f32 loc(#loc92)
    %68 = dataflow.invariant %afterCond_27, %58 : i1, i32 -> i32 loc(#loc93)
    %69 = arith.addi %68, %63 : i32 loc(#loc93)
    %70 = arith.extui %69 : i32 to i64 loc(#loc93)
    %71 = arith.index_cast %70 : i64 to index loc(#loc93)
    %dataResult_32, %addressResults_33 = handshake.load [%71] %86#0, %129 : index, f32 loc(#loc93)
    %72 = math.fma %dataResult, %dataResult_32, %afterValue_28 : f32 loc(#loc94)
    %73 = handshake.constant %48 {value = 0 : index} : index loc(#loc86)
    %74 = handshake.constant %48 {value = 1 : index} : index loc(#loc86)
    %75 = arith.select %50, %74, %73 : index loc(#loc86)
    %76 = handshake.mux %75 [%falseResult_31, %afterValue_18] : index, f32 loc(#loc86)
    %77 = handshake.constant %40 {value = 0 : index} : index loc(#loc82)
    %78 = handshake.constant %40 {value = 1 : index} : index loc(#loc82)
    %79 = arith.select %42, %78, %77 : index loc(#loc82)
    %80 = handshake.mux %79 [%falseResult_21, %4] : index, f32 loc(#loc82)
    %81 = dataflow.invariant %afterCond_11, %36 : i1, i32 -> i32 loc(#loc80)
    %82 = arith.addi %81, %43 : i32 loc(#loc80)
    %83 = arith.extui %82 : i32 to i64 loc(#loc80)
    %84 = arith.index_cast %83 : i64 to index loc(#loc80)
    %dataResult_34, %addressResult = handshake.store [%84] %80, %115 : index, f32 loc(#loc80)
    %85:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc64)
    %86:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_33) {id = 1 : i32} : (index) -> (f32, none) loc(#loc64)
    %87 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_34, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc64)
    %88 = dataflow.carry %willContinue, %falseResult, %trueResult_51 : i1, none, none -> none loc(#loc69)
    %trueResult_35, %falseResult_36 = handshake.cond_br %22, %88 : none loc(#loc74)
    %89 = dataflow.carry %willContinue_3, %falseResult_36, %trueResult_49 : i1, none, none -> none loc(#loc74)
    %trueResult_37, %falseResult_38 = handshake.cond_br %30, %89 : none loc(#loc77)
    %90 = dataflow.carry %willContinue_9, %falseResult_38, %trueResult_47 : i1, none, none -> none loc(#loc77)
    %trueResult_39, %falseResult_40 = handshake.cond_br %42, %90 : none loc(#loc82)
    %91 = dataflow.carry %willContinue_15, %falseResult_40, %trueResult_45 : i1, none, none -> none loc(#loc82)
    %trueResult_41, %falseResult_42 = handshake.cond_br %50, %91 : none loc(#loc86)
    %92 = dataflow.carry %willContinue_25, %falseResult_42, %trueResult_43 : i1, none, none -> none loc(#loc86)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue_25, %85#1 : none loc(#loc86)
    %93 = handshake.constant %91 {value = 0 : index} : index loc(#loc86)
    %94 = handshake.constant %91 {value = 1 : index} : index loc(#loc86)
    %95 = arith.select %50, %94, %93 : index loc(#loc86)
    %96 = handshake.mux %95 [%falseResult_44, %trueResult_41] : index, none loc(#loc86)
    %trueResult_45, %falseResult_46 = handshake.cond_br %willContinue_15, %96 : none loc(#loc82)
    %97 = handshake.constant %90 {value = 0 : index} : index loc(#loc82)
    %98 = handshake.constant %90 {value = 1 : index} : index loc(#loc82)
    %99 = arith.select %42, %98, %97 : index loc(#loc82)
    %100 = handshake.mux %99 [%falseResult_46, %trueResult_39] : index, none loc(#loc82)
    %trueResult_47, %falseResult_48 = handshake.cond_br %willContinue_9, %100 : none loc(#loc77)
    %101 = handshake.constant %89 {value = 0 : index} : index loc(#loc77)
    %102 = handshake.constant %89 {value = 1 : index} : index loc(#loc77)
    %103 = arith.select %30, %102, %101 : index loc(#loc77)
    %104 = handshake.mux %103 [%falseResult_48, %trueResult_37] : index, none loc(#loc77)
    %trueResult_49, %falseResult_50 = handshake.cond_br %willContinue_3, %104 : none loc(#loc74)
    %105 = handshake.constant %88 {value = 0 : index} : index loc(#loc74)
    %106 = handshake.constant %88 {value = 1 : index} : index loc(#loc74)
    %107 = arith.select %22, %106, %105 : index loc(#loc74)
    %108 = handshake.mux %107 [%falseResult_50, %trueResult_35] : index, none loc(#loc74)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue, %108 : none loc(#loc69)
    %109 = handshake.constant %1 {value = 0 : index} : index loc(#loc69)
    %110 = handshake.constant %1 {value = 1 : index} : index loc(#loc69)
    %111 = arith.select %12, %110, %109 : index loc(#loc69)
    %112 = handshake.mux %111 [%falseResult_52, %trueResult] : index, none loc(#loc69)
    %113 = dataflow.carry %willContinue, %falseResult, %trueResult_61 : i1, none, none -> none loc(#loc69)
    %trueResult_53, %falseResult_54 = handshake.cond_br %22, %113 : none loc(#loc74)
    %114 = dataflow.carry %willContinue_3, %falseResult_54, %trueResult_59 : i1, none, none -> none loc(#loc74)
    %trueResult_55, %falseResult_56 = handshake.cond_br %30, %114 : none loc(#loc77)
    %115 = dataflow.carry %willContinue_9, %falseResult_56, %trueResult_57 : i1, none, none -> none loc(#loc77)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue_9, %87 : none loc(#loc77)
    %116 = handshake.constant %114 {value = 0 : index} : index loc(#loc77)
    %117 = handshake.constant %114 {value = 1 : index} : index loc(#loc77)
    %118 = arith.select %30, %117, %116 : index loc(#loc77)
    %119 = handshake.mux %118 [%falseResult_58, %trueResult_55] : index, none loc(#loc77)
    %trueResult_59, %falseResult_60 = handshake.cond_br %willContinue_3, %119 : none loc(#loc74)
    %120 = handshake.constant %113 {value = 0 : index} : index loc(#loc74)
    %121 = handshake.constant %113 {value = 1 : index} : index loc(#loc74)
    %122 = arith.select %22, %121, %120 : index loc(#loc74)
    %123 = handshake.mux %122 [%falseResult_60, %trueResult_53] : index, none loc(#loc74)
    %trueResult_61, %falseResult_62 = handshake.cond_br %willContinue, %123 : none loc(#loc69)
    %124 = handshake.mux %111 [%falseResult_62, %trueResult] : index, none loc(#loc69)
    %125 = dataflow.carry %willContinue, %falseResult, %trueResult_79 : i1, none, none -> none loc(#loc69)
    %trueResult_63, %falseResult_64 = handshake.cond_br %22, %125 : none loc(#loc74)
    %126 = dataflow.carry %willContinue_3, %falseResult_64, %trueResult_77 : i1, none, none -> none loc(#loc74)
    %trueResult_65, %falseResult_66 = handshake.cond_br %30, %126 : none loc(#loc77)
    %127 = dataflow.carry %willContinue_9, %falseResult_66, %trueResult_75 : i1, none, none -> none loc(#loc77)
    %trueResult_67, %falseResult_68 = handshake.cond_br %42, %127 : none loc(#loc82)
    %128 = dataflow.carry %willContinue_15, %falseResult_68, %trueResult_73 : i1, none, none -> none loc(#loc82)
    %trueResult_69, %falseResult_70 = handshake.cond_br %50, %128 : none loc(#loc86)
    %129 = dataflow.carry %willContinue_25, %falseResult_70, %trueResult_71 : i1, none, none -> none loc(#loc86)
    %trueResult_71, %falseResult_72 = handshake.cond_br %willContinue_25, %86#1 : none loc(#loc86)
    %130 = handshake.constant %128 {value = 0 : index} : index loc(#loc86)
    %131 = handshake.constant %128 {value = 1 : index} : index loc(#loc86)
    %132 = arith.select %50, %131, %130 : index loc(#loc86)
    %133 = handshake.mux %132 [%falseResult_72, %trueResult_69] : index, none loc(#loc86)
    %trueResult_73, %falseResult_74 = handshake.cond_br %willContinue_15, %133 : none loc(#loc82)
    %134 = handshake.constant %127 {value = 0 : index} : index loc(#loc82)
    %135 = handshake.constant %127 {value = 1 : index} : index loc(#loc82)
    %136 = arith.select %42, %135, %134 : index loc(#loc82)
    %137 = handshake.mux %136 [%falseResult_74, %trueResult_67] : index, none loc(#loc82)
    %trueResult_75, %falseResult_76 = handshake.cond_br %willContinue_9, %137 : none loc(#loc77)
    %138 = handshake.constant %126 {value = 0 : index} : index loc(#loc77)
    %139 = handshake.constant %126 {value = 1 : index} : index loc(#loc77)
    %140 = arith.select %30, %139, %138 : index loc(#loc77)
    %141 = handshake.mux %140 [%falseResult_76, %trueResult_65] : index, none loc(#loc77)
    %trueResult_77, %falseResult_78 = handshake.cond_br %willContinue_3, %141 : none loc(#loc74)
    %142 = handshake.constant %125 {value = 0 : index} : index loc(#loc74)
    %143 = handshake.constant %125 {value = 1 : index} : index loc(#loc74)
    %144 = arith.select %22, %143, %142 : index loc(#loc74)
    %145 = handshake.mux %144 [%falseResult_78, %trueResult_63] : index, none loc(#loc74)
    %trueResult_79, %falseResult_80 = handshake.cond_br %willContinue, %145 : none loc(#loc69)
    %146 = handshake.mux %111 [%falseResult_80, %trueResult] : index, none loc(#loc69)
    %147 = handshake.join %112, %124, %146 : none, none, none loc(#loc64)
    %148 = handshake.constant %147 {value = true} : i1 loc(#loc64)
    handshake.return %148 : i1 loc(#loc64)
  } loc(#loc64)
  handshake.func @_Z18depthwise_conv_dsaPKfS0_Pfjjjjjjj(%arg0: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg1: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg2: memref<?xf32, strided<[1], offset: ?>> loc(fused<#di_subprogram5>[#loc15]), %arg3: i32 loc(fused<#di_subprogram5>[#loc15]), %arg4: i32 loc(fused<#di_subprogram5>[#loc15]), %arg5: i32 loc(fused<#di_subprogram5>[#loc15]), %arg6: i32 loc(fused<#di_subprogram5>[#loc15]), %arg7: i32 loc(fused<#di_subprogram5>[#loc15]), %arg8: i32 loc(fused<#di_subprogram5>[#loc15]), %arg9: i32 loc(fused<#di_subprogram5>[#loc15]), %arg10: none loc(fused<#di_subprogram5>[#loc15]), ...) -> none attributes {argNames = ["input", "kernel", "output", "C", "H", "W", "KH", "KW", "stride_h", "stride_w", "start_token"], loom.annotations = ["loom.accel"], resNames = ["done_token"]} {
    %0 = handshake.join %arg10 : none loc(#loc64)
    %1 = handshake.constant %0 {value = 1 : i32} : i32 loc(#loc2)
    %2 = handshake.constant %0 {value = 0 : i32} : i32 loc(#loc2)
    %3 = handshake.constant %0 {value = 0.000000e+00 : f32} : f32 loc(#loc2)
    %4 = handshake.constant %0 {value = 0 : i64} : i64 loc(#loc2)
    %5 = arith.subi %arg4, %arg6 : i32 loc(#loc65)
    %6 = arith.divui %5, %arg8 : i32 loc(#loc65)
    %7 = arith.addi %6, %1 : i32 loc(#loc65)
    %8 = arith.subi %arg5, %arg7 : i32 loc(#loc66)
    %9 = arith.divui %8, %arg9 : i32 loc(#loc66)
    %10 = arith.addi %9, %1 : i32 loc(#loc66)
    %11 = arith.cmpi eq, %arg3, %2 : i32 loc(#loc71)
    %trueResult, %falseResult = handshake.cond_br %11, %0 : none loc(#loc69)
    %12 = arith.cmpi eq, %7, %2 : i32 loc(#loc2)
    %13 = arith.cmpi eq, %10, %2 : i32 loc(#loc2)
    %14 = arith.cmpi eq, %arg6, %2 : i32 loc(#loc2)
    %15 = arith.cmpi eq, %arg7, %2 : i32 loc(#loc2)
    %16 = handshake.constant %falseResult {value = 1 : index} : index loc(#loc69)
    %17 = arith.index_cast %2 : i32 to index loc(#loc69)
    %18 = arith.index_cast %arg3 : i32 to index loc(#loc69)
    %index, %willContinue = dataflow.stream %17, %16, %18 {loom.annotations = ["loom.loop.parallel degree=4 schedule=1"], step_op = "+=", stop_cond = "!="} loc(#loc69)
    %afterValue, %afterCond = dataflow.gate %index, %willContinue : index, i1 -> index, i1 loc(#loc69)
    %19 = dataflow.invariant %afterCond, %falseResult : i1, none -> none loc(#loc69)
    %20 = arith.index_cast %afterValue : index to i32 loc(#loc69)
    %21 = dataflow.invariant %afterCond, %12 : i1, i1 -> i1 loc(#loc74)
    %trueResult_0, %falseResult_1 = handshake.cond_br %21, %19 : none loc(#loc74)
    %22 = arith.muli %20, %arg4 : i32 loc(#loc2)
    %23 = arith.muli %20, %arg6 : i32 loc(#loc2)
    %24 = arith.muli %20, %7 : i32 loc(#loc2)
    %25 = handshake.constant %falseResult_1 {value = 1 : index} : index loc(#loc74)
    %26 = arith.index_cast %7 : i32 to index loc(#loc74)
    %index_2, %willContinue_3 = dataflow.stream %17, %25, %26 {loom.annotations = ["loom.loop.unroll factor=4"], step_op = "+=", stop_cond = "<"} loc(#loc74)
    %afterValue_4, %afterCond_5 = dataflow.gate %index_2, %willContinue_3 : index, i1 -> index, i1 loc(#loc74)
    %27 = dataflow.invariant %afterCond_5, %falseResult_1 : i1, none -> none loc(#loc74)
    %28 = arith.index_cast %afterValue_4 : index to i32 loc(#loc74)
    %29 = dataflow.invariant %afterCond, %13 : i1, i1 -> i1 loc(#loc77)
    %trueResult_6, %falseResult_7 = handshake.cond_br %29, %27 : none loc(#loc77)
    %30 = arith.muli %28, %arg8 : i32 loc(#loc2)
    %31 = dataflow.invariant %afterCond_5, %22 : i1, i32 -> i32 loc(#loc2)
    %32 = arith.addi %30, %31 : i32 loc(#loc2)
    %33 = dataflow.invariant %afterCond_5, %24 : i1, i32 -> i32 loc(#loc2)
    %34 = arith.addi %28, %33 : i32 loc(#loc2)
    %35 = arith.muli %34, %10 : i32 loc(#loc2)
    %36 = handshake.constant %falseResult_7 {value = 1 : index} : index loc(#loc77)
    %37 = arith.index_cast %4 : i64 to index loc(#loc77)
    %38 = arith.index_cast %10 : i32 to index loc(#loc77)
    %index_8, %willContinue_9 = dataflow.stream %37, %36, %38 {loom.annotations = ["loom.loop.tripcount typical=16 avg=16 min=1 max=64"], step_op = "+=", stop_cond = "<"} loc(#loc77)
    %afterValue_10, %afterCond_11 = dataflow.gate %index_8, %willContinue_9 : index, i1 -> index, i1 loc(#loc77)
    %39 = dataflow.invariant %afterCond_11, %falseResult_7 : i1, none -> none loc(#loc77)
    %40 = arith.index_cast %afterValue_10 : index to i64 loc(#loc77)
    %41 = dataflow.invariant %afterCond, %14 : i1, i1 -> i1 loc(#loc82)
    %trueResult_12, %falseResult_13 = handshake.cond_br %41, %39 : none loc(#loc82)
    %42 = arith.trunci %40 : i64 to i32 loc(#loc80)
    %43 = arith.muli %arg9, %42 : i32 loc(#loc2)
    %44 = handshake.constant %falseResult_13 {value = 1 : index} : index loc(#loc82)
    %45 = arith.index_cast %arg6 : i32 to index loc(#loc82)
    %index_14, %willContinue_15 = dataflow.stream %17, %44, %45 {step_op = "+=", stop_cond = "!="} loc(#loc82)
    %afterValue_16, %afterCond_17 = dataflow.gate %index_14, %willContinue_15 : index, i1 -> index, i1 loc(#loc82)
    %46 = dataflow.carry %willContinue_15, %3, %75 : i1, f32, f32 -> f32 loc(#loc82)
    %afterValue_18, %afterCond_19 = dataflow.gate %46, %willContinue_15 : f32, i1 -> f32, i1 loc(#loc82)
    handshake.sink %afterCond_19 : i1 loc(#loc82)
    %trueResult_20, %falseResult_21 = handshake.cond_br %willContinue_15, %46 : f32 loc(#loc82)
    %47 = dataflow.invariant %afterCond_17, %falseResult_13 : i1, none -> none loc(#loc82)
    %48 = arith.index_cast %afterValue_16 : index to i32 loc(#loc82)
    %49 = dataflow.invariant %afterCond, %15 : i1, i1 -> i1 loc(#loc86)
    %trueResult_22, %falseResult_23 = handshake.cond_br %49, %47 : none loc(#loc86)
    %50 = dataflow.invariant %afterCond_11, %32 : i1, i32 -> i32 loc(#loc2)
    %51 = arith.addi %50, %48 : i32 loc(#loc2)
    %52 = arith.muli %51, %arg5 : i32 loc(#loc2)
    %53 = dataflow.invariant %afterCond_17, %43 : i1, i32 -> i32 loc(#loc2)
    %54 = arith.addi %52, %53 : i32 loc(#loc2)
    %55 = dataflow.invariant %afterCond_5, %23 : i1, i32 -> i32 loc(#loc2)
    %56 = arith.addi %48, %55 : i32 loc(#loc2)
    %57 = arith.muli %56, %arg7 : i32 loc(#loc2)
    %58 = handshake.constant %falseResult_23 {value = 1 : index} : index loc(#loc86)
    %59 = arith.index_cast %arg7 : i32 to index loc(#loc86)
    %index_24, %willContinue_25 = dataflow.stream %37, %58, %59 {step_op = "+=", stop_cond = "!="} loc(#loc86)
    %afterValue_26, %afterCond_27 = dataflow.gate %index_24, %willContinue_25 : index, i1 -> index, i1 loc(#loc86)
    %60 = dataflow.carry %willContinue_25, %afterValue_18, %71 : i1, f32, f32 -> f32 loc(#loc86)
    %afterValue_28, %afterCond_29 = dataflow.gate %60, %willContinue_25 : f32, i1 -> f32, i1 loc(#loc86)
    handshake.sink %afterCond_29 : i1 loc(#loc86)
    %trueResult_30, %falseResult_31 = handshake.cond_br %willContinue_25, %60 : f32 loc(#loc86)
    %61 = arith.index_cast %afterValue_26 : index to i64 loc(#loc86)
    %62 = arith.trunci %61 : i64 to i32 loc(#loc92)
    %63 = dataflow.invariant %afterCond_27, %54 : i1, i32 -> i32 loc(#loc92)
    %64 = arith.addi %63, %62 : i32 loc(#loc92)
    %65 = arith.extui %64 : i32 to i64 loc(#loc92)
    %66 = arith.index_cast %65 : i64 to index loc(#loc92)
    %dataResult, %addressResults = handshake.load [%66] %84#0, %91 : index, f32 loc(#loc92)
    %67 = dataflow.invariant %afterCond_27, %57 : i1, i32 -> i32 loc(#loc93)
    %68 = arith.addi %67, %62 : i32 loc(#loc93)
    %69 = arith.extui %68 : i32 to i64 loc(#loc93)
    %70 = arith.index_cast %69 : i64 to index loc(#loc93)
    %dataResult_32, %addressResults_33 = handshake.load [%70] %85#0, %128 : index, f32 loc(#loc93)
    %71 = math.fma %dataResult, %dataResult_32, %afterValue_28 : f32 loc(#loc94)
    %72 = handshake.constant %47 {value = 0 : index} : index loc(#loc86)
    %73 = handshake.constant %47 {value = 1 : index} : index loc(#loc86)
    %74 = arith.select %49, %73, %72 : index loc(#loc86)
    %75 = handshake.mux %74 [%falseResult_31, %afterValue_18] : index, f32 loc(#loc86)
    %76 = handshake.constant %39 {value = 0 : index} : index loc(#loc82)
    %77 = handshake.constant %39 {value = 1 : index} : index loc(#loc82)
    %78 = arith.select %41, %77, %76 : index loc(#loc82)
    %79 = handshake.mux %78 [%falseResult_21, %3] : index, f32 loc(#loc82)
    %80 = dataflow.invariant %afterCond_11, %35 : i1, i32 -> i32 loc(#loc80)
    %81 = arith.addi %80, %42 : i32 loc(#loc80)
    %82 = arith.extui %81 : i32 to i64 loc(#loc80)
    %83 = arith.index_cast %82 : i64 to index loc(#loc80)
    %dataResult_34, %addressResult = handshake.store [%83] %79, %114 : index, f32 loc(#loc80)
    %84:2 = handshake.extmemory[ld = 1, st = 0] (%arg0 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults) {id = 0 : i32} : (index) -> (f32, none) loc(#loc64)
    %85:2 = handshake.extmemory[ld = 1, st = 0] (%arg1 : memref<?xf32, strided<[1], offset: ?>>) (%addressResults_33) {id = 1 : i32} : (index) -> (f32, none) loc(#loc64)
    %86 = handshake.extmemory[ld = 0, st = 1] (%arg2 : memref<?xf32, strided<[1], offset: ?>>) (%dataResult_34, %addressResult) {id = 2 : i32} : (f32, index) -> none loc(#loc64)
    %87 = dataflow.carry %willContinue, %falseResult, %trueResult_51 : i1, none, none -> none loc(#loc69)
    %trueResult_35, %falseResult_36 = handshake.cond_br %21, %87 : none loc(#loc74)
    %88 = dataflow.carry %willContinue_3, %falseResult_36, %trueResult_49 : i1, none, none -> none loc(#loc74)
    %trueResult_37, %falseResult_38 = handshake.cond_br %29, %88 : none loc(#loc77)
    %89 = dataflow.carry %willContinue_9, %falseResult_38, %trueResult_47 : i1, none, none -> none loc(#loc77)
    %trueResult_39, %falseResult_40 = handshake.cond_br %41, %89 : none loc(#loc82)
    %90 = dataflow.carry %willContinue_15, %falseResult_40, %trueResult_45 : i1, none, none -> none loc(#loc82)
    %trueResult_41, %falseResult_42 = handshake.cond_br %49, %90 : none loc(#loc86)
    %91 = dataflow.carry %willContinue_25, %falseResult_42, %trueResult_43 : i1, none, none -> none loc(#loc86)
    %trueResult_43, %falseResult_44 = handshake.cond_br %willContinue_25, %84#1 : none loc(#loc86)
    %92 = handshake.constant %90 {value = 0 : index} : index loc(#loc86)
    %93 = handshake.constant %90 {value = 1 : index} : index loc(#loc86)
    %94 = arith.select %49, %93, %92 : index loc(#loc86)
    %95 = handshake.mux %94 [%falseResult_44, %trueResult_41] : index, none loc(#loc86)
    %trueResult_45, %falseResult_46 = handshake.cond_br %willContinue_15, %95 : none loc(#loc82)
    %96 = handshake.constant %89 {value = 0 : index} : index loc(#loc82)
    %97 = handshake.constant %89 {value = 1 : index} : index loc(#loc82)
    %98 = arith.select %41, %97, %96 : index loc(#loc82)
    %99 = handshake.mux %98 [%falseResult_46, %trueResult_39] : index, none loc(#loc82)
    %trueResult_47, %falseResult_48 = handshake.cond_br %willContinue_9, %99 : none loc(#loc77)
    %100 = handshake.constant %88 {value = 0 : index} : index loc(#loc77)
    %101 = handshake.constant %88 {value = 1 : index} : index loc(#loc77)
    %102 = arith.select %29, %101, %100 : index loc(#loc77)
    %103 = handshake.mux %102 [%falseResult_48, %trueResult_37] : index, none loc(#loc77)
    %trueResult_49, %falseResult_50 = handshake.cond_br %willContinue_3, %103 : none loc(#loc74)
    %104 = handshake.constant %87 {value = 0 : index} : index loc(#loc74)
    %105 = handshake.constant %87 {value = 1 : index} : index loc(#loc74)
    %106 = arith.select %21, %105, %104 : index loc(#loc74)
    %107 = handshake.mux %106 [%falseResult_50, %trueResult_35] : index, none loc(#loc74)
    %trueResult_51, %falseResult_52 = handshake.cond_br %willContinue, %107 : none loc(#loc69)
    %108 = handshake.constant %0 {value = 0 : index} : index loc(#loc69)
    %109 = handshake.constant %0 {value = 1 : index} : index loc(#loc69)
    %110 = arith.select %11, %109, %108 : index loc(#loc69)
    %111 = handshake.mux %110 [%falseResult_52, %trueResult] : index, none loc(#loc69)
    %112 = dataflow.carry %willContinue, %falseResult, %trueResult_61 : i1, none, none -> none loc(#loc69)
    %trueResult_53, %falseResult_54 = handshake.cond_br %21, %112 : none loc(#loc74)
    %113 = dataflow.carry %willContinue_3, %falseResult_54, %trueResult_59 : i1, none, none -> none loc(#loc74)
    %trueResult_55, %falseResult_56 = handshake.cond_br %29, %113 : none loc(#loc77)
    %114 = dataflow.carry %willContinue_9, %falseResult_56, %trueResult_57 : i1, none, none -> none loc(#loc77)
    %trueResult_57, %falseResult_58 = handshake.cond_br %willContinue_9, %86 : none loc(#loc77)
    %115 = handshake.constant %113 {value = 0 : index} : index loc(#loc77)
    %116 = handshake.constant %113 {value = 1 : index} : index loc(#loc77)
    %117 = arith.select %29, %116, %115 : index loc(#loc77)
    %118 = handshake.mux %117 [%falseResult_58, %trueResult_55] : index, none loc(#loc77)
    %trueResult_59, %falseResult_60 = handshake.cond_br %willContinue_3, %118 : none loc(#loc74)
    %119 = handshake.constant %112 {value = 0 : index} : index loc(#loc74)
    %120 = handshake.constant %112 {value = 1 : index} : index loc(#loc74)
    %121 = arith.select %21, %120, %119 : index loc(#loc74)
    %122 = handshake.mux %121 [%falseResult_60, %trueResult_53] : index, none loc(#loc74)
    %trueResult_61, %falseResult_62 = handshake.cond_br %willContinue, %122 : none loc(#loc69)
    %123 = handshake.mux %110 [%falseResult_62, %trueResult] : index, none loc(#loc69)
    %124 = dataflow.carry %willContinue, %falseResult, %trueResult_79 : i1, none, none -> none loc(#loc69)
    %trueResult_63, %falseResult_64 = handshake.cond_br %21, %124 : none loc(#loc74)
    %125 = dataflow.carry %willContinue_3, %falseResult_64, %trueResult_77 : i1, none, none -> none loc(#loc74)
    %trueResult_65, %falseResult_66 = handshake.cond_br %29, %125 : none loc(#loc77)
    %126 = dataflow.carry %willContinue_9, %falseResult_66, %trueResult_75 : i1, none, none -> none loc(#loc77)
    %trueResult_67, %falseResult_68 = handshake.cond_br %41, %126 : none loc(#loc82)
    %127 = dataflow.carry %willContinue_15, %falseResult_68, %trueResult_73 : i1, none, none -> none loc(#loc82)
    %trueResult_69, %falseResult_70 = handshake.cond_br %49, %127 : none loc(#loc86)
    %128 = dataflow.carry %willContinue_25, %falseResult_70, %trueResult_71 : i1, none, none -> none loc(#loc86)
    %trueResult_71, %falseResult_72 = handshake.cond_br %willContinue_25, %85#1 : none loc(#loc86)
    %129 = handshake.constant %127 {value = 0 : index} : index loc(#loc86)
    %130 = handshake.constant %127 {value = 1 : index} : index loc(#loc86)
    %131 = arith.select %49, %130, %129 : index loc(#loc86)
    %132 = handshake.mux %131 [%falseResult_72, %trueResult_69] : index, none loc(#loc86)
    %trueResult_73, %falseResult_74 = handshake.cond_br %willContinue_15, %132 : none loc(#loc82)
    %133 = handshake.constant %126 {value = 0 : index} : index loc(#loc82)
    %134 = handshake.constant %126 {value = 1 : index} : index loc(#loc82)
    %135 = arith.select %41, %134, %133 : index loc(#loc82)
    %136 = handshake.mux %135 [%falseResult_74, %trueResult_67] : index, none loc(#loc82)
    %trueResult_75, %falseResult_76 = handshake.cond_br %willContinue_9, %136 : none loc(#loc77)
    %137 = handshake.constant %125 {value = 0 : index} : index loc(#loc77)
    %138 = handshake.constant %125 {value = 1 : index} : index loc(#loc77)
    %139 = arith.select %29, %138, %137 : index loc(#loc77)
    %140 = handshake.mux %139 [%falseResult_76, %trueResult_65] : index, none loc(#loc77)
    %trueResult_77, %falseResult_78 = handshake.cond_br %willContinue_3, %140 : none loc(#loc74)
    %141 = handshake.constant %124 {value = 0 : index} : index loc(#loc74)
    %142 = handshake.constant %124 {value = 1 : index} : index loc(#loc74)
    %143 = arith.select %21, %142, %141 : index loc(#loc74)
    %144 = handshake.mux %143 [%falseResult_78, %trueResult_63] : index, none loc(#loc74)
    %trueResult_79, %falseResult_80 = handshake.cond_br %willContinue, %144 : none loc(#loc69)
    %145 = handshake.mux %110 [%falseResult_80, %trueResult] : index, none loc(#loc69)
    %146 = handshake.join %111, %123, %145 : none, none, none loc(#loc64)
    handshake.return %146 : none loc(#loc67)
  } loc(#loc64)
  func.func private @llvm.var.annotation.p0.p0(memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, memref<?xi8, strided<[1], offset: ?>>, i32, memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func @main() -> i32 {
    %true = arith.constant true loc(#loc41)
    %false = arith.constant false loc(#loc41)
    %0 = seq.const_clock  low loc(#loc41)
    %c2_i32 = arith.constant 2 : i32 loc(#loc41)
    %1 = ub.poison : i64 loc(#loc41)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c144_i64 = arith.constant 144 : i64 loc(#loc2)
    %c0_i64 = arith.constant 0 : i64 loc(#loc2)
    %c10_i32 = arith.constant 10 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c256_i64 = arith.constant 256 : i64 loc(#loc2)
    %c5_i32 = arith.constant 5 : i32 loc(#loc2)
    %c-2_i32 = arith.constant -2 : i32 loc(#loc2)
    %cst = arith.constant 1.000000e+01 : f32 loc(#loc2)
    %c36_i64 = arith.constant 36 : i64 loc(#loc2)
    %c4_i32 = arith.constant 4 : i32 loc(#loc2)
    %c8_i32 = arith.constant 8 : i32 loc(#loc2)
    %c3_i32 = arith.constant 3 : i32 loc(#loc2)
    %cst_0 = arith.constant 9.99999974E-5 : f32 loc(#loc2)
    %2 = memref.get_global @str : memref<23xi8> loc(#loc2)
    %3 = memref.get_global @str.2 : memref<23xi8> loc(#loc2)
    %alloca = memref.alloca() : memref<256xf32> loc(#loc2)
    %alloca_1 = memref.alloca() : memref<36xf32> loc(#loc2)
    %alloca_2 = memref.alloca() : memref<144xf32> loc(#loc2)
    %alloca_3 = memref.alloca() : memref<144xf32> loc(#loc2)
    %4 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc52)
      %12 = arith.remui %11, %c10_i32 : i32 loc(#loc52)
      %13 = arith.uitofp %12 : i32 to f32 loc(#loc52)
      %14 = arith.index_cast %arg0 : i64 to index loc(#loc52)
      memref.store %13, %alloca[%14] : memref<256xf32> loc(#loc52)
      %15 = arith.addi %arg0, %c1_i64 : i64 loc(#loc49)
      %16 = arith.cmpi ne, %15, %c256_i64 : i64 loc(#loc53)
      scf.condition(%16) %15 : i64 loc(#loc46)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block15>[#loc29])):
      scf.yield %arg0 : i64 loc(#loc46)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc46)
    %5 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %11 = arith.trunci %arg0 : i64 to i32 loc(#loc54)
      %12 = arith.remui %11, %c5_i32 : i32 loc(#loc54)
      %13 = arith.addi %12, %c-2_i32 : i32 loc(#loc54)
      %14 = arith.sitofp %13 : i32 to f32 loc(#loc54)
      %15 = arith.divf %14, %cst : f32 loc(#loc54)
      %16 = arith.index_cast %arg0 : i64 to index loc(#loc54)
      memref.store %15, %alloca_1[%16] : memref<36xf32> loc(#loc54)
      %17 = arith.addi %arg0, %c1_i64 : i64 loc(#loc50)
      %18 = arith.cmpi ne, %17, %c36_i64 : i64 loc(#loc55)
      scf.condition(%18) %17 : i64 loc(#loc47)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block16>[#loc31])):
      scf.yield %arg0 : i64 loc(#loc47)
    } attributes {loom.stream = {cmp_on_update = true, iv = 0 : i64, step_op = "+=", stop_cond = "!="}} loc(#loc47)
    %cast = memref.cast %alloca : memref<256xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc42)
    %cast_4 = memref.cast %alloca_1 : memref<36xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc42)
    %cast_5 = memref.cast %alloca_2 : memref<144xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc42)
    call @_Z18depthwise_conv_cpuPKfS0_Pfjjjjjjj(%cast, %cast_4, %cast_5, %c4_i32, %c8_i32, %c8_i32, %c3_i32, %c3_i32, %c1_i32, %c1_i32) : (memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, memref<?xf32, strided<[1], offset: ?>>, i32, i32, i32, i32, i32, i32, i32) -> () loc(#loc42)
    %cast_6 = memref.cast %alloca_3 : memref<144xf32> to memref<?xf32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput, %ready = esi.wrap.vr %cast, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput_7, %ready_8 = esi.wrap.vr %cast_4, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput_9, %ready_10 = esi.wrap.vr %cast_6, %true : memref<?xf32, strided<[1], offset: ?>> loc(#loc43)
    %chanOutput_11, %ready_12 = esi.wrap.vr %c4_i32, %true : i32 loc(#loc43)
    %chanOutput_13, %ready_14 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc43)
    %chanOutput_15, %ready_16 = esi.wrap.vr %c8_i32, %true : i32 loc(#loc43)
    %chanOutput_17, %ready_18 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc43)
    %chanOutput_19, %ready_20 = esi.wrap.vr %c3_i32, %true : i32 loc(#loc43)
    %chanOutput_21, %ready_22 = esi.wrap.vr %c1_i32, %true : i32 loc(#loc43)
    %chanOutput_23, %ready_24 = esi.wrap.vr %c1_i32, %true : i32 loc(#loc43)
    %chanOutput_25, %ready_26 = esi.wrap.vr %true, %true : i1 loc(#loc43)
    %6 = handshake.esi_instance @_Z18depthwise_conv_dsaPKfS0_Pfjjjjjjj_esi "_Z18depthwise_conv_dsaPKfS0_Pfjjjjjjj_inst0" clk %0 rst %false(%chanOutput, %chanOutput_7, %chanOutput_9, %chanOutput_11, %chanOutput_13, %chanOutput_15, %chanOutput_17, %chanOutput_19, %chanOutput_21, %chanOutput_23, %chanOutput_25) : (!esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<memref<?xf32, strided<[1], offset: ?>>>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i32>, !esi.channel<i1>) -> !esi.channel<i1> loc(#loc43)
    %rawOutput, %valid = esi.unwrap.vr %6, %true : i1 loc(#loc43)
    %7:3 = scf.while (%arg0 = %c0_i64) : (i64) -> (i64, i1, i32) {
      %11 = arith.index_cast %arg0 : i64 to index loc(#loc57)
      %12 = memref.load %alloca_2[%11] : memref<144xf32> loc(#loc57)
      %13 = memref.load %alloca_3[%11] : memref<144xf32> loc(#loc57)
      %14 = arith.subf %12, %13 : f32 loc(#loc57)
      %15 = math.absf %14 : f32 loc(#loc57)
      %16 = arith.cmpf ule, %15, %cst_0 : f32 loc(#loc57)
      %17:3 = scf.if %16 -> (i64, i32, i32) {
        %19 = arith.addi %arg0, %c1_i64 : i64 loc(#loc51)
        %20 = arith.cmpi eq, %19, %c144_i64 : i64 loc(#loc51)
        %21 = arith.extui %20 : i1 to i32 loc(#loc48)
        %22 = arith.cmpi ne, %19, %c144_i64 : i64 loc(#loc56)
        %23 = arith.extui %22 : i1 to i32 loc(#loc48)
        scf.yield %19, %21, %23 : i64, i32, i32 loc(#loc57)
      } else {
        scf.yield %1, %c2_i32, %c0_i32 : i64, i32, i32 loc(#loc57)
      } loc(#loc57)
      %18 = arith.trunci %17#2 : i32 to i1 loc(#loc48)
      scf.condition(%18) %17#0, %16, %17#1 : i64, i1, i32 loc(#loc48)
    } do {
    ^bb0(%arg0: i64 loc(fused<#di_lexical_block17>[#loc35]), %arg1: i1 loc(fused<#di_lexical_block17>[#loc35]), %arg2: i32 loc(fused<#di_lexical_block17>[#loc35])):
      scf.yield %arg0 : i64 loc(#loc48)
    } loc(#loc48)
    %8 = arith.index_castui %7#2 : i32 to index loc(#loc48)
    %9 = scf.index_switch %8 -> i32 
    case 1 {
      scf.yield %c0_i32 : i32 loc(#loc48)
    }
    default {
      %intptr = memref.extract_aligned_pointer_as_index %2 : memref<23xi8> -> index loc(#loc58)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc58)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc58)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc58)
      scf.yield %c1_i32 : i32 loc(#loc59)
    } loc(#loc48)
    %10 = arith.select %7#1, %c0_i32, %9 : i32 loc(#loc2)
    scf.if %7#1 {
      %intptr = memref.extract_aligned_pointer_as_index %3 : memref<23xi8> -> index loc(#loc44)
      %11 = arith.index_cast %intptr : index to i64 loc(#loc44)
      %12 = llvm.inttoptr %11 : i64 to !llvm.ptr loc(#loc44)
      %13 = llvm.call @puts(%12) : (!llvm.ptr) -> i32 loc(#loc44)
    } loc(#loc2)
    return %10 : i32 loc(#loc45)
  } loc(#loc41)
  func.func private @llvm.lifetime.start.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
  func.func private @llvm.fabs.f32(f32) -> f32 loc(#loc2)
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree"]} loc(#loc2)
  func.func private @llvm.lifetime.end.p0(memref<?xi8, strided<[1], offset: ?>>) loc(#loc2)
} loc(#loc)
#loc = loc("tests/app/depthwise_conv/depthwise_conv.cpp":0:0)
#loc2 = loc(unknown)
#loc3 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":26:0)
#loc4 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":27:0)
#loc10 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":39:0)
#loc11 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":40:0)
#loc12 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":41:0)
#loc13 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":45:0)
#loc14 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":49:0)
#loc16 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":63:0)
#loc17 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":64:0)
#loc18 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":67:0)
#loc19 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":69:0)
#loc20 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":71:0)
#loc21 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":74:0)
#loc22 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":85:0)
#loc23 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":75:0)
#loc24 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":79:0)
#loc25 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":80:0)
#loc26 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":81:0)
#loc27 = loc("tests/app/depthwise_conv/depthwise_conv.cpp":89:0)
#loc28 = loc("tests/app/depthwise_conv/main.cpp":6:0)
#loc30 = loc("tests/app/depthwise_conv/main.cpp":29:0)
#loc32 = loc("tests/app/depthwise_conv/main.cpp":34:0)
#loc33 = loc("tests/app/depthwise_conv/main.cpp":38:0)
#loc34 = loc("tests/app/depthwise_conv/main.cpp":41:0)
#loc36 = loc("tests/app/depthwise_conv/main.cpp":45:0)
#loc37 = loc("tests/app/depthwise_conv/main.cpp":46:0)
#loc38 = loc("tests/app/depthwise_conv/main.cpp":47:0)
#loc39 = loc("tests/app/depthwise_conv/main.cpp":51:0)
#loc40 = loc("tests/app/depthwise_conv/main.cpp":53:0)
#loc41 = loc(fused<#di_subprogram3>[#loc28])
#loc42 = loc(fused<#di_subprogram3>[#loc33])
#loc43 = loc(fused<#di_subprogram3>[#loc34])
#loc44 = loc(fused<#di_subprogram3>[#loc39])
#loc45 = loc(fused<#di_subprogram3>[#loc40])
#di_lexical_block20 = #llvm.di_lexical_block<scope = #di_lexical_block15, file = #di_file1, line = 28>
#di_lexical_block21 = #llvm.di_lexical_block<scope = #di_lexical_block16, file = #di_file1, line = 33>
#di_lexical_block22 = #llvm.di_lexical_block<scope = #di_lexical_block17, file = #di_file1, line = 44>
#di_lexical_block25 = #llvm.di_lexical_block<scope = #di_lexical_block20, file = #di_file1, line = 28>
#di_lexical_block26 = #llvm.di_lexical_block<scope = #di_lexical_block21, file = #di_file1, line = 33>
#di_lexical_block27 = #llvm.di_lexical_block<scope = #di_lexical_block22, file = #di_file1, line = 44>
#loc49 = loc(fused<#di_lexical_block20>[#loc29])
#loc50 = loc(fused<#di_lexical_block21>[#loc31])
#loc51 = loc(fused<#di_lexical_block22>[#loc35])
#di_lexical_block30 = #llvm.di_lexical_block<scope = #di_lexical_block27, file = #di_file1, line = 45>
#loc52 = loc(fused<#di_lexical_block25>[#loc30])
#loc53 = loc(fused[#loc46, #loc49])
#loc54 = loc(fused<#di_lexical_block26>[#loc32])
#loc55 = loc(fused[#loc47, #loc50])
#loc56 = loc(fused[#loc48, #loc51])
#di_lexical_block33 = #llvm.di_lexical_block<scope = #di_lexical_block30, file = #di_file1, line = 45>
#loc57 = loc(fused<#di_lexical_block30>[#loc36])
#loc58 = loc(fused<#di_lexical_block33>[#loc37])
#loc59 = loc(fused<#di_lexical_block33>[#loc38])
#di_lexical_block45 = #llvm.di_lexical_block<scope = #di_subprogram5, file = #di_file, line = 67>
#loc61 = loc(fused<#di_subprogram4>[#loc3])
#loc62 = loc(fused<#di_subprogram4>[#loc4])
#loc63 = loc(fused<#di_subprogram4>[#loc14])
#loc65 = loc(fused<#di_subprogram5>[#loc16])
#loc66 = loc(fused<#di_subprogram5>[#loc17])
#loc67 = loc(fused<#di_subprogram5>[#loc27])
#di_lexical_block47 = #llvm.di_lexical_block<scope = #di_lexical_block45, file = #di_file, line = 67>
#loc69 = loc(fused<#di_lexical_block45>[#loc18])
#di_lexical_block49 = #llvm.di_lexical_block<scope = #di_lexical_block47, file = #di_file, line = 67>
#loc70 = loc(fused<#di_lexical_block46>[#loc5])
#loc71 = loc(fused<#di_lexical_block47>[#loc18])
#di_lexical_block51 = #llvm.di_lexical_block<scope = #di_lexical_block49, file = #di_file, line = 69>
#loc72 = loc(fused[#loc68, #loc70])
#di_lexical_block53 = #llvm.di_lexical_block<scope = #di_lexical_block51, file = #di_file, line = 69>
#loc74 = loc(fused<#di_lexical_block51>[#loc19])
#di_lexical_block55 = #llvm.di_lexical_block<scope = #di_lexical_block53, file = #di_file, line = 69>
#loc75 = loc(fused<#di_lexical_block52>[#loc6])
#di_lexical_block57 = #llvm.di_lexical_block<scope = #di_lexical_block55, file = #di_file, line = 71>
#di_lexical_block59 = #llvm.di_lexical_block<scope = #di_lexical_block57, file = #di_file, line = 71>
#loc77 = loc(fused<#di_lexical_block57>[#loc20])
#di_lexical_block61 = #llvm.di_lexical_block<scope = #di_lexical_block59, file = #di_file, line = 71>
#loc78 = loc(fused<#di_lexical_block58>[#loc7])
#di_lexical_block63 = #llvm.di_lexical_block<scope = #di_lexical_block61, file = #di_file, line = 74>
#loc79 = loc(fused<#di_lexical_block60>[#loc13])
#loc80 = loc(fused<#di_lexical_block61>[#loc22])
#di_lexical_block65 = #llvm.di_lexical_block<scope = #di_lexical_block63, file = #di_file, line = 74>
#loc82 = loc(fused<#di_lexical_block63>[#loc21])
#di_lexical_block67 = #llvm.di_lexical_block<scope = #di_lexical_block65, file = #di_file, line = 74>
#loc83 = loc(fused<#di_lexical_block64>[#loc8])
#di_lexical_block69 = #llvm.di_lexical_block<scope = #di_lexical_block67, file = #di_file, line = 75>
#loc84 = loc(fused[#loc81, #loc83])
#di_lexical_block70 = #llvm.di_lexical_block<scope = #di_lexical_block68, file = #di_file, line = 35>
#di_lexical_block71 = #llvm.di_lexical_block<scope = #di_lexical_block69, file = #di_file, line = 75>
#loc86 = loc(fused<#di_lexical_block69>[#loc23])
#di_lexical_block72 = #llvm.di_lexical_block<scope = #di_lexical_block70, file = #di_file, line = 35>
#di_lexical_block73 = #llvm.di_lexical_block<scope = #di_lexical_block71, file = #di_file, line = 75>
#loc87 = loc(fused<#di_lexical_block70>[#loc9])
#loc88 = loc(fused<#di_lexical_block72>[#loc10])
#loc89 = loc(fused<#di_lexical_block72>[#loc11])
#loc90 = loc(fused<#di_lexical_block72>[#loc12])
#loc91 = loc(fused[#loc85, #loc87])
#loc92 = loc(fused<#di_lexical_block73>[#loc24])
#loc93 = loc(fused<#di_lexical_block73>[#loc25])
#loc94 = loc(fused<#di_lexical_block73>[#loc26])
