# LOOM Tag Boundary Specification

## Overview

LOOM uses three inline-only tag-boundary operations:

- `fabric.add_tag`
- `fabric.map_tag`
- `fabric.del_tag`

They may appear directly only inside `fabric.module`.

## `fabric.add_tag`

`fabric.add_tag` attaches one constant runtime tag value to an untagged value.

- hardware structure is carried by the result tagged type
- runtime configuration is the attached `tag`

Config-memory contribution:

- one runtime field storing the configured tag value

## `fabric.del_tag`

`fabric.del_tag` removes the tag field and forwards the value payload
unchanged.

`fabric.del_tag` contributes no runtime configuration bits.

## `fabric.map_tag`

`fabric.map_tag` preserves the value payload type and rewrites the runtime tag
according to one configurable lookup table.

Hardware parameters in `[]`:

- `table_size`

Runtime configuration in `attributes {}`:

- `table`

`table` is an array of `table_size` entries. Each entry is a triple:

- `valid`
- `src_tag`
- `dst_tag`

Canonical textual example:

```mlir
%retagged = fabric.map_tag %tagged
    [table_size = 2 : i64]
    attributes {table = [[1 : i1, 0 : i2, 3 : i3],
                         [1 : i1, 1 : i2, 2 : i3]]}
    : !fabric.tagged<!fabric.bits<32>, i2>
      -> !fabric.tagged<!fabric.bits<32>, i3>
```

Machine layout for one entry is low-to-high:

- `valid`
- `src_tag`
- `dst_tag`

`fabric.map_tag` may change tag width, but it may not change the value payload
type.
