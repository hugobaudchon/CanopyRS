# Data model

The pipeline threads two typed tables between components. Each is a (Geo)DataFrame plus a primary key,
with foreign-key columns linking rows to their parents: `Imagery` forms a containment tree
(`parent_id` — a tile in its raster, a crop in its tile), `Objects` a lineage chain
(`prev_object_id`), and each object points at the image it was found in (`image_id`).

::: canopyrs.engine.data.Imagery

::: canopyrs.engine.data.Objects
