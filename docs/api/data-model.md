# Data model

The pipeline threads typed tables between components. Each is a (Geo)DataFrame plus a primary key,
with foreign-key columns linking rows to their parents: the three imagery roles — `Sources`, `Tiles`,
`Crops`, sharing one `Imagery` base — form a containment tree (`parent_id` — a tile in its raster, a
crop in its tile), `Objects` a lineage chain (`prev_object_id`), and each object points at the image
it was found in (`image_id`).

::: canopyrs.engine.data.Sources

::: canopyrs.engine.data.Tiles

::: canopyrs.engine.data.Crops

::: canopyrs.engine.data.Imagery

::: canopyrs.engine.data.Objects
