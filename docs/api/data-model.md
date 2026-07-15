# Data model

The pipeline threads three typed tables between components. Each is a (Geo)DataFrame plus a primary key, with foreign-key columns linking it to its parents (`source_id`, `tile_id`, `prev_object_id`).

::: canopyrs.engine.data.Sources

::: canopyrs.engine.data.Tiles

::: canopyrs.engine.data.Objects
