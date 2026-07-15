"""Classifier: a predicted class + score per object. Two input modes, picked by what the pipeline has
available (Objects preferred):

  - **on objects** (the pipeline case): per-object crop Objects from the polygon tilerizer, each
    already pointing at its crop (``object.imagery``). Reads each crop and carries the object forward
    with its class (``prev_object_id`` -> the input object), preserving geometry and lineage.
  - **on tiles** (standalone): a tile Imagery table read directly — one classification per image.
    Emits one Object per image, geometry = the image footprint, pointing at its image.

Either way it just adds the classifier columns; geometry/links are inherited from whatever it consumed.
"""

from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY
from canopyrs.engine.constants import Col, GeomKind, ImageKind
from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.contracts import Need, one_of
from canopyrs.engine.components.base import Component, register_component
from canopyrs.engine.tilemeta import box_of


@register_component("classifier")
class Classifier(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(CLASSIFIER_REGISTRY)
        # Prefer per-object crops (Objects that point at their image); else classify whole tiles.
        self.requires = (one_of(
            Need(Objects, links=("imagery",)),
            Need(Imagery, kind=ImageKind.TILE),
        ),)
        columns = [Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORE, Col.CLASSIFIER_SCORES]
        if config.class_names:
            columns.append(Col.CLASSIFIER_CLASS_NAME)
        self.produces = Need(Objects, columns=tuple(columns))   # adds class columns; geometry/links inherited

    def run(self, data) -> Objects:
        classifier = self._model_class(self.config)
        return self._on_objects(data, classifier) if isinstance(data, Objects) else self._on_tiles(data, classifier)

    def _on_objects(self, objects: Objects, classifier) -> Objects:
        crops = objects.linked("imagery")   # the per-object crops (one crop per object)
        loader = self._loader(crops, batch_size=self.config.batch_size)
        image_ids, predictions, class_scores = classifier.infer_v2(loader)

        by_image = objects.df.set_index(Col.IMAGE_ID)   # one crop per object
        geometry, geom_kind, prev_object_ids = [], [], []
        for image_id in image_ids:
            object_row = by_image.loc[image_id]
            geometry.append(object_row[Col.GEOMETRY])
            geom_kind.append(object_row[Col.GEOM_KIND])
            prev_object_ids.append(object_row[Col.OBJECT_ID])
        out = Objects.build(geometry=geometry, geom_kind=geom_kind, prev_object_id=prev_object_ids,
                            crs=objects.df.crs, prev_objects=objects,
                            **self._class_columns(predictions, class_scores))
        print(f"Classifier: classified {len(out)} objects.")
        return out

    def _on_tiles(self, tiles: Imagery, classifier) -> Objects:
        loader = self._loader(tiles, batch_size=self.config.batch_size)
        image_ids, predictions, class_scores = classifier.infer_v2(loader)

        meta_by_image = tiles.df.set_index(Col.IMAGE_ID)[Col.METADATA]
        crs = tiles.df[Col.METADATA].iloc[0]["crs"] if len(tiles) else None
        geometry = [box_of(meta_by_image[image_id]) for image_id in image_ids]   # each image's footprint
        out = Objects.build(geometry=geometry, geom_kind=GeomKind.BOX, image_id=list(image_ids),
                            imagery=tiles, crs=crs,
                            **self._class_columns(predictions, class_scores))
        print(f"Classifier: classified {len(out)} tiles.")
        return out

    def _class_columns(self, predictions, class_scores) -> dict:
        """The classifier output columns from aligned per-item predictions + score vectors."""
        names = self.config.class_names
        classes, top_scores, all_scores, class_names = [], [], [], []
        for prediction, scores in zip(predictions, class_scores):
            classes.append(prediction)
            top_scores.append(scores[prediction] if prediction is not None else None)
            all_scores.append(scores)
            if names:
                class_names.append(names[prediction] if (prediction is not None and 0 <= prediction < len(names)) else None)
        columns = {Col.CLASSIFIER_CLASS: classes, Col.CLASSIFIER_SCORE: top_scores, Col.CLASSIFIER_SCORES: all_scores}
        if names:
            columns[Col.CLASSIFIER_CLASS_NAME] = class_names
        return columns
