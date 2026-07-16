"""Classifier: a predicted class + score per object. One mode: each input Object points at its own
crop (``object.imagery`` -> Crops), the classifier reads the crops and carries the objects forward
with their class (``prev_object_id`` -> the input object), preserving geometry and lineage.

The '1 image = 1 class' case (a seeded crops folder, no detections) is not a second mode: the
pipeline derives one Object per crop at construction (``Objects.from_imagery``), so the classifier
always sees Objects-on-Crops.
"""

from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY
from canopyrs.engine.constants import Col
from canopyrs.engine.data import Crops, Objects
from canopyrs.engine.contracts import Need
from canopyrs.engine.components.base import Component, register_component


@register_component("classifier")
class Classifier(Component):
    def __init__(self, config):
        super().__init__(config)
        self._model_class = self._model(CLASSIFIER_REGISTRY)
        # Objects each pointing at their own crop — never whole tiles (several objects would silently
        # share one class); a polygon tilerizer (or the pipeline's crop-seed derivation) makes these.
        self.requires = (Need(Objects, links=("imagery",), on=Crops),)
        columns = [Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORE, Col.CLASSIFIER_SCORES]
        if config.class_names:
            columns.append(Col.CLASSIFIER_CLASS_NAME)
        # Adds the class columns; geometry and links are inherited through the lineage.
        self.produces = Need(Objects, columns=tuple(columns), links=("prev_objects",), on=Crops)

    def run(self, objects: Objects) -> Objects:
        classifier = self._model_class(self.config)
        crops = objects.linked("imagery")   # the per-object crops (one crop per object)
        loader = self._loader(crops, batch_size=self.config.batch_size)
        image_ids, predictions, class_scores = classifier.infer(loader)

        rows = objects.df.set_index(Col.IMAGE_ID).loc[image_ids]   # one crop per object, in loader order
        out = Objects.build(geometry=rows[Col.GEOMETRY].values,
                            geom_kind=rows[Col.GEOM_KIND].values,
                            prev_object_id=rows[Col.OBJECT_ID].values,
                            crs=objects.df.crs, prev_objects=objects,
                            **self._class_columns(predictions, class_scores))
        print(f"Classifier: classified {len(out)} objects.")
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
