from engine.models.detector.faster_rcnn import FasterRCNNWrapper
from engine.models.detector.retina_net import RetinaNetWrapper
# from engine.models.segmenter.detectree2 import Detectree2TracedWrapper
# from engine.models.segmenter.oam_tcd_torchscript import OamTcdTorchScriptWrapper
from engine.models.segmenter.sam import SamPredictorWrapper
from engine.models.segmenter.sam2 import Sam2PredictorWrapper


DETECTOR_REGISTRY = {
    'faster_rcnn': FasterRCNNWrapper,
    'retina_net': RetinaNetWrapper
}


SEGMENTER_REGISTRY = {
    'sam': SamPredictorWrapper,
    'sam2': Sam2PredictorWrapper,
    # 'oam-tcd': OamTcdTorchScriptWrapper,
    # 'detectree2': Detectree2TracedWrapper
}


EMBEDDER_REGISTRY = {}
CLASSIFIER_REGISTRY = {}
