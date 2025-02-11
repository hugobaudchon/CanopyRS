from engine.models.detector.detectron2_infer import Detectron2PredictorWrapper
from engine.models.detector.pt_faster_rcnn import FasterRCNNWrapper
from engine.models.detector.pt_retina_net import RetinaNetWrapper
# from engine.models.segmenter.detectree2 import Detectree2TracedWrapper
# from engine.models.segmenter.oam_tcd_torchscript import OamTcdTorchScriptWrapper
from engine.models.segmenter.sam import SamPredictorWrapper
from engine.models.segmenter.sam2 import Sam2PredictorWrapper


DETECTOR_REGISTRY = {
    'faster_rcnn': FasterRCNNWrapper,
    'retina_net': RetinaNetWrapper,
    'dino_detrex': Detectron2PredictorWrapper,
    'faster_rcnn_detectron2': Detectron2PredictorWrapper,
}


SEGMENTER_REGISTRY = {
    'sam': SamPredictorWrapper,
    'sam2': Sam2PredictorWrapper,
    # 'oam-tcd': OamTcdTorchScriptWrapper,
    # 'detectree2': Detectree2TracedWrapper
}


EMBEDDER_REGISTRY = {}
CLASSIFIER_REGISTRY = {}
