from engine.models.detector.deepforest_infer import DeepForestWrapper
from engine.models.detector.detectron2_infer import Detectron2DetectorWrapper
from engine.models.detector.pt_faster_rcnn import FasterRCNNWrapper
from engine.models.detector.pt_retina_net import RetinaNetWrapper

from engine.models.segmenter.sam import SamPredictorWrapper
from engine.models.segmenter.sam2 import Sam2PredictorWrapper
from engine.models.segmenter.detectron2_infer import Detectron2SegmenterWrapper

from engine.models.classifier.resnet_classifier import ResNetClassifier
from engine.models.classifier.swin_classifier import SwinClassifier


DETECTOR_REGISTRY = {
    'faster_rcnn': FasterRCNNWrapper,
    'retina_net': RetinaNetWrapper,
    'dino_detrex': Detectron2DetectorWrapper,
    'faster_rcnn_detectron2': Detectron2DetectorWrapper,
    'detectree2': Detectron2DetectorWrapper,
    'deepforest': DeepForestWrapper,
}


SEGMENTER_REGISTRY = {
    'sam': SamPredictorWrapper,
    'sam2': Sam2PredictorWrapper,
    'detectree2': Detectron2SegmenterWrapper
}


EMBEDDER_REGISTRY = {}
CLASSIFIER_REGISTRY = {
    'resnet': ResNetClassifier,
    'swin': SwinClassifier
}
