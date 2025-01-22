import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
import torchmetrics
from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset
from huggingface_hub import hf_hub_download
from shapely import box
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
import wandb

from engine.models.utils import WarmupStepLR


class DetectorWrapperBase(ABC):
    def __init__(self, config, ):
        self.config = config

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = None

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            # backend='faster_coco_eval',   # Requires additional dependencies
            iou_type="bbox",
            # max_detection_thresholds=[1, 10, self.box_predictions_per_image]
        ).to(self.device)

    def load_checkpoint(self, checkpoint_state_dict_path):
        checkpoint_state_dict_path = Path(checkpoint_state_dict_path)
        if 'huggingface.co' in checkpoint_state_dict_path.parts:
            if "huggingface.co" not in checkpoint_state_dict_path.as_posix():
                raise ValueError("The provided Path does not contain a valid Hugging Face URL.")
            # Remove the "https://huggingface.co/" part
            path = Path(str(checkpoint_state_dict_path).split("huggingface.co/")[-1])
            if "resolve" not in path.parts:
                raise ValueError("The provided Path is not in the expected Hugging Face format.")
            # Extract repo_id and filename
            repo_id = "/".join(path.parts[:2])
            filename = path.name
            checkpoint_state_dict_path = hf_hub_download(repo_id=repo_id, filename=filename)

        if checkpoint_state_dict_path:
            try:
                self.model.load_state_dict(torch.load(checkpoint_state_dict_path, weights_only=False))
            except RuntimeError as e:
                state_dict = try_rename_state_dict_keys_with_model(checkpoint_state_dict_path)
                self.model.load_state_dict(state_dict)

    @abstractmethod
    def forward(self, images, targets=None):
        pass

    def _evaluate(self, data_loader, epoch=None):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc=f"Epoch {epoch + 1} (scoring)" if epoch is not None else "Scoring",
                                             leave=True)
            for batch_idx, (images, targets) in enumerate(data_loader_with_progress):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.forward(images, targets)
                predictions.extend(outputs)

                # Update MAP metric for validation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            message=f".*{'Encountered more than 100 detections in a single image'}.*")
                    self.map_metric.update(outputs, targets)

        # Compute and log MAP metric
        scores = self.map_metric.compute()
        self.map_metric.reset()  # Reset metric for next epoch/validation

        return scores, predictions

    def score(self, test_ds: DetectionLabeledRasterCocoDataset, collate_fn: callable):
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=3, persistent_workers=True)

        scores, predictions = self._evaluate(test_dl)
        print(f"Score results: {scores}")
        boxes, boxes_scores = detector_result_to_lists(predictions)
        # Map tile paths to their corresponding raster names
        # it's important to get the paths sorted by ids as the associated predictions will also be sorted by those ids.
        tiles_paths = [value["path"] for key, value in sorted(test_ds.tiles.items(), key=lambda item: item[0])]
        return tiles_paths, boxes, boxes_scores, scores

    def _save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def _train_one_epoch(self, optimizer, data_loader, epoch):
        self.model.train()
        accumulated_loss = 0.0
        data_loader_with_progress = tqdm(data_loader, desc=f"Epoch {epoch + 1} (training)", leave=True)
        for batch_idx, (images, targets) in enumerate(data_loader_with_progress):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.forward(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss = loss / self.config.grad_accumulation_steps

            loss.backward()
            accumulated_loss += loss.detach().item()

            # Perform optimization step only after accumulating enough gradients
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logging at intervals of train_log_interval effective batches
            if (batch_idx + 1) % (self.config.train_log_interval * self.config.grad_accumulation_steps) == 0:
                # average_loss = accumulated_loss / self.config.train_log_interval
                # self.writer.add_scalar('Loss/train', average_loss, self.config.base_params_config.batch_size * (epoch * len(data_loader) + batch_idx))
                accumulated_loss = 0.0

        # Ensure any remaining gradients are applied
        if (batch_idx + 1) % self.config.grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    def train(self,
              train_ds: DetectionLabeledRasterCocoDataset,
              valid_ds: DetectionLabeledRasterCocoDataset,
              collate_fn: callable):
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = optim.SGD(
            params,
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        scheduler = WarmupStepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma,
            warmup_steps=self.config.scheduler_warmup_steps,
            base_lr=self.config.learning_rate / 100
        )

        train_dl = DataLoader(train_ds, batch_size=self.config.base_params_config.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)
        valid_dl = DataLoader(valid_ds, batch_size=self.config.base_params_config.batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        print(f"Training for {self.config.n_epochs} epochs...")
        print(f'Effective batch size'
              f' = batch_size * grad_accumulation_steps'
              f' = {self.config.base_params_config.batch_size} * {self.config.grad_accumulation_steps}'
              f' = {self.config.base_params_config.batch_size * self.config.grad_accumulation_steps}')

        for epoch in range(self.config.n_epochs):
            # also log the current learning rate
            # self.writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

            self._train_one_epoch(optimizer, train_dl, epoch=epoch)
            scores, predictions = self._evaluate(valid_dl, epoch=epoch)
            # self.writer.add_scalar('metric/map', scores['map'], epoch)
            # self.writer.add_scalar('metric/map_50', scores['map_50'], epoch)
            # self.writer.add_scalar('metric/map_75', scores['map_75'], epoch)

            scheduler.step()

            if epoch % self.config.save_model_every_n_epoch == 0:
                self._save_model(save_path=self.model_output_folder / f"{self.config.output_name}_{epoch}.pt")

    @staticmethod
    def get_data_augmentation_transform():
        data_augmentation_transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ShiftScaleRotate(p=0.2),        # this can put boxes out of the image and then the training crashes on an image without boxes

            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(hue_shift_limit=10, p=0.1),
            A.RGBShift(p=0.1),
            A.RandomGamma(p=0.1),
            A.Blur(p=0.1),
            # A.ToGray(p=0.02),     # probably not a good idea as it would start detecting trees in water etc
            # A.ToSepia(p=0.02),     # probably not a good idea as it would start detecting trees in water etc
        ],
            bbox_params=A.BboxParams(
                format='pascal_voc',  # Specify the format of your bounding boxes
                label_fields=['labels'],  # Specify the field that contains the labels
                min_area=0.,
                # Minimum area of a bounding box. All bboxes that have an area smaller than this value will be removed
                min_visibility=0.,
                # Minimum visibility of a bounding box. All bboxes that have a visibility smaller than this value will be removed
            ))
        return data_augmentation_transform

    def _infer(self, data_loader):
        self.model.eval()

        predictions = []

        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc="Inferring detector...",
                                             leave=True)
            for images in data_loader_with_progress:
                images = list(img.to(self.device) for img in images)
                outputs = self.forward(images)
                predictions.extend(outputs)

        return predictions

    def infer(self, infer_ds: UnlabeledRasterDataset, collate_fn: callable):
        infer_dl = DataLoader(infer_ds, batch_size=self.config.batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        results = self._infer(infer_dl)
        boxes, boxes_scores, classes = detector_result_to_lists(results)
        tiles_paths = infer_ds.tile_paths
        return tiles_paths, boxes, boxes_scores, classes


def try_rename_state_dict_keys_with_model(checkpoint_state_dict_path):
    # Structure the OrderedDict keys to match requirements
    checkpoint = torch.load(checkpoint_state_dict_path, weights_only=True)
    # Create a new OrderedDict with the keys prefixed with "model."
    new_state_dict = OrderedDict()
    if all(s.startswith("model.") for s in checkpoint.keys()):
        # try removing the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = key[6:]
            new_state_dict[new_key] = value
    else:
        # try adding the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = 'model.' + key  # Prefix "model." to each key
            new_state_dict[new_key] = value
    return new_state_dict

def detector_result_to_lists(detector_result):
    detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
    for x in detector_result:
        x['boxes'] = [box(*b) for b in x['boxes']]
        x['scores'] = x['scores'].tolist()
        x['classes'] = x['labels'].tolist()
    boxes = [x['boxes'] for x in detector_result]
    scores = [x['scores'] for x in detector_result]
    classes = [x['classes'] for x in detector_result]

    return boxes, scores, classes
