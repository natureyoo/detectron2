# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm, Linear, AvgPool2d
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
# from detectron2.modeling.sampling import TripletSelector
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from ..backbone.resnet import BottleneckBlock, make_stage
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals

SIM_NET_REGISTRY = Registry("SIM_NET")
SIM_NET_REGISTRY.__doc__ = """
Registry for sim head, which retrieves same masks as anchor masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_sim_net(cfg, input_shape):
    """
    Build a similarity network defined by `cfg.MODEL.SIM_NET.NAME`.
    """
    name = cfg.MODEL.SIM_NET.NAME
    return SIM_NET_REGISTRY.get(name)(cfg, input_shape)


class TripletSelector:
    def __init__(self, max=None):
        self.max = max

    def get_triplet(self, indices):
        style_indices = [ind.style.cpu().numpy() for ind in indices]
        class_indices = [ind.gt_classes.cpu().numpy() for ind in indices]

        triplet_indices = np.array([])
        for style in set(style_indices[0]):
            anchor_idx = np.where(style_indices[0] == style)[0]
            positive_idx = np.where(style_indices[1] == style)[0]
            negative_idx = np.where(class_indices[2] == class_indices[0][anchor_idx[0]])[0]     # have same category
            if positive_idx.size == 0 or negative_idx.size == 0:
                continue
            np.random.shuffle(positive_idx)
            np.random.shuffle(negative_idx)
            positive_idx = np.concatenate([positive_idx for _ in range(anchor_idx.size // positive_idx.size + 1)])
            positive_idx = positive_idx[:anchor_idx.size] + len(style_indices[0])
            negative_idx = np.concatenate([negative_idx for _ in range(anchor_idx.size // negative_idx.size + 1)])
            negative_idx = negative_idx[:anchor_idx.size] + len(style_indices[0]) + len(style_indices[1])
            cur_indices = np.hstack((anchor_idx.reshape(-1, 1), positive_idx.reshape(-1, 1), negative_idx.reshape(-1, 1)))

            if triplet_indices.shape[0] == 0:
                triplet_indices = cur_indices
            else:
                triplet_indices = np.vstack((triplet_indices, cur_indices))

        if self.max is not None:
            np.random.shuffle(triplet_indices)
            triplet_indices = triplet_indices[:self.max]

        return triplet_indices


def triplet_loss(embedding_vecs, indices, sampler, margin=1.0):
    triplet_pairs = sampler.get_triplet(indices)
    if triplet_pairs.shape[0] == 0:
        return 0
    ap_distance = (embedding_vecs[triplet_pairs[:, 0]] - embedding_vecs[triplet_pairs[:, 1]]).pow(2).sum(1).pow(0.5)
    an_distance = (embedding_vecs[triplet_pairs[:, 0]] - embedding_vecs[triplet_pairs[:, 2]]).pow(2).sum(1).pow(0.5)
    loss = F.relu(ap_distance - an_distance + margin)

    return loss.mean()


def ce_loss(embedded_vecs, instances, vis_period=0, margin=1.0):
    """
    compare the three embedded vectors (anchor, positive, negative)

    Args:
        embedded_vecs (Tensor): A tensor of shape (B, V) where B is the total number of predicted masks in all images,
            V is the dimension of embedding vectors.
        instances (list[Instances]): A list of N Instances, where N is the number of images in the batch.
            These instances are in 1:1 correspondence with the embedded_vecs. The ground-truth labels (class, box, mask,
            pair_id, style, ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization
        margin (float): margin for Soft Triplet Loss

    Returns:
        cross entropy loss (Tensor): A scalar tensor containing the loss
    """
    return 0


def sim_inference(pred_mask_logits, pred_instances):
    pass


class SimHead(nn.Module):
    """
    A match head with several conv layers, plus an average pooling layer and fc layer.
    The output is the vectors to be compared with other objects
    """

    def __init__(self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="",
                 vis_period=0, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        self.vis_period = vis_period

        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.pool = AvgPool2d((self._output_size[1], self._output_size[2]))
        self._output_size = self._output_size[0]

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = self.pool(x)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))

        return x


@SIM_NET_REGISTRY.register()
class SimNet(nn.Module):
    """
    Match Network in paper 'DeepFashion2'
    """
    @configurable
    def __init__(self, input_shape: ShapeSpec, *, in_features, conv_dims: List[int], fc_dims: List[int], last_dim: int,
                 conv_norm="", sim_pooler, vis_period=0, proposal_matcher, proposal_append_gt=True, margin, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            fc_dims (list[int]): a list of fc dimensions
            last_dim : layer num of last layer
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            
        """
        super().__init__()
        self._output_size = 2
        self.in_features=in_features
        self.proposal_append_gt = proposal_append_gt
        self.proposal_matcher = proposal_matcher
        self.sim_pooler = sim_pooler
        self.sim_head = SimHead(input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, vis_period=vis_period, **kwargs)
        self.final_fc = Linear(self._output_size, last_dim)
        weight_init.c2_xavier_fill(self.final_fc)

        self.sampler = TripletSelector()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {"vis_period": cfg.VIS_PERIOD}
        in_features = cfg.MODEL.SIM_NET.IN_FEATURES
        pooler_resolution = cfg.MODEL.SIM_NET.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.SIM_NET.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.SIM_NET.POOLER_TYPE
        conv_dim = cfg.MODEL.SIM_NET.CONV_DIM
        num_conv = cfg.MODEL.SIM_NET.NUM_CONV
        fc_dim = cfg.MODEL.SIM_NET.FC_DIM
        num_fc = cfg.MODEL.SIM_NET.NUM_FC
        last_dim = cfg.MODEL.SIM_NET.LAST_DIM
        margin = cfg.MODEL.SIM_NET.MARGIN

        proposal_matcher = Matcher(
            cfg.MODEL.SIM_NET.IOU_THRESHOLDS,
            cfg.MODEL.SIM_NET.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        proposal_append_gt = cfg.MODEL.SIM_NET.PROPOSAL_APPEND_GT

        in_channels = [input_shape[f].channels for f in in_features][0]
        input_shape = ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)

        ret["sim_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        ret.update(
            in_features=in_features,
            conv_dims=[conv_dim] * num_conv,
            conv_norm=cfg.MODEL.SIM_NET.NORM,
            input_shape=input_shape,
            fc_dims=[fc_dim] * num_fc,
            last_dim=last_dim,
            proposal_matcher=proposal_matcher,
            proposal_append_gt=proposal_append_gt,
            margin=margin,
        )
        return ret

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor, style: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:     # matched_labels == 1, style > 0, class별로 최대한 evenly-distributed
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            style = style[matched_idxs]
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        sampled_idxs = torch.nonzero((style > 0) & (matched_labels == 1), as_tuple=True)
        return sampled_idxs, gt_classes[sampled_idxs], style[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth boxes,
         with evenly-distributed for classes

        Args:
            See :meth:`SIMHead.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
            # gt bbox를 proposal에 추가, RPN에서 내보내는 proposal은 1000개, 여기에 gt bbox 2개 추가하여 proposals 는 1002개 됨
        proposals_with_gt = []
        num_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets): # targets[0] 은 한 이미지 내의 여러 object에 대함
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )   # 실제 image에 N개의 obj있고, M개의 proposal있다면, matrix shape은 N*M
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes, styles = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes, targets_per_image.style
            )   # matched_labels == 1, style > 0, class별로 최대한 evenly-distributed
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if (trg_name.startswith("gt_") or trg_name in ['pair_id', 'style'])\
                            and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                        # 'gt_boxes', 'gt_masks', 'pair_id', 'style' 을 proposal에 넣음
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            proposals_with_gt.append(proposals_per_image)
            num_samples.append(len(gt_classes))
        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("sim_head/num_samples", np.mean(num_samples))

        return proposals_with_gt    # style > 0 인 애들에 대한 proposals를 return

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None):
        del images
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            proposal_boxes = [x.proposal_boxes for x in proposals]
        else:
            proposal_boxes = self.pred_boxes(proposals)
        features = self.sim_pooler(features, proposal_boxes)
        if features.shape[0] == 0:
            return None, None
        x = self.sim_head(features)
        if self.training:
            loss = triplet_loss(x, proposals, self.sampler)
            return {'loss_sim': loss}
        else:
            return proposals, x

    def pred_boxes(self, proposals):
        result = []
        for p in proposals:
            class_set = set(p.pred_classes[p.scores >= 0.5].cpu().numpy())
            p.pred_idx = torch.zeros(len(p), dtype=torch.bool)
            for c in class_set:
                p.pred_idx[np.where(p.pred_classes.cpu().numpy() == c)[0][0]] = True
            result.append(p.pred_boxes[p.pred_idx])
       
        return result
