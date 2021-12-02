# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.utils import get_root_logger
import torch.nn as nn

from .untils import tokenize

@DETECTORS.register_module()
class DenseCLIP_RetinaNet(SingleStageDetector):
    """
    DenseCLIP for RetinaNet
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 context_length,
                 class_names,
                 tau=0.07,
                 token_embed_dim=512, 
                 text_dim=1024, 
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 seg_loss=False,
                 clip_head=True,
                 init_cfg=None):
        # super().__init__(init_cfg)
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)
        self.context_length = context_length

        self.tau = tau

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.use_seg_loss = seg_loss
        self.class_names = class_names
        self.clip_head = clip_head

        # learnable textual contexts
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    def extract_feat(self, img, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        text_features = self.compute_text_features(x, dummy=dummy)
        score_maps = self.compute_score_maps(x, text_features)
        x = list(x[:-1])
        x[3] = torch.cat([x[3], score_maps[3]], dim=1)
        if self.with_neck:
            x = self.neck(x)
        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x

    def compute_score_maps(self, x, text_features):
        # B, K, C
        _, visual_embeddings = x[4]
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone, 
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.class_names), C, device=global_feat.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff
        return text_embeddings

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img, dummy=True)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img, self.use_seg_loss)
        if self.use_seg_loss:
            x, score_map = x
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.use_seg_loss:
            losses['loss_seg'] = self.compute_seg_loss(img, score_map, img_metas, gt_bboxes, gt_labels)
        return losses

    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_labels):
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_labels)
        loss = F.binary_cross_entropy(F.sigmoid(score_map), target, weight=mask, reduction='sum')
        loss = loss / mask.sum()
        return loss

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.class_names), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.class_names), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.
        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.
        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels



@DETECTORS.register_module()
class DenseCLIP_MaskRCNN(TwoStageDetector):
    """
    DenseCLIP for Mask-RCNN
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 context_length,
                 class_names,
                 seg_loss=False,
                 clip_head=True,
                 text_adapter=None,
                 tau=0.07,
                 token_embed_dim=512, 
                 text_dim=1024,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.context_length = context_length
        self.tau = tau
        # add a [background] class
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.use_seg_loss = seg_loss
        self.class_names = class_names
        self.clip_head = clip_head

        # learnable textual contexts
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        text_features = self.compute_text_features(x, dummy=dummy)
        score_maps = self.compute_score_maps(x, text_features)
        x = list(x[:-1])
        x[3] = torch.cat([x[3], score_maps[3]], dim=1)
        if self.with_neck:
            x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0]
        else:
            return x

    def compute_score_maps(self, x, text_features):
        # B, K, C
        _, visual_embeddings = x[4]
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone, 
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape

        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, dummy=True)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img, use_seg_loss=self.use_seg_loss)
        if self.use_seg_loss:
            x, score_map = x

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        if self.use_seg_loss:
            losses.update(self.compute_seg_loss(img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels))

        return losses

    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels):
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_masks, gt_labels)
        loss = F.binary_cross_entropy(F.sigmoid(score_map), target, weight=mask, reduction='sum')
        loss = loss / mask.sum()
        loss = {'loss_aux_seg': loss}
        return loss

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_masks, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.class_names), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.class_names), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )