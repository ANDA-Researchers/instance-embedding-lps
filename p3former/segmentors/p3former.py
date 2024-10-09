import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 offset_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_offset: bool = True) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)
        self.use_offset = use_offset
        if self.use_offset:
            self.offset_head = MODELS.build(offset_head)
    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x = self.extract_feat(batch_inputs_dict)
        
        # Save features
        batch_inputs_dict['features'] = x.features

        #  Loss
        losses = dict()

        # Offset head forward and calculate loss
        if self.use_offset:
            pred_offsets = self.offset_head(x, batch_inputs_dict)
            offset_loss = self.offset_loss(pred_offsets, batch_data_samples)
            losses['offset_loss'] = sum(offset_loss)

        for batch_i, p in enumerate(batch_inputs_dict['points']):
            batch_inputs_dict['points'][batch_i] = p[:,:3]

        # Point shifting
        embedding = [offset.detach() + xyz for offset, xyz in zip(pred_offsets, batch_inputs_dict['points'])]

        batch_inputs_dict['embedding'] = embedding

        # Decode head forward and calculate loss
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        if self.use_offset:
            pred_offsets = self.offset_head(x, batch_inputs_dict)
            for batch_i, p in enumerate(batch_inputs_dict['points']):
                batch_inputs_dict['points'][batch_i] = p[:,:3]
            assert len(pred_offsets[0]) == len(batch_inputs_dict['points'][0])
            embedding = [offset + xyz for offset, xyz in zip(pred_offsets, batch_inputs_dict['points'])]
            batch_inputs_dict['embedding'] = embedding

            # validate_offset(pred_offsets, batch_inputs_dict, batch_data_samples, vis=True)
        pts_semantic_preds, pts_instance_preds = self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds, pts_instance_preds, batch_data_samples)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds, batch_data_samples):
        for i in range(len(pts_semantic_preds)):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': pts_semantic_preds[i],
                                                'pts_instance_mask': pts_instance_preds[i]})})
        return batch_data_samples
    
    def offset_loss(self, pred_offsets, batch_data_samples):
        loss_list_list = []
        for i, b in enumerate(batch_data_samples):
            valid = torch.from_numpy(b.gt_pts_seg.pts_valid).cuda()
            gt_offsets = b.gt_pts_seg.pts_offsets
            pt_diff = pred_offsets[i] - gt_offsets   # (N, 3)
            pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
            valid = valid.view(-1).float()
            offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
            loss_len = len((loss_list_list,))
            if len(loss_list_list) < loss_len:
                loss_list_list = [[] for j in range(loss_len)]
            for j in range(loss_len):
                loss_list_list[j].append((offset_norm_loss,)[j])
        mean_loss_list = []
        for i in range(len(loss_list_list)):
            mean_loss_list.append(torch.mean(torch.stack(loss_list_list[i])))
        return mean_loss_list
