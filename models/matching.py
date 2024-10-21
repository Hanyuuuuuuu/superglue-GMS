# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os

import numpy as np
import torch
from torch.autograd._functions import tensor


from .superpoint import SuperPoint
from .superglue import SuperGlue

import cv2
from cv2.xfeatures2d import matchGMS

from models.select_data import Select_data



class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue)
    图像匹配前端（SuperPoint + SuperGlue）"""

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
          运行SuperPoint（可选）和SuperGlue
         如果输入中存在['keypoints0'，'keypoints1']，则跳过SuperPoint
         参数：
         data：字典，最小键为['image0'，'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        # 如果未提供关键点、分数和描述符，则提取SuperPoint。
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k + '0': v for k, v in pred0.items()}}
            # print(pred)
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k + '1': v for k, v in pred1.items()}}

        # print(pred)
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        # 批量处理所有特征
        # 我们应该要么 i）每个批次只有一张图片，要么
        # ii）批次中所有图像的本地特征数量相同。
        # 将两个字典`data`和`pred`合并，并将合并后的结果赋值给`data`。
        # 然后，对于`data`中的每个键值对，如果其值是列表或元组类型，就将其转换为
        # PyTorch的Tensor类型，并重新赋值给`data`中的该键。
        # 这个过程中，使用了Python3.5中的新特性` ** ` 来进行字典的合并。
        data = {**data, **pred}
        # print(data)

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # 进行匹配。
        # pred = {**pred, **self.superglue(data)}
        data_gms = Select_data(data)
        # print('pred', pred)
        # print('data_gms', data_gms)
        pred = {**data_gms, **self.superglue(data_gms)}
        # print(pred['matches0'].shape)

        return pred
