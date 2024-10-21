import time

import numpy as np
import torch

import cv2
from cv2.xfeatures2d import matchGMS

from .opencv_demo import draw_matches, DrawingType


def Select_data(data):
    if 'image0' in data:
        img0 = data['image0'].cpu().numpy() * 255  # 将tensor转换为numpy数组，并将像素值从[0,1]转换为[0,255]
        img0 = img0.astype(np.uint8)  # 将像素值的数据类型转换为uint8
        img0 = np.transpose(img0, (0, 2, 3, 1))  # 将通道维度调整到最后一维
        img0 = img0[0]  # 去掉batch维度
    if 'image1' in data:
        img1 = data['image1'].cpu().numpy() * 255  # 将tensor转换为numpy数组，并将像素值从[0,1]转换为[0,255]
        img1 = img1.astype(np.uint8)  # 将像素值的数据类型转换为uint8
        img1 = np.transpose(img1, (0, 2, 3, 1))  # 将通道维度调整到最后一维
        img1 = img1[0]  # 去掉batch维度
        # print(img0)

    if 'keypoints0' in data:
        # 转换为cv2.KeyPoint对象列表
        keypoints0 = []
        for kpt in data['keypoints0'][0]:
            x, y = kpt[0].item(), kpt[1].item()
            keypoints0.append(cv2.KeyPoint(x, y, 1))
    if 'keypoints1' in data:
        # 转换为cv2.KeyPoint对象列表
        keypoints1 = []
        for kpt in data['keypoints1'][0]:
            x, y = kpt[0].item(), kpt[1].item()
            keypoints1.append(cv2.KeyPoint(x, y, 1))
    if 'descriptors0' in data:
        desc0 = data['descriptors0'][0].cpu().numpy()
        desc0 = (desc0 * 255).astype(np.uint8)
        desc0 = np.round(desc0.T).astype(np.uint8)
    if 'descriptors1' in data:
        desc1 = data['descriptors1'][0].cpu().numpy()
        desc1 = (desc1 * 255).astype(np.uint8)
        desc1 = np.round(desc1.T).astype(np.uint8)

    # 使用BFMatcher进行特征点匹配
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc0, desc1)
    print('BF', len(matches))
    # withScale和withRotation参数用于控制是否考虑尺度和旋转变换。t
    # hresholdFactor参数用于控制筛选的阈值，值越大，筛选出的匹配点越少
    matches_gms = matchGMS(img0.shape[:2], img1.shape[:2], keypoints0, keypoints1, matches, withScale=True,
                           withRotation=True, thresholdFactor=7)

    s=time.time()
    # RANSAC
    obj = []
    scene = []
    matches_gms = [matchs for matchs in matches_gms]
    for match in matches_gms:
        obj.append(keypoints0[match.queryIdx].pt)
        scene.append(keypoints1[match.trainIdx].pt)
    kp1_ransac = np.array(obj)
    kp2_ransac = np.array(scene)

    H, inliers = cv2.findHomography(kp1_ransac, kp2_ransac, cv2.RANSAC)
    # 创建关键点对象
    good_matches_ransac = [match for i, match in enumerate(matches_gms) if inliers[i]]
    # img_matches_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good_matches_ransac,
    #                                      None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print('ransac', len(good_matches_ransac))
    e = time.time()
    # 输出匹配结果和GMS匹配时间
    # print('Found', len(matches_gms), 'matches')
    print('GMS takes', e - s, 'seconds')



    # print(matches_gms)
    # 输出匹配结果
    print('GMS', len(matches_gms))

     # 绘制匹配结果
    # output = draw_matches(img0, img1, keypoints0, keypoints1, matches_gms, DrawingType.LINES_AND_POINTS)
    #
     # 显示匹配结果
    # cv2.imshow("show", output)
    # cv2.waitKey(0)

    # 提取matches_gms中的特征点
    keypoints0 = [keypoints0[match.queryIdx].pt for match in matches_gms]
    keypoints1 = [keypoints1[match.trainIdx].pt for match in matches_gms]

    '''keypoints数据格式处理'''
    # print(keypoints0)
    kpts0 = torch.tensor(keypoints0).unsqueeze(0).cuda()
    kpts1 = torch.tensor(keypoints1).unsqueeze(0).cuda()

    data_gms = {'keypoints0': kpts0,
                'keypoints1': kpts1}

    # 获取data字典中keypoints0和keypoints1的数据
    data_keypoints0 = data['keypoints0']
    data_keypoints1 = data['keypoints1']

    # 获取data_gms字典中keypoints0和keypoints1的数据
    data_gms_keypoints0 = data_gms['keypoints0']
    data_gms_keypoints1 = data_gms['keypoints1']

    # 将data_gms_keypoints0转换为numpy数组
    '''gms_keypoints0 = data_gms_keypoints0.numpy()[0]
       numpy只能处理CPU上的数据。需要使用`.cpu()`方法将数据从GPU复制到CPU上，
       然后再使用`.numpy()`方法将其转换为numpy数组'''
    gms_keypoints0 = data_gms_keypoints0.cpu().numpy()[0]
    gms_keypoints1 = data_gms_keypoints1.cpu().numpy()[0]

    # 初始化保存索引的列表
    indices0 = []
    indices1 = []

    # 遍历data中的关键点
    for i, kp0 in enumerate(data_keypoints0[0]):
        # 将关键点转换为numpy数组
        # kp_np0 = kp0.numpy()
        kp_np0 = kp0.cpu().numpy()
        # 判断是否与gms_keypoints0中的任意一个关键点相同
        if np.any(np.all(np.isclose(gms_keypoints0, kp_np0, rtol=1e-03), axis=1)):
            # 如果相同，则将索引保存到列表中
            indices0.append(i)

    for i, kp1 in enumerate(data_keypoints1[0]):
        # 将关键点转换为numpy数组
        kp_np1 = kp1.cpu().numpy()
        # 判断是否与gms_keypoints1中的任意一个关键点相同
        if np.any(np.all(np.isclose(gms_keypoints1, kp_np1, rtol=1e-03), axis=1)):
            # 如果相同，则将索引保存到列表中
            indices1.append(i)

    # print(data)
    # 根据data中keypoints的索引，提取相应的descs,scores,imangs
    # print(data['descriptors0'].shape)
    #  # 将索引转换为tensor，并移动到GPU上
    indices_tensor0 = torch.tensor(indices0, device=data['descriptors0'].device)
    indices_tensor1 = torch.tensor(indices1, device=data['descriptors1'].device)
    # print(data['keypoints0'].shape)
    keypoints0 = torch.index_select(data['keypoints0'], 1, indices_tensor0)
    keypoints1 = torch.index_select(data['keypoints1'], 1, indices_tensor1)

    descriptors0 = torch.index_select(data['descriptors0'], 2, indices_tensor0)
    descriptors1 = torch.index_select(data['descriptors1'], 2, indices_tensor1)

    scores0 = torch.index_select(data['scores0'], 1, indices_tensor0)
    scores1 = torch.index_select(data['scores1'], 1, indices_tensor1)

    images0 = data['image0']
    images1 = data['image1']

    data_gms = {'keypoints0': keypoints0,
                'keypoints1': keypoints1,
                'descriptors0': descriptors0,
                'descriptors1': descriptors1,
                'image0': images0,
                'image1': images1,
                'scores0': scores0,
                'scores1': scores1
                }

    return data_gms
