import numpy as np
from enum import Enum
import time
import cv2

from cv2.xfeatures2d import matchGMS

from models.superglue import SuperGlue
from models.superpoint import SuperPoint


# 定义一个枚举类 DrawingType
# 每个枚举项都有一个对应的值，可以通过枚举项名或值来访问。枚举项的值必须是唯一的。
class DrawingType(Enum):
    # 枚举项 ONLY_LINES，值为 1，表示只有线段
    ONLY_LINES = 1
    # 枚举项 LINES_AND_POINTS，值为 2，表示既有线段又有点
    LINES_AND_POINTS = 2
    # 枚举项 COLOR_CODED_POINTS_X，值为 3，表示点按照 X 坐标颜色编码
    COLOR_CODED_POINTS_X = 3
    # 枚举项 COLOR_CODED_POINTS_Y，值为 4，表示点按照 Y 坐标颜色编码
    COLOR_CODED_POINTS_Y = 4
    # 枚举项 COLOR_CODED_POINTS_XpY，值为 5，表示点按照 X 和 Y 坐标的和颜色编码
    COLOR_CODED_POINTS_XpY = 5


# `src1`和`src2`分别是两张待匹配的图片，
# `kp1`和`kp2`分别是两张图片的关键点，
# `matches`是两张图片的匹配结果，
# `drawing_type`是绘制类型。
'''
def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    # 定义画布的高度为两张图片高度的最大值，宽度为两张图片宽度之和
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    # 创建一个三通道的空白画布
    output = np.zeros((height, width, 3), dtype=np.uint8)
    # 将第一张图片复制到画布左侧
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    # # 假设 src1 是形状为 (1, 480, 640) 的数组
    # src1 = np.random.rand(1, 480, 640)
    #
    # # 将 src1 转换为形状为 (480, 640, 1) 的数组
    # src1 = np.transpose(src1, (1, 2, 0))
    #
    # # 创建一个形状为 (480, 640, 3) 的数组
    # output = np.zeros((480, 640, 3))
    #
    # # 将 src1 广播到 output 中
    # output[0:src1.shape[0], 0:src1.shape[1], :] = src1

    # 将第二张图片复制到画布右侧
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    # # 检查目标数组的形状是否正确
    # if output.shape[1] != src1.shape[1]:
    #     raise ValueError("目标数组的第二个维度与源数组的第二个维度不匹配")
    #
    # # 将src2的大小调整为与src1相同
    # src2_resized = np.resize(src2.cpu().numpy(), (1, src1.shape[0], src1.shape[1]))
    #
    # # 将src2调整为目标数组的形状
    # src2_resized = np.resize(src2_resized, output[:, :src2_resized.shape[1], src1.shape[1]:].shape)
    #
    # # 将src2复制到output数组中
    # output[:, :src2_resized.shape[1], src1.shape[1]:] = src2_resized
    
    # 根据不同的绘制类型进行不同的绘制
    # 只绘制连线
    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            # 获取匹配点的坐标
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            # 绘制连线
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))
    # 绘制连线和关键点
    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            # 获取匹配点的坐标
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            # 绘制连线
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            # 获取匹配点的坐标
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            # 绘制关键点
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    # 根据不同的绘制类型生成不同的颜色映射
    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            # 获取匹配点的坐标
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            # 根据不同的绘制类型计算颜色映射索引
            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1] * .5 + left[1] - src1.shape[0] * .5) * 256. / (
                            src1.shape[0] * .5 + src1.shape[1] * .5))  # manhattan gradient

            # 根据颜色映射索引获取颜色，并绘制关键点
            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output
'''
def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]
    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))
    elif drawing_type == DrawingType.LINES_AND_POINTS:
        max_dist = max([m.distance for m in matches])
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            color = tuple(map(int, [0, 0, 255 * (1 - matches[i].distance / max_dist)]))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), color)
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)
    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1] * .5 + left[1] - src1.shape[0] * .5) * 256. / (
                            src1.shape[0] * .5 + src1.shape[1] * .5))  # manhattan gradient
            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


if __name__ == '__main__':
    # 读取图像

    img1 = cv2.imread("../img/Mikolajczyk/bricks/img_6592.ppm")
    img2 = cv2.imread("../img/Mikolajczyk/bricks/img_6596.ppm")

    start = time.time()
    # 创建ORB特征检测器
    orb = cv2.ORB_create(10000)
    # 设置ORB特征检测器的快速阈值为0
    orb.setFastThreshold(0)

    # 对图像进行ORB特征检测和特征描述
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # print(kp1)

    # print(des1)
    # 创建BFMatcher匹配器
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # 对两幅图像的特征描述进行匹配
    matches = matcher.match(des1, des2)
    print('bf', len(matches))

    # # 自适应算法计算最佳网格边缘距离
    # grid_size = 20
    # max_offset = 5
    # grid_points = [[] for _ in range(grid_size * grid_size)]
    # for match in matches:
    #     x, y = kp1[match.queryIdx].pt
    #     grid_x = int(x / img1.shape[1] * grid_size)
    #     grid_y = int(y / img1.shape[0] * grid_size)
    #     for i in range(max(0, grid_x - max_offset), min(grid_size, grid_x + max_offset + 1)):
    #         for j in range(max(0, grid_y - max_offset), min(grid_size, grid_y + max_offset + 1)):
    #             grid_points[i * grid_size + j].append(match)
    #
    # # 将网格边缘的特征点归属到相邻的其他网格
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if i > 0:
    #             grid_points[i * grid_size + j].extend(grid_points[(i - 1) * grid_size + j])
    #         if i < grid_size - 1:
    #             grid_points[i * grid_size + j].extend(grid_points[(i + 1) * grid_size + j])
    #         if j > 0:
    #             grid_points[i * grid_size + j].extend(grid_points[i * grid_size + j - 1])
    #         if j < grid_size - 1:
    #             grid_points[i * grid_size + j].extend(grid_points[i * grid_size + j + 1])
    #
    # # 计算每个匹配点的得分
    # scores = np.zeros(len(matches))
    # for i, match in enumerate(matches):
    #     score = 0
    #     for other_match in grid_points[int(kp1[match.queryIdx].pt[1] / img1.shape[0] * grid_size) * grid_size + int(
    #             kp1[match.queryIdx].pt[0] / img1.shape[1] * grid_size)]:
    #         if other_match.queryIdx == match.queryIdx or other_match.trainIdx == match.trainIdx:
    #             continue
    #         dx1 = kp1[match.queryIdx].pt[0] - kp1[other_match.queryIdx].pt[0]
    #         dy1 = kp1[match.queryIdx].pt[1] - kp1[other_match.queryIdx].pt[1]
    #         dx2 = kp2[match.trainIdx].pt[0] - kp2[other_match.trainIdx].pt[0]
    #         dy2 = kp2[match.trainIdx].pt[1] - kp2[other_match.trainIdx].pt[1]
    #         if dx1 * dx2 + dy1 * dy2 > 0:
    #             score += 1
    #     scores[i] = score

    # 调用matchGMS函数进行GMS匹配
    # 这段代码是调用了一个名为`matchGMS`的函数，
    # 该函数的作用是对两张图片的关键点进行匹配，并使用GMS算法进行筛选，最终返回筛选后的匹配结果。
    # 其中，`img1.shape[:2]`和`img2.shape[:2]`分别表示两张图片的宽度和高度，
    # `kp1`和`kp2`分别表示两张图片的关键点，
    # `matches_all`表示所有匹配结果，
    # `withScale`和`withRotation`表示是否考虑尺度和旋转变换，
    # `thresholdFactor`表示GMS算法的阈值因子。

    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches, withScale=True, withRotation=True,
                           thresholdFactor=6)
    # print('matches_gms', matches_gms)
    end = time.time()

    # 输出匹配结果和GMS匹配时间
    print('GMS', len(matches_gms), 'matches')
    print('GMS takes', end - start, 'seconds')

    # 绘制匹配结果
    output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.LINES_AND_POINTS)

    # 显示匹配结果
    cv2.imshow("show", output)
    cv2.waitKey(0)
