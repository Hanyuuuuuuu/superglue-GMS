#! /usr/bin/env python3
#
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
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import time
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
from models.select_data import Select_data

if __name__ == '__main__':
    # 运行SuperGlue模型演示
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='img/Mikolajczyk/cars',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default='img/result',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg', '*.ppm', '*.pgm'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    # Superpoint检测到的关键点的最大数量（'-1'保留所有关键点）
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    # SuperPoint关键点检测器置信度阈值
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    # SuperPoint非极大值抑制（NMS）半径（必须为正数）。
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    # SuperGlue执行的Sinkhorn迭代次数
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    # SuperGlue匹配阈值。
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    # 不要在屏幕上显示图像。如果远程运行时非常有用。
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    # 强制PyTorch在CPU模式下运行。
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    opt = parser.parse_args()
    print(opt)

    '''这段代码是用来解析命令行参数的，
    其中 opt.resize 表示需调整的图片大小。
    如果 opt.resize 是一个长度为 2 的数组，表示需要将图片调整为指定的宽度和高度；
    如果 opt.resize 是一个长度为 1 的正整数，表示需要将图片的最大边缩放到指定的大小；
    如果 opt.resize 是一个长度大于 2 的数组，会抛出 ValueError 异常。
    最后会打印出将要进行的图片调整操作。'''
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    # matching是Matching类的实例，用于进行图像匹配。
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    '''这段代码是使用VideoStreamer类来读取视频帧。
    opt.input是视频文件的路径，
    opt.resize是一个布尔值，表示是否需要调整帧的大小。
    opt.skip是一个整数，表示每隔多少帧读取一次。
    opt.image_glob是一个字符串，表示要读取的图像文件的通配符。
    opt.max_length是一个整数，表示读取的最大帧数。'''
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    '''调用next_frame()方法可以读取下一帧，
    并返回该帧和一个布尔值ret，表示读取是否成功。
    如果ret为False，则说明读取失败。
    如果ret为True，则说明读取成功，
    frame变量包含了读取的帧。
    如果视频已经读取完毕，则next_frame()方法会返回(None, False)。
    这里使用assert语句来确保第一帧读取成功，如果读取失败则会抛出异常'''
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    '''这段代码的作用是将视频帧转换为张量，
    并使用SuperPoint算法提取关键点和描述符。
    然后将提取的数据存储在last_data字典中，
    并将帧张量和帧图像ID存储在last_frame和last_image_id变量中，以备后续使用。
    其中，keys是一个包含关键点和描述符的名称的列表。'''
    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0
    #
    # frame_tensor = frame2tensor(frame, device)
    # pred_data = matching.superpoint({'image': frame_tensor})
    # pred_data = {k + '1': pred_data[k] for k in keys}
    # pred_data['image1'] = frame_tensor

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    # 创建一个窗口来显示演示。
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640 * 2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    # 打印键盘帮助菜单。
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')
    # 计时器，用于计算程序的平均运行时间

    timer = AverageTimer()
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        # 进行特征点匹配
        '''last_data:i、k，s、d~0
           pred:k,s,d~1,matches0-1,matching_scores0-1
        '''
        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})  # 匹配结果，包括匹配后的特征点、匹配矩阵等信息。
        # pred = matching({'image0': frame_tensor, 'image1': frame_tensor})
        # print('dame_pred', pred)
        # kpts0 = last_data['keypoints0'][0].cpu().numpy()  # 上一帧图像中的特征点坐标，转换为 numpy 数组。
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()  # 当前帧图像中的特征点坐标，转换为 numpy 数组
        matches = pred['matches0'][0].cpu().numpy()
        # print(matches)
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        # print(kpts0.shape)
        # print(valid.shape)
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        print('SG {}'.format(len(mkpts0)))
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k + '0': pred[k + '1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)

                pred_data = {k + '1': pred[k + '1'] for k in keys}
                pred_data['image1'] = frame_tensor

            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold'] * (1 + d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold'] + d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints
        e = time.time()
        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            # stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()
