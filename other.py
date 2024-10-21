# descriptors1 = [desc1[match.trainIdx] for match in matches_gms]
# descriptors0 = [desc0[match.queryIdx] for match in matches_gms]
# image0 = torch.from_numpy(img0).unsqueeze(0)
# image1 = torch.from_numpy(img1).unsqueeze(0)
# scores0 = (torch.Tensor([match.distance for match in matches_gms]),)
# scores1 = (torch.Tensor([match.distance for match in matches_gms]),)
# 提取关键点得分
# scores0 = [match.distance for match in matches_gms]
# data_gms = {'keypoints0': keypoints0,
#             'keypoints1': keypoints1,
#             'descriptors0': descriptors0,
#             'descriptors1': descriptors1,
#             # 'scores0': scores0
#             }
# print(data_gms)

# # 填充kpts的维度
# # 计算需要填充的数量
# pad_len0 = data['keypoints0'].shape[1] - kpts0.shape[1]
# pad_len1 = data['keypoints1'].shape[1] - kpts1.shape[1]
# # 进行填充
# kpts0 = torch.nn.functional.pad(kpts0, (0, 0, 0, pad_len0), mode='constant', value=0)
# kpts1 = torch.nn.functional.pad(kpts1, (0, 0, 0, pad_len1), mode='constant', value=0)


# '''desc数据格式处理'''
        # # desc0, desc1 = data_gms['descriptors0'], data_gms['descriptors1']
        # # 将输入的特征描述符和关键点坐标转换为PyTorch张量，并将它们放在GPU上进行计算
        # desc0 = torch.tensor(descriptors0, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to('cuda')
        # desc1 = torch.tensor(descriptors1, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to('cuda')
        # # print(desc0)
        # # 将原始数据转换为标准化数据
        # # 转换为浮点数类型的张量
        # desc0 = desc0.float()
        # desc1 = desc1.float()
        # # 计算均值和标准差
        # mean0 = torch.mean(desc0)
        # std0 = torch.std(desc0)
        # mean1 = torch.mean(desc1)
        # std1 = torch.std(desc1)
        #
        # # 标准化处理
        # transform0 = transforms.Compose([
        #     transforms.Normalize(mean=[mean0], std=[std0])
        # ])
        # desc0 = transform0(desc0)
        # transform1 = transforms.Compose([
        #     transforms.Normalize(mean=[mean1], std=[std1])
        # ])
        # desc1 = transform1(desc1)
        # data_gms = {desc0, desc1}
        # print(data_gms)
        # 计算需要填充的长度
        # pad_len0 = data['scores0'].shape[1] - desc0.shape[2]
        # pad_len1 = data['scores1'].shape[1] - desc1.shape[2]
        # 进行填充
        # desc0 = torch.nn.functional.pad(desc0, (0, pad_len0))
        # desc1 = torch.nn.functional.pad(desc1, (0, pad_len1))

        # '''scores数据格式处理'''
        # print(data['scores0'])

        # 将数据转为二维张量
        # scores0 = torch.unsqueeze(scores0[0], 0)
        # scores1 = torch.unsqueeze(scores1[0], 0)
        # 对数据进行归一化
        # scores0 = (scores0 - scores0.min()) / (scores0.max() - scores0.min())
        # scores1 = (scores1 - scores1.min()) / (scores1.max() - scores1.min())
        # 将数据转移到GPU上
        # scores0 = scores0.to('cuda')
        # scores1 = scores1.to('cuda')

        # 填充scores的维度
        # 计算需要填充的数量
        # pad_len0 = data['scores0'].shape[1] - scores0.shape[1]
        # pad_len1 = data['scores1'].shape[1] - scores1.shape[1]
        # 进行填充
        # scores0 = torch.nn.functional.pad(scores0, (0, pad_len0), mode='constant', value=0)
        # scores1 = torch.nn.functional.pad(scores1, (0, pad_len1), mode='constant', value=0)

        # # 转换为二维tensor并移到cuda上
        # scores0 = torch.unsqueeze(scores0[0], 0).to('cuda:0')

        # matches_gms = data_gms['matches']
        # # 从matches_gms中提取匹配的索引
        # matches_idx = matches_gms.nonzero()[0]
        #
        # # 从data中提取scores0和scores1
        # scores0 = data['scores0'][matches_idx]
        # scores1 = data['scores1'][matches_idx]
        # print(data['scores1'].shape)
        # print(scores0.shape)

        # '''imamg数据格式处理'''
        # # 将数据类型转换为float
        # image0 = image0.float()
        # image1 = image1.float()
        # # 将数据归一化到0到1之间
        # image0 /= 255
        # image1 /= 255
        # # 将数据移动到GPU上（如果有的话）
        # if torch.cuda.is_available():
        #     image0 = image0.cuda()
        # if torch.cuda.is_available():
        #     image1 = image1.cuda()

        # 将维度顺序改为与data[image0]相同
        # image0 = image0.transpose(1, 3).transpose(2, 3)
        # image1 = image1.transpose(1, 3).transpose(2, 3)

        # print(image0.shape)
        # print(data['image0'].shape)