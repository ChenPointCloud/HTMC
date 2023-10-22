import numpy as np
import matplotlib.pyplot as plt

# 从文本文件读取数据
# ori = np.loadtxt('checkpoints/hie589_mask1/recall.txt')
# inter = np.loadtxt('checkpoints/hie589_mask1_inter/recall.txt')
# ori = np.loadtxt('checkpoints/hie589_mask1/mask_loss.txt')
# inter = np.loadtxt('checkpoints/hie589_mask1_inter/mask_loss.txt')
# ori = np.loadtxt('index2/precisions_mask_batch.txt')
# inter = np.loadtxt('index0.5/precisions_mask_batch.txt')
ori = np.loadtxt('index2/recalls_mask_batch.txt')
inter = np.loadtxt('index0.5/recalls_mask_batch.txt')
# decay = np.loadtxt('checkpoints/no_weight_decay/recall.txt')


x = np.arange(len(ori))

# 绘制折线图
# plt.plot(x, ori, label='Detach', marker='o')
# plt.plot(x, inter, label='Attach', marker='v')

# 绘制折线图
plt.scatter(x, ori, label='Square', marker='o')
plt.scatter(x, inter, label='Root', marker='v')

# 添加标题和轴标签
# plt.title('Line Plot')
plt.xlabel('Index')
plt.ylabel('Recall')
plt.legend()

# 显示图形
plt.show()


def KPConv(query_points,
           support_points,
           neighbors_indices,
           features,
           K_values,
           fixed='center',
           KP_extent=1.0,
           KP_influence='linear',
           aggregation_mode='sum'):
    """
    This function initiates the kernel point disposition before building KPConv graph ops
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return: output_features float32[n_points, out_fdim]
    """

    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = create_kernel_points(K_radius,
                                          num_kpoints,
                                          num_kernels=1,
                                          dimension=points_dim,
                                          fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = tf.Variable(K_points_numpy.astype(np.float32),
                           name='kernel_points',
                           trainable=False,
                           dtype=tf.float32)

    return KPConv_ops(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_points,
                      K_values,
                      KP_extent,
                      KP_influence,
                      aggregation_mode)