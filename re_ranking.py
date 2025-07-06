import numpy as np

def re_ranking(dist_q_g, dist_q_q, dist_g_g, k1=20, k2=6, lambda_value=0.3):
    all_num = dist_q_q.shape[0] + dist_g_g.shape[0]
    original_dist = np.zeros((all_num, all_num), dtype=np.float32)
    # 构建大矩阵
    original_dist[:dist_q_q.shape[0], :dist_q_q.shape[1]] = dist_q_q
    original_dist[:dist_q_g.shape[0], dist_q_q.shape[1]:] = dist_q_g
    original_dist[dist_q_q.shape[0]:, :dist_q_q.shape[1]] = dist_q_g.T
    original_dist[dist_q_q.shape[0]:, dist_q_q.shape[1]:] = dist_g_g

    V = np.zeros_like(original_dist, dtype=np.float32)
    # 计算 k-reciprocal 邻居
    for i in range(all_num):
        forward_k = np.argsort(original_dist[i])[:k1+1]
        backward_k = [j for j in forward_k
                      if i in np.argsort(original_dist[j])[:k1+1]]
        k_recip = np.array(backward_k)
        weight = np.exp(-original_dist[i, k_recip])
        V[i, k_recip] = weight / np.sum(weight)

    # 局部 QE
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = np.mean(V[np.argsort(original_dist[i])[:k2]], axis=0)
        V = V_qe

    # Jaccard 距离
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    for i in range(all_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            temp_min[0, invIndex[j]] += np.minimum(V[i, j], V[invIndex[j], j])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    # 最终融合
    final_dist = (1 - lambda_value) * jaccard_dist + lambda_value * original_dist
    return final_dist[:dist_q_g.shape[0], dist_q_q.shape[1]:]
