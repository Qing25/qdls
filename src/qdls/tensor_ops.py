import torch
import numpy as np

def safesoftmax(x, dim=-1):
    """ equals `torch.softmax(x, dim=-1)` """
    if type(x) is torch.Tensor:
        return torch.softmax(x-x.max(-1, keepdim=True)[0], dim)
    else:
        x = np.exp(x - np.max(x))
        return x/x.sum()

def feature_transform(feature):
    """ 变换完之后，方差1，均值0 """
    head_feature = torch.log(torch.sqrt(torch.sigmoid(feature)))
    std, mean = torch.std_mean(head_feature, 1, keepdim=True)
    return (head_feature-mean) / std


def select_span_rep(s,e,hidden):
    """
    先将hidden第一位拼接上0向量，使用的index也加1，再gather

    Args:
        s (torch.Tensor): (bsz,)
        e (torch.Tensor): (bsz,)
        hidden (torch.Tensor):  (bsz,seqlen, dim)

    Returns:
        span_rep: (bsz, seq_max+1, dim)
        mask:     (bsz, seq_max+1)          1的位置表示是Mask掉了

    Example:
        >>> s = torch.tensor([4,10,0,1])
        >>> e = torch.tensor([10,12,0,5])
        >>> hidden = torch.randn(4,30,2)
        >>> select_span_rep(s,e,hidden)

    """
    assert torch.all(e.ge(s)), f"结束索引必须大于等于开始索引: e:{e.cpu().numpy()}, s:{s.cpu().numpy()}"
    bsz,seqlen, dim = hidden.size()
    seq_max = torch.max(e-s)+1             # 最小是1最大不限>_>不能超过最大长度
    index = torch.tensor(
        [ [i+1 if i <= _e else 0  for i in range(_s, _s+seq_max)] for _s,_e in zip(s,e)],  # i+1 因为最前面要拼上一个0向量作为pad的表示
        device=hidden.device
    )[:,:,None].repeat((1,1,dim))     # index 不能超过seqlen

    mask = torch.where(index==0, torch.ones_like(index), torch.zeros_like(index))          # index等于0的位置是pad需要mask掉，设置为1
    shift_right = torch.cat([torch.zeros(bsz,1,dim, device=hidden.device), hidden], dim=1)
    span_rep = shift_right.gather(1, index)
    return span_rep, mask[:,:,0]


def select_func(m, i):
    """
        从向量序列中选取对应位置的值
    Args:
        m ( Tensor ): (bsz,seqlen,...)
        i ( Tensor ): (bsz,)

    Returns:
        torch.Tensor: (bsz, 1)
    """
    return torch.stack([ l[idx] for l, idx in zip(m, i)])


def get_max_startend(start_logits, end_logits, max_a_len=None):
    """
        MRC寻找最大概率的区间
    """
    se_sum = end_logits[:,None,:] + start_logits[:,:,None]
    # 限制值的范围是s<e, 最大长度为 max_a_len         # s!=e
    mask = torch.tril(torch.triu(torch.ones_like(se_sum), 0), max_a_len)   # (0,0) is necessary !!!!!!!!!!!!
    r = (mask * se_sum).masked_fill_((1-mask).bool(), float('-inf'))    # score值全是负的，导致0 > score，选出来s>e了
    start_max, end_max = r.max(2)[0].max(1)[1], r.max(1)[0].max(1)[1]
    assert all(end_max.ge(start_max)), "结束索引必须大于等于开始索引"
    return start_max, end_max


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    Label Smoothing起到的作用实际上是抑制了feature norm,
    此时网络输出的softmax prob永远无法达到设定的 1-α/(K-1) ,
    loss曲面上不再存在平缓区域,处处都有较大的梯度指向各个类中心,
    所以特征会更加聚拢. https://zhuanlan.zhihu.com/p/302843504
    For example:
    ```
    >>> y = torch.tensor([[1,0,0,0,0,0]])
    >>> label_smoothing(y)
        tensor([[0.9200, 0.0200, 0.0200, 0.0200, 0.0200, 0.0200]])
    ```
    '''
    num_classes = inputs.size(-1)
    return ((1-epsilon) * inputs) + (epsilon / (num_classes-1))


def mean_pooling(hidden_states, attention_mask):
    """输入一个向量序列， 根据attention_mask计算均值向量
        值为1的地方保留， 值为0的位置被mask
    Args:
        hidden_states (Tensor):  (bsz, seqlen, dim)
        attention_mask (Tensor): (bsz, seqlen)

    Returns:
        Tensor: (bsz, dim)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask