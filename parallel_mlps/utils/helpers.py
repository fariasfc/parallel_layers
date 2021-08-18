import torch


def min_ix_argmin(a, n_hidden, ignore_zeros=False, rtol=0):
    """Get the min value of a with the lowest n_hidden in case of draw.

    Args:
        a ([type]): [description]
        n_hidden ([type]): [description]
        ptol (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if rtol > 0:
        # Considering values in a as the same when compared to the max value given a tolerance.
        # old_a = a.clone()
        a = a.clone()
        min_a = torch.min(a)
        ixs = (min_a / a) >= (1 - rtol)
        a[ixs] = min_a

    if ignore_zeros:
        # min_value = np.nanmin(a[a != 0])
        min_value = a[a != 0].min()
    else:
        # min_value = np.nanmin(a)
        min_value = a.min()
    min_ixs = torch.where(a == min_value)[0]
    min_hidden = torch.argmin(n_hidden[min_ixs])
    i = torch.min(min_hidden)
    ix = min_ixs[i]

    return ix