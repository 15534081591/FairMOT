"""
Fairmot for training and evaluation
"""

import mindspore.nn as nn


class WithLossCell(nn.Cell):
    """Cell with loss function.."""

    def __init__(self, net, loss):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss

    def construct(self, image, hm, reg_mask, ind, wh, reg, ids):
        """Cell with loss function."""
        feature = self._net(image)
        return self._loss(feature, hm, reg_mask, ind, wh, reg, ids)

    @property
    def backbone_network(self):
        """Return net."""
        return self._net


class WithNetCell(nn.Cell):
    """Cell with infer_net function.."""

    def __init__(self, net, infer_net):
        super(WithNetCell, self).__init__(auto_prefix=False)
        self._net = net
        self._infer_net = infer_net

    def construct(self, image):
        """Cell with loss function."""
        feature = self._net(image)
        return self._infer_net(feature)
