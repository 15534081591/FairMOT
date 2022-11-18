"""export fairmot."""
import numpy as np
from mindspore import context, Tensor, export
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint

from src.config import Opts
from src.backbone_dla_conv import DLASegConv
from src.infer_net import InferNet
from src.fairmot_pose import WithNetCell


def fairmot_export(opt):
    """export fairmot to mindir or air."""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        save_graphs=False,
        device_id=opt.id)
    backbone_net = DLASegConv(opt.heads,
                              down_ratio=4,
                              final_kernel=1,
                              last_level=5,
                              head_conv=256,
                              is_training=True)
    load_checkpoint(opt.load_model, net=backbone_net)
    infer_net = InferNet()
    net = WithNetCell(backbone_net, infer_net)
    net.set_train(False)
    input_data = Tensor(np.zeros([1, 3, 608, 1088]), mstype.float32)
    export(net, input_data, file_name='fairmot', file_format="MINDIR")


if __name__ == '__main__':
    opt_ = Opts().get_config()
    fairmot_export(opt_)
