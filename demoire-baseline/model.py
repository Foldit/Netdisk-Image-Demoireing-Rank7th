import math
import collections
from collections import OrderedDict
from itertools import repeat
import pickle

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class WaveletTransform(nn.Layer):
    def __init__(self, scale=1, dec=True, params_path='./wavelet_weights_c2.pkl',
                 transpose=True):
        super(WaveletTransform, self).__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2D(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias_attr=False)
        else:
            self.conv = nn.Conv2DTranspose(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias_attr=False)
        #
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv2DTranspose):
                f = open(params_path, 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                dct = u.load()
                # dct = pickle.load(f)
                f.close()
                initializer = paddle.nn.initializer.Assign(dct['rec%d' % ks])
                # m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                initializer(m.weight)
                m.weight.stop_gradient = True

    def forward(self, x):
        if self.dec:
            # pdb.set_trace()
            output = self.conv(x)
            if self.transpose:
                osz = output.shape
                # print(osz)
                # output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
                output = output.reshape([osz[0], 3, -1, osz[2], osz[3]])
                output = paddle.transpose(output, [0, 2, 1, 3, 4])
                output = output.reshape(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.shape
                # xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
                xx = xx.reshape([xsz[0], -1, 3, xsz[2], xsz[3]])
                xx = paddle.transpose(xx, [0, 2, 1, 3, 4])
                xx = xx.reshape(xsz)
            output = self.conv(xx)
        return output


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.shape[2]
    filter_rows = weight.shape[2]
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class _ConvNd(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if transposed:
            # self.weight = Parameter(torch.Tensor(
            #     in_channels, out_channels // groups, *kernel_size))
            self.weight = self.create_parameter([in_channels, out_channels // groups, *kernel_size], dtype='float32',
                                                  default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))
        else:
            # self.weight = Parameter(torch.Tensor(
            #     out_channels, in_channels // groups, *kernel_size))
            self.weight = self.create_parameter([out_channels, in_channels // groups, *kernel_size], dtype='float32',
                                                  default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))
        self.bias = self.create_parameter([out_channels], is_bias=True, attr=bias, dtype='float32',
                                            default_initializer=paddle.nn.initializer.Uniform(-stdv, stdv))

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def act(act_type, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2D(nc)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2D(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.Pad2D(padding, mode='reflect')
    elif pad_type == 'replicate':
        layer = nn.Pad2D(padding, mode='replicate')
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Layer):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


class ResidualDenseBlock_5C(nn.Layer):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(paddle.concat((x, x1), 1))
        x3 = self.conv3(paddle.concat((x, x1, x2), 1))
        x4 = self.conv4(paddle.concat((x, x1, x2, x3), 1))
        x5 = self.conv5(paddle.concat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2


class DMDB2(nn.Layer):
    """
    DeMoireing  Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='leakyrelu', mode='CNA', delia=1):
        super(DMDB2, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
                                          norm_type, act_type, mode)

        self.deli = nn.Sequential(
            Conv2d(64, 64, 3, stride=1, dilation=delia),
            nn.LeakyReLU(0.2),
        )
        self.deli2 = nn.Sequential(
            Conv2d(64, 64, 3, stride=1),
            nn.LeakyReLU(0.2),
        )
        # self.sam1 = SAM(64,64,1)
        # self.sam2 = SAM(64,64,1)

    def forward(self, x):
        # att1 = self.sam1(x)
        # att2 = self.sam2(x)

        out = self.RDB1(x)
        out = out + x
        out2 = self.RDB2(out)
        deli1 = self.deli(x)
        deli2 = 0.2 * self.deli2(self.deli(x))
        out3 = deli1 + deli2
        # out3 = self.deli(x) + 0.2 * self.deli2(self.deli(x))
        return out2 * 0.2 + out3

class WDNet(nn.Layer):
    def __init__(self, in_channel=3):
        super(WDNet, self).__init__()

        self.cascade1 = nn.Sequential(
            Conv2d(48, 64, 1, stride=1, padding=0),

            nn.LeakyReLU(0.2),

            Conv2d(64, 64, 3, stride=1),

            nn.LeakyReLU(0.2),
        )

        self.cascade2 = nn.Sequential(

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=1),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=2),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=5),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=7),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=12),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=19),

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                  norm_type=None, act_type='leakyrelu', mode='CNA', delia=31)
        )

        self.final = nn.Sequential(
            conv_block(64, 48, kernel_size=1, norm_type=None, act_type=None)
        )
        self.xbranch = nn.Sequential(
            conv_block(3, 64, kernel_size=3, norm_type=None, act_type='leakyrelu')
        )

    def forward(self, x):
        x1 = self.cascade1(x)
        # pdb.set_trace()

        x1 = self.cascade2(x1)

        x = self.final(x1)

        return x

if __name__ == '__main__':
    net = WDNet()
    img = paddle.randn([1,48,224,224])
    out = net(img)
    print(out.shape)
