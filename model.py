from torch.nn import BatchNorm2d, Conv1d, Parameter

from GSWaN_models import *


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A)).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list, nv1, nv2):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Predictive_head_mlp(nn.Module):
    def __init__(self, skip_channels, end_channels, out_dim, a):
        super().__init__()
        self.a = a
        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(self, x):
        x = self.a(x)
        x = self.a(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class GWNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, cat_feat_gc=False,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 apt_size=10, is_gat=False, is_gcn_attention=False, gcn_n_head=0, softmax_temp=1.0,
                 args=None):
        super().__init__()
        self.args = args
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj
        depth = list(range(blocks * layers))

        # ACTIVATION
        if args.activation == 'relu':
            self.a = F.relu
        elif args.activation == 'mish':
            self.a = mish
        else:
            print('WARNING: no valid activation is selected, using identity instead.\n' +
                  'valid activations: relu, mish.\m' +
                  'activation selected: ' + repr(args.activation))
            self.a = torch.nn.Sequential()

        # START CONV EMBED
        if cat_feat_gc:
            self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d(in_channels=in_dim - 1,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
            self.start_embed = self.start_embed_cat_feat_gc
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.start_embed = self.start_embed_v1

        # nodevec embed
        self.fixed_supports = supports or []
        self.supports_len = len(self.fixed_supports)
        if do_graph_conv:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]
        else:
            self.residual_convs = ModuleList([Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        if addaptadj:
            self.supports_len += 1
            self.forward_adj = self.forward_addaptadj
        else:
            self.forward_adj = self.forward_fixedadj

        # temporal conv
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.skip_convs = ModuleList([Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])

        # GCN
        if is_gat:
            lL = [12, 10, 9, 7, 6, 4, 3, 1]
            self.graph_convs = ModuleList([GraphTransformerLayer(dilation_channels,
                                                                 residual_channels,
                                                                 num_nodes,
                                                                 lL[_],
                                                                 dropout,
                                                                 support_len=self.supports_len,
                                                                 is_attention=is_gcn_attention,
                                                                 n_head=gcn_n_head,
                                                                 softmax_temp=softmax_temp,
                                                                 args=args)
                                           for _ in depth])
        else:
            self.graph_convs = ModuleList([GraphConvNet(dilation_channels,
                                                        residual_channels,
                                                        dropout,
                                                        support_len=self.supports_len)
                                           for _ in depth])

        # batch norm
        if args.is_batch_norm:
            self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        else:
            self.bn = ModuleList([torch.nn.Sequential() for _ in depth])

        # end conv
        if not 'head_type' in vars(args).keys():
            self.predictive_head = Predictive_head_mlp(skip_channels, end_channels, out_dim, self.a)
        elif args.head_type == 'MLP':
            self.predictive_head = Predictive_head_mlp(skip_channels, end_channels, out_dim, self.a)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict(dropout=args.dropout,
                        supports=supports,
                        do_graph_conv=args.do_graph_conv,
                        addaptadj=args.addaptadj,
                        aptinit=aptinit,
                        in_dim=args.in_dim,
                        apt_size=args.apt_size,
                        out_dim=args.seq_length,
                        residual_channels=args.nhid,
                        dilation_channels=args.nhid,
                        skip_channels=args.nhid * 8,
                        end_channels=args.nhid * 16,
                        cat_feat_gc=args.cat_feat_gc,
                        is_gat=args.is_gat,
                        is_gcn_attention=args.is_gcn_attention,
                        gcn_n_head=args.gcn_n_head,
                        softmax_temp=args.softmax_temp,
                        args=args)
        defaults.update(**kwargs)
        model = cls(device, args.num_nodes, **defaults)
        return model

    def load_checkpoint(self, state_dict):
        """It is assumed that ckpt was trained to predict a subset of timesteps."""
        bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']  # only weights that depend on seq_length
        b, w = state_dict.pop(bk), state_dict.pop(wk)
        self.load_state_dict(state_dict, strict=False)
        cur_state_dict = self.state_dict()
        cur_state_dict[bk][:b.size(0)] = b
        cur_state_dict[wk][:w.size(0)] = w
        self.load_state_dict(cur_state_dict)

    def start_embed_cat_feat_gc(self, x):
        # f1, f2 = x[:, [0]], x[:, 1:]
        f1, f2 = x[:, 0:1, :, :], x[:, 1:2, :, :]
        x1 = self.start_conv(f1)
        x2 = self.a(self.cat_feature_conv(f2))  # F.leaky_relu
        x = x1 + x2
        return x

    def start_embed_v1(self, x):
        x = self.start_conv(x[:, 0:2, :, :])
        return x

    def forward_fixedadj(self, x):
        return x, self.fixed_supports

    def forward_addaptadj(self, x):
        # calculate the current adaptive adj matrix once per iteration
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adjacency_matrices = self.fixed_supports + [adp]
        return x, adjacency_matrices

    def forward_main(self, x):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            # if self.args.verbose >= 4:
            #     print('in_len: ' + str(in_len))
            x = nn.functional.pad(x, [self.receptive_field - in_len, 0, 0, 0])
        x = self.start_embed(x)
        x, adjacency_matrices = self.forward_adj(x)

        # WaveNet layers
        skip = torch.zeros(1, 1, 1, 1).to(x.device)
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            # temporal residual
            x_t = filter * gate
            skip = self.skip_convs[i](x_t)[:, :, :, -1:] + skip
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break
            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x_t, adjacency_matrices, self.nodevec1, self.nodevec2)
                x_s = x_t * self.args.cat_feat_gc + graph_out
            else:
                x_s = self.residual_convs[i](x_t)
            # residual
            x = x_s + residual[:, :, :, -x_s.size(3):]  # TODO(SS): Mean/Max Pool?
            x = self.bn[i](x)
        x = skip
        return x

    def forward(self, x):
        x = self.forward_main(x)
        x = self.predictive_head(x)
        return x
