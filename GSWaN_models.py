import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ModuleList, Linear


def mish(x):
    return x.mul(torch.nn.functional.softplus(x).tanh())


class GraphTransformerLayer(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 N,
                 L,
                 dropout,
                 is_attention=False,
                 support_len=3,
                 order=2,
                 n_head=0,
                 softmax_temp=1.0,
                 args=None):
        super().__init__()
        self.args = args
        self.is_attention = is_attention
        self.support_len = support_len
        self.n_head = n_head
        self.softmax_temp = softmax_temp

        if args.activation == 'relu':
            self.a = F.relu
        if args.activation == 'mish':
            self.a = mish

        self.l_key_conv = ModuleList([Conv2d(c_in, 2 * c_out, (1, L), padding=(0, 0), bias=True)
                                      for i_head in range(n_head)])
        self.l_query_conv = ModuleList([Conv2d(c_in, 2 * c_out, (1, L), padding=(0, 0), bias=True)
                                        for i_head in range(n_head)])
        self.nout = ((n_head + 1) * order * support_len + 1)
        # final_conv is basically value
        if is_attention:
            self.final_attention_dense = Linear(c_in * self.nout, self.nout, bias=True)

        # ADD or conCATenate the final gcn heads
        if self.args.gcn_head_aggregate == 'SUM':
            pass
        elif self.args.gcn_head_aggregate == 'CAT':
            c_in = self.nout * c_in
        else:
            print('ERROR, value supposed to be SUM or CAT', self.args.gcn_head_aggregate)
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)

        self.dropout = dropout
        self.order = order

    def forward(self, x, support, nv1, nv2):
        # out = torch.empty((self.nout,)+x.size()).to(x.device)
        # out[0] = x # out[0,:,:,:,:] = x
        # iout = 1
        out = [x]
        H = self.nout
        B, D, N, L = x.shape
        nv1 = nv1.t().reshape(1, D, N, 1).expand(B, D, N, L)
        nv2 = nv2.view(1, D, N, 1).expand(B, D, N, L)
        for a in support:
            # support = torch.stack(support)
            # for ia in range(support.size(0)):
            # a = support[ia]
            for i_head in range(self.n_head + 1):
                # prepearing a for batch
                a_ = a.expand(B, N, N)  # shape: BNN
                x1 = x.clone()  # shape: BDNL
                if i_head >= 1:
                    # transformer
                    keys_input = x.clone()
                    query_input = x.clone()
                    if self.args.spatial_PE_gcn:
                        keys_input += nv1
                        query_input += nv2
                    keys = self.l_key_conv[i_head - 1](keys_input).squeeze(-1)  # shape: BDN
                    query = self.l_query_conv[i_head - 1](query_input).squeeze(-1)  # shape: BDN
                    e = torch.einsum('ben,bem->bnm', (keys, query)).contiguous()  # shape: BNN
                    # e = F.leaky_relu(e)
                    e = self.a(e)
                    zero_vec = -9e15 * torch.ones_like(e)
                    a_ = torch.where(a_ > 0, e, zero_vec)  # shape: BNN
                    a_ /= self.softmax_temp
                    a_ = F.softmax(a_, dim=1)
                    a_ = F.dropout(a_, self.dropout, training=self.training)
                for k in range(1, self.order + 1):
                    # equivalent to nconv, this einsum is multiplication with adjacency matrix
                    x2 = torch.einsum('ncvl,nvw->ncwl', (x1, a_)).contiguous()  # shape: BDNL
                    # out[iout,:,:,:,:] = x2 # this is assignment, not reference
                    # out[iout] = x2
                    # iout += 1
                    out.append(x2)
                    x1 = x2

        if self.is_attention:
            # HB D NL
            # attentive pooling
            out2 = torch.stack(out)
            out2 = out2.transpose(1, 0)
            # out2 = out.transpose(1,0)
            out2 = out2.reshape(B, H * D, N * L)
            out2 = out2.mean(dim=-1)
            final_attention = self.final_attention_dense(out2)
            final_attention = final_attention.view(B, H, 1, 1, 1)
            final_attention = final_attention.transpose(1, 0)  # shape: H,B,1,1,1
            final_attention /= self.softmax_temp
            final_attention = F.softmax(final_attention, dim=0)
            # out = out * final_attention
            for ii in range(len(out)):
                out[ii] = out[ii] * final_attention[ii, :, :, :, :]

        if self.args.gcn_head_aggregate == 'SUM':
            out = torch.stack(out)
            h = out.sum(dim=0)
        elif self.args.gcn_head_aggregate == 'CAT':
            out = torch.stack(out)
            out = out.transpose(1, 0)
            '''
            RuntimeError:
            view size is not compatible with input tensor's size and stride
            (at least one dimension spans across two contiguous subspaces).
            Use .reshape(...) instead.
            '''
            h = out.reshape(B, H * D, N, L)  # shape: BDNL
        else:
            raise Exception('ERROR, value supposed to be SUM or CAT' + str(self.args.gcn_head_aggregate))

        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
