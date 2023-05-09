import numpy as np
import torch


# TODO:
# parameterize stuff

class Augment:
    def __init__(self, args):
        self.args = args
        self.la = []
        print('augmentations start')
        if args.augment_occlude_spatial_probability > 0:
            print('occlude_spatial_probability', args.augment_occlude_spatial_probability,
                  'occlude_spatial_scale', args.augment_occlude_spatial_scale)
            self.la.append(self.occlude_spatial)
        if args.augment_occlude_temporal_probability > 0:
            print('occlude_temporal_probability', args.augment_occlude_temporal_probability,
                  'occlude_temporal_scale', args.augment_occlude_temporal_scale)
            self.la.append(self.occlude_temporal)
        if args.augment_swap_spatial_k > 1:
            print('swap_spatial_k', args.augment_swap_spatial_k)
            self.la.append(self.swap_spatial)
        if args.augment_swap_temporal_k > 1:
            print('swap_temporal_k', args.augment_swap_temporal_k)
            self.la.append(self.swap_temporal)
        if args.augment_scramble_spatial_probability > 0:
            print('scramble_spatial_probability', args.augment_scramble_spatial_probability)
            self.la.append(self.scramble_spatial)
        if args.augment_scramble_temporal_probability > 0:
            print('scramble_temporal_probability', args.augment_scramble_temporal_probability)
            self.la.append(self.scramble_temporal)
        if args.augment_uniform_noise_scale > 0:
            print('uniform_noise_scale', args.augment_uniform_noise_scale)
            self.la.append(self.uniform_noise)
        print('augmentations end')
        return

    def augment(self, x):
        with torch.no_grad():
            for ia in self.la:
                x = ia(x)
        return x

    def occlude_spatial(self, x):  # bdnm
        b, d, n, m = x.size()
        mask = torch.rand(b, 1, n, 1)
        mask = mask < self.args.augment_occlude_spatial_probability
        mask = mask.expand(-1, d, -1, m).clone()
        mask[:, 1:, :, :] = False
        tensor_random = torch.rand_like(mask.float())[mask].to(x.device)
        x[mask] *= tensor_random * self.args.augment_occlude_spatial_scale
        return x

    def occlude_temporal(self, x):  # bdnm
        b, d, n, m = x.size()
        mask = torch.rand(b, 1, 1, m)
        mask = mask < self.args.augment_occlude_temporal_probability
        mask = mask.expand(-1, d, n, -1).clone()
        mask[:, 1:, :, :] = False
        tensor_random = torch.rand_like(mask.float())[mask].to(x.device)
        x[mask] *= tensor_random * self.args.augment_occlude_temporal_scale
        return x

    def swap_spatial(self, x):  # bdnm
        k = self.args.augment_swap_spatial_k
        for ib in range(x.size(0)):
            i1 = np.random.choice(np.arange(x.size(2)), k, replace=False).astype(int)
            i2 = np.random.permutation(i1)
            x[ib, :, i1, :] = x[ib, :, i2, :]
        return x

    def swap_temporal(self, x):
        k = self.args.augment_swap_temporal_k
        for ib in range(x.size(0)):
            i1 = np.random.choice(np.arange(x.size(3)), k, replace=False).astype(int)
            i2 = np.random.permutation(i1)
            x[ib, :, :, i1] = x[ib, :, :, i2]
        return x

    def scramble_spatial(self, x):
        for ib in range(x.size(0)):
            for inn in range(x.size(2)):
                if torch.rand(1) < self.args.augment_scramble_spatial_probability:
                    x[ib, :, inn, :] = x[ib, :, inn, np.random.permutation(np.arange(x.size(3)))]
        return x

    def scramble_temporal(self, x):
        for ib in range(x.size(0)):
            for il in range(x.size(3)):
                if torch.rand(1) < self.args.augment_scramble_spatial_probability:
                    x[ib, :, :, il] = x[ib, :, np.random.permutation(np.arange(x.size(2))), il]
        return x

    def uniform_noise(self, x):  # bdnm
        rand = (torch.rand_like(x) - 0.5) * self.args.augment_uniform_noise_scale
        rand[:, 1:, :, :] = 0
        return x + rand

    def datapoint_zero_mean(self, x):  # bdnm
        mu = x[:, 0:1, :, :].mean(dim=[1, 2, 3])
        x[:, 0:1, :, :] = x[:, 0:1, :, :] - mu.view(-1, 1, 1, 1)
        return x

    def temporal_zero_mean(self, x):  # bdnm
        mu = x[:, 0:1, :, :].mean(dim=[1, 2])
        x[:, 0:1, :, :] = x[:, 0:1, :, :] - mu.view(mu.size(0), 1, 1, mu.size(1))
        return x

    def spatial_zero_mean(self, x):  # bdnm
        mu = x[:, 0:1, :, :].mean(dim=[1, 3])
        x[:, 0:1, :, :] = x[:, 0:1, :, :] - mu.view(mu.size(0), 1, mu.size(1), 1)
        return x
