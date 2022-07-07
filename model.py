import functools

import torch.nn as nn


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out



class autoencoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', decoder_num = 2, decoder_out_ind = 100, feat=[6, 9, 10, 12], ensemble=False):
        super(autoencoder, self).__init__()
        self.ensemble = ensemble
        assert(n_blocks >= 0)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.feat = feat
        self.decoder_num = decoder_num
        self.encoder = nn.Sequential(*model)
        self.decoder_out_ind = decoder_out_ind  # ????
        if decoder_num == 1:
            mk_decoder = self.mk_decoder_224
        else:
            mk_decoder = self.mk_decoder
        for decoder_ind in range(decoder_num):
            setattr(self, 'decoder_{}'.format(decoder_ind), mk_decoder(n_downsampling, norm_layer, ngf, use_bias, output_nc))

    def mk_decoder(self, n_downsampling, norm_layer, ngf, use_bias, output_nc):
        model = []
        model += [
                  nn.Conv2d(256, 128, 3, 1, 1),
                  norm_layer(128),
                  nn.ReLU(True)]
        model += [nn.Conv2d(128, output_nc, kernel_size=3, padding=1)]
        model += [nn.Sigmoid()]
        model = nn.Sequential(*model)
        return model

    def mk_decoder_224(self, n_downsampling, norm_layer, ngf, use_bias, output_nc):
        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model += [nn.Sigmoid()]
        model = nn.Sequential(*model)
        return model

    def decoder_forw(self, x, decoder_ind):
        for ind, mm in enumerate(getattr(self, 'decoder_{}'.format(decoder_ind))):
            x = mm(x)
            if ind == self.decoder_out_ind:
                return x
        return x

    def forward(self, input):
        """Standard forward"""
        x = input
        features = []
        for ind, mm in enumerate(self.encoder):
            x = mm(x)
            if ind in self.feat:
                features.append(x)

        outs = []
        y = x.clone()

        if self.ensemble:
            for decoder_ind in range(self.decoder_num):
                for feat in features:
                    outs.append(self.decoder_forw(feat, decoder_ind))
            return outs, None, None

        else:
            for decoder_ind in range(self.decoder_num):
                outs.append(self.decoder_forw(x, decoder_ind))
            return outs, y, features

