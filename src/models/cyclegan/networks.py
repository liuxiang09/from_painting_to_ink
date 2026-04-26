from __future__ import annotations

import functools

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(norm_type: str = "instance"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda _: Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    def init_func(m: nn.Module):
        classname = m.__class__.__name__
        has_weight = hasattr(m, "weight") and m.weight is not None
        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            if has_weight:
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            if has_weight:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net: nn.Module, device: torch.device, init_type: str = "normal", init_gain: float = 0.02) -> nn.Module:
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan", target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"gan mode [{gan_mode}] not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 8,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()

        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


def define_G(
    input_nc: int,
    output_nc: int,
    ngf: int,
    netG: str,
    norm: str,
    use_dropout: bool,
    init_type: str,
    init_gain: float,
    device: torch.device,
) -> nn.Module:
    norm_layer = get_norm_layer(norm)
    if netG == "unet_256":
        net = UNetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_128":
        net = UNetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError(f"generator model [{netG}] is not recognized")
    return init_net(net, device, init_type, init_gain)


def define_D(
    input_nc: int,
    ndf: int,
    netD: str,
    n_layers_D: int,
    norm: str,
    init_type: str,
    init_gain: float,
    device: torch.device,
) -> nn.Module:
    norm_layer = get_norm_layer(norm)
    if netD == "basic":
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f"discriminator model [{netD}] is not recognized")
    return init_net(net, device, init_type, init_gain)
