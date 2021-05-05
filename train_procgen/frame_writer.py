from skvideo.io import FFmpegWriter
import torchvision.utils as vutils


def image_tensor_to_rgb_grid(image_tensor):
    """Converts an image tensor to a montage of images.

    Args:
        image_tensor (Tensor): tensor containing (possibly stacked) frames.
            Tensor values should be in [0, 1], and tensor shape should be […,
            n_frames*chans_per_frame, H, W]; the last three dimensions are
            essential, but the trailing dimensions do not matter.

    Returns:
         grid (Tensor): a [3*H*W] RGB image containing all the stacked frames
            passed in as input, arranged in a (roughly square) grid.
    """
    assert isinstance(image_tensor, th.Tensor)
    image_tensor = image_tensor.detach().cpu()

    # make sure shape is correct & data is in the right range
    assert image_tensor.ndim >= 3, image_tensor.shape
    assert th.all((-0.01 <= image_tensor) & (image_tensor <= 1.01)), \
        f"this only takes intensity values in [0,1], but range is " \
        f"[{image_tensor.min()}, {image_tensor.max()}]"
    n_chans = 3
    assert (image_tensor.shape[-3] % n_chans) == 0, \
        f"expected image to be stack of frames with {n_chans} channels " \
        f"each, but image tensor is of shape {image_tensor.shape}"

    # Reshape into [N,3,H,W] or [N,1,H,W], depending on how many channels there
    # are per frame.
    nchw_tensor = image_tensor.reshape((-1, n_chans) + image_tensor.shape[-2:])

    if n_chans == 1:
        # tile grayscale to RGB
        nchw_tensor = th.cat((nchw_tensor, ) * 3, dim=-3)

    # make sure it really is RGB
    assert nchw_tensor.ndim == 4 and nchw_tensor.shape[1] == 3

    # clamp to right value range
    clamp_tensor = th.clamp(nchw_tensor, 0, 1.)

    # number of rows scales with sqrt(num frames)
    # (this keeps image roughly square)
    nrow = max(1, int(math.sqrt(clamp_tensor.shape[0])))

    # now convert to an image grid
    grid = vutils.make_grid(clamp_tensor,
                            nrow=nrow,
                            normalize=False,
                            scale_each=False,
                            range=(0, 1))
    assert grid.ndim == 3 and grid.shape[0] == 3, grid.shape

    return grid


class TensorFrameWriter:
    """Writes N*(F*C)*H*W tensor frames to a video file."""
    def __init__(self, out_path, fps=25, config=None, adjust_axis=True, make_grid=True):
        self.out_path = out_path
        ffmpeg_out_config = {
            '-r': str(fps),
            '-vcodec': 'libx264',
            '-pix_fmt': 'yuv420p',
        }
        if config is not None:
            ffmpeg_out_config.update(config)

        self.writer = FFmpegWriter(out_path, outputdict=ffmpeg_out_config)
        self.adjust_axis = adjust_axis
        self.make_grid = make_grid

    def add_tensor(self, tensor):
        """Add a tensor of shape [..., C, H, W] representing the frame stacks
        for a single time step. Call this repeatedly for each time step you
        want to add."""
        if self.writer is None:
            raise RuntimeError("Cannot run add_tensor() again after closing!")
        grid = tensor
        if self.make_grid:
            grid = image_tensor_to_rgb_grid(tensor)
        np_grid = grid.numpy()
        if self.adjust_axis:
            # convert to (H, W, 3) numpy array
            np_grid = np_grid.transpose((1, 2, 0))
        byte_grid = (np_grid * 255).round().astype('uint8')
        self.writer.writeFrame(byte_grid)

    def __enter__(self):
        assert self.writer is not None, \
            "cannot __enter__ this again once it is closed"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None

    def __del__(self):
        self.close()
