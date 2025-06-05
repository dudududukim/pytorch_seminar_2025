from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Create a tensorboard logger to log_dir."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_images(tag, images, step)

    def histo_summary(self, tag, values, step, bins='tensorflow', walltime=None, max_bins=None):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(
            tag, values, global_step=step, bins=bins, walltime=walltime, max_bins=max_bins
        )
        self.writer.flush()  # Explicit flush to ensure data is written