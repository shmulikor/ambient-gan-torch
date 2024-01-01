# import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # with self.writer.as_default():
        #     tf.summary.scalar(tag, data=value, step=step)
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args::images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """
        # with self.writer.as_default():
        # imgs = None
        # for i, j in enumerate(images):
        #     img = ((j*0.5+0.5)*255).round().astype('uint8')
        #     if len(img.shape) == 3:
        #         img = img.transpose(1, 2, 0)
        #     else:
        #         img = img[:, :, np.newaxis]
        #     img = img[np.newaxis, :]
        #     if not imgs is None:
        #         imgs = np.append(imgs, img, axis=0)
        #     else:
        #         imgs = img
        
        # tf.summary.image('{}'.format(tag), imgs, max_outputs=len(imgs), step=step)
        # for i, _ in enumerate(images):
        #     images[i] = ((images[i] * 0.5 + 0.5) * 255).round().astype('uint8')
        
        # if len(images.shape) == 4:
        #     pass
        #     # TODO - deal with 3-channels images
        # else:
        #     images = np.expand_dims(images, axis=1)
        
        # self.writer.add_images(tag, images, step)
        self.writer.add_image(tag, images, step)

    # def histo_summary(self, tag, values, step, bins=1000):
    #     """Log a histogram of the tensor of values."""
    #     with self.writer.as_default():
    #         tf.summary.histogram('{}'.format(tag), values, buckets=bins, step=step)



