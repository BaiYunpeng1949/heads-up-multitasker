import cv2


def write_video(filepath, fps, rgb_images, width, height):
    """
    Writes a video from images.
    Args:
      filepath: Path where the video will be saved.
      fps: the video writing fps.
      rgb_images: the rgb images that will be drawn into the video.
      width: the video viewport width.
      height: the video viewport height.
    Raises:
      ValueError: If frames per second (fps) is not set (set_fps is not called)
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, tuple([width, height]))
    for img in rgb_images:
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()
    print('\nThe video has been made and released to: {}.'.format(filepath))
