import os
from tensorboardX import SummaryWriter
import numpy as np


class Logger:
    """Logger that writes to TensorBoard and (optionally) to Weights & Biases.

    The wandb run is initialised lazily via :meth:`init_wandb`, so existing
    call sites that do not care about wandb keep working unchanged.
    """

    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

        self._wandb_run = None
        self._wandb = None

    def init_wandb(
        self,
        project: str,
        run_name: str,
        config: dict,
        entity: str = None,
        group: str = None,
        tags=None,
        mode: str = "online",
    ):
        """Initialise a wandb run that mirrors everything logged here.

        Safe to call only once per Logger instance.
        """
        try:
            import wandb
        except ImportError:
            print("[Logger] wandb not installed; skipping wandb init.")
            return None

        self._wandb = wandb
        self._wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
            dir=self._log_dir,
            mode=mode,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )
        return self._wandb_run

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)
        if self._wandb_run is not None:
            self._wandb_run.log({name: scalar}, step=int(step_))

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)
        if self._wandb_run is not None:
            prefixed = {f"{group_name}_{phase}/{k}": v for k, v in scalar_dict.items()}
            self._wandb_run.log(prefixed, step=int(step))

    def log_dict(self, data: dict, step: int, prefix: str = ""):
        """Log a flat dict of scalars under an optional prefix."""
        for k, v in data.items():
            name = f"{prefix}{k}" if prefix else k
            self.log_scalar(v, name, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)
        if self._wandb_run is not None:
            self._wandb_run.log(
                {name: self._wandb.Image(np.asarray(image).transpose(1, 2, 0))},
                step=int(step),
            )

    def log_video(self, video_frames, name, step, fps=10):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)
        if self._wandb_run is not None:
            video_frames_np = np.asarray(video_frames)
            if video_frames_np.dtype != np.uint8:
                video_frames_np = np.clip(video_frames_np, 0, 255).astype(np.uint8)
            self._wandb_run.log(
                {name: self._wandb.Video(video_frames_np[0], fps=fps, format="mp4")},
                step=int(step),
            )

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):

        videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]

        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        max_length = videos[0].shape[0]
        for i in range(max_videos_to_save):
            if videos[i].shape[0] > max_length:
                max_length = videos[i].shape[0]

        for i in range(max_videos_to_save):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)

        videos = np.stack(videos[:max_videos_to_save], 0)
        self.log_video(videos, video_title, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()

    def finish(self):
        try:
            self._summ_writer.close()
        except Exception:
            pass
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
