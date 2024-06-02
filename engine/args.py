import argparse


def get_shell_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--noise-scheduler",
        choices=["DDPMScheduler", "DDIMScheduler", "EulerDiscreteScheduler"],
        type=str,
        help="Name of the noise scheduler",
        default="DDPMScheduler",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        help="Number of the denoising steps",
        default=1000,
    )
    return parser.parse_args()
