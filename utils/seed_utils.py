import  torch

from lightning import seed_everything
def set_seed(seed: int) -> None:
    r"""

    :param seed:
    :return:
    """
    seed_everything(seed, workers=True)
    # 保证更严格的可复现性（稍微影响运行性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False