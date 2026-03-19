"""Console/table formatting and persistent stats logs."""

from __future__ import annotations

from pathlib import Path

SEP = " | "
HEADER = (
    f"{'Epoch':>12s}{SEP}{'Phase':>6s}{SEP}{'Loss':>12s}{SEP}"
    f"{'Recon':>12s}{SEP}{'KL':>12s}{SEP}{'LPIPS':>10s}{SEP}"
    f"{'Adv_G':>10s}{SEP}{'D_loss':>10s}{SEP}"
    f"{'Beta':>8s}{SEP}{'LR':>10s}{SEP}{'FID':>8s}{SEP}"
    f"{'SSIM':>6s}{SEP}{'PSNR':>7s}"
)
DIVIDER = "-" * len(HEADER)

_STATS_LOG_SEP = " | "
_STATS_LOG_HEADER = (
    f"{'Model':<25s}{_STATS_LOG_SEP}{'Epoch':>10s}{_STATS_LOG_SEP}"
    f"{'Loss':>12s}{_STATS_LOG_SEP}{'Recon':>12s}{_STATS_LOG_SEP}"
    f"{'KL':>12s}{_STATS_LOG_SEP}{'LPIPS':>10s}{_STATS_LOG_SEP}"
    f"{'Adv_G':>10s}{_STATS_LOG_SEP}{'D_loss':>10s}{_STATS_LOG_SEP}"
    f"{'Beta':>8s}{_STATS_LOG_SEP}{'FID':>10s}{_STATS_LOG_SEP}"
    f"{'SSIM':>8s}{_STATS_LOG_SEP}{'PSNR':>8s}"
)
_STATS_LOG_DIVIDER = "-" * len(_STATS_LOG_HEADER)


def fmt_row(
    epoch: int,
    total_epochs: int,
    phase: str,
    loss: float,
    recon: float,
    kl: float,
    lpips_val: float = 0.0,
    adv_g: float = 0.0,
    d_loss: float = 0.0,
    beta: float = 0.0,
    lr: float = 0.0,
    fid: float | None = None,
    ssim: float | None = None,
    psnr: float | None = None,
) -> str:
    fid_s = f"{fid:>8.2f}" if fid is not None else f"{'-':>8s}"
    ssim_s = f"{ssim:>6.4f}" if ssim is not None else f"{'-':>6s}"
    psnr_s = f"{psnr:>7.2f}" if psnr is not None else f"{'-':>7s}"
    return (
        f"{epoch + 1:>4d}/{total_epochs:<5d}  {SEP}{phase:>6s}{SEP}"
        f"{loss:>12.4f}{SEP}{recon:>12.4f}{SEP}{kl:>12.4f}{SEP}"
        f"{lpips_val:>10.4f}{SEP}{adv_g:>10.4f}{SEP}{d_loss:>10.4f}{SEP}"
        f"{beta:>8.4f}{SEP}{lr:>10.2e}{SEP}"
        f"{fid_s}{SEP}{ssim_s}{SEP}{psnr_s}"
    )


def append_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
    beta: float,
    fid: float | None = None,
    ssim: float | None = None,
    psnr: float | None = None,
) -> None:
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    if not write_header:
        try:
            with open(log_path, "r", encoding="utf-8") as rf:
                write_header = not any(_STATS_LOG_HEADER in line for line in rf)
        except OSError:
            write_header = True
    with open(log_path, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(f"{_STATS_LOG_DIVIDER}\n")
            fh.write(f"{_STATS_LOG_HEADER}\n")
            fh.write(f"{_STATS_LOG_DIVIDER}\n")
        fid_s = f"{fid:>10.2f}" if fid is not None else f"{'-':>10s}"
        ssim_s = f"{ssim:>8.4f}" if ssim is not None else f"{'-':>8s}"
        psnr_s = f"{psnr:>8.2f}" if psnr is not None else f"{'-':>8s}"
        row = (
            f"{model_type:<25s}{_STATS_LOG_SEP}"
            f"{epoch + 1:>4d}/{total_epochs:<5d}{_STATS_LOG_SEP}"
            f"{metrics['loss']:>12.4f}{_STATS_LOG_SEP}"
            f"{metrics['recon_loss']:>12.4f}{_STATS_LOG_SEP}"
            f"{metrics['kl_loss']:>12.4f}{_STATS_LOG_SEP}"
            f"{metrics.get('lpips_loss', 0.0):>10.4f}{_STATS_LOG_SEP}"
            f"{metrics.get('adv_g_loss', 0.0):>10.4f}{_STATS_LOG_SEP}"
            f"{metrics.get('d_loss', 0.0):>10.4f}{_STATS_LOG_SEP}"
            f"{beta:>8.4f}{_STATS_LOG_SEP}"
            f"{fid_s}{_STATS_LOG_SEP}"
            f"{ssim_s}{_STATS_LOG_SEP}"
            f"{psnr_s}"
        )
        fh.write(f"{row}\n")


def append_train_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    beta: float,
) -> None:
    append_stats(log_path, model_type, epoch, total_epochs, train_metrics, beta=beta)


def append_val_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    val_metrics: dict[str, float],
    beta: float,
    fid: float,
    ssim: float,
    psnr: float,
) -> None:
    append_stats(
        log_path,
        model_type,
        epoch,
        total_epochs,
        val_metrics,
        beta=beta,
        fid=fid,
        ssim=ssim,
        psnr=psnr,
    )

