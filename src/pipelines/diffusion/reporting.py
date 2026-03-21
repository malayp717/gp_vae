"""Console formatting and persistent stats logs for diffusion models."""

from __future__ import annotations

from pathlib import Path

SEP = " | "
_LOG_SEP = " | "


def header_for(model_type: str) -> str:
    if model_type == "sccd":
        return (
            f"{'Epoch':>12s}{SEP}{'Phase':>6s}{SEP}{'Loss':>12s}{SEP}"
            f"{'Recon':>12s}{SEP}{'Bnd':>10s}{SEP}{'Rank':>10s}{SEP}"
            f"{'SigMu':>10s}{SEP}{'SigMin':>10s}{SEP}{'LR':>10s}{SEP}{'FID':>8s}{SEP}"
            f"{'SSIM':>6s}{SEP}{'PSNR':>7s}"
        )
    return (
        f"{'Epoch':>12s}{SEP}{'Phase':>6s}{SEP}{'Loss':>12s}{SEP}"
        f"{'Noise':>12s}{SEP}{'LR':>10s}{SEP}{'FID':>8s}"
    )


def divider_for(model_type: str) -> str:
    return "-" * len(header_for(model_type))


def _log_header_for(model_type: str) -> str:
    if model_type == "sccd":
        return (
            f"{'Model':<25s}{_LOG_SEP}{'Epoch':>10s}{_LOG_SEP}"
            f"{'Loss':>12s}{_LOG_SEP}{'Recon':>12s}{_LOG_SEP}"
            f"{'Bnd':>10s}{_LOG_SEP}{'Rank':>10s}{_LOG_SEP}"
            f"{'SigMu':>10s}{_LOG_SEP}{'SigMin':>10s}{_LOG_SEP}{'FID':>10s}{_LOG_SEP}{'SSIM':>8s}{_LOG_SEP}{'PSNR':>8s}"
        )
    return (
        f"{'Model':<25s}{_LOG_SEP}{'Epoch':>10s}{_LOG_SEP}"
        f"{'Loss':>12s}{_LOG_SEP}{'Noise':>12s}{_LOG_SEP}{'FID':>10s}"
    )


def _log_divider_for(model_type: str) -> str:
    return "-" * len(_log_header_for(model_type))


def fmt_row(
    model_type: str,
    epoch: int,
    total_epochs: int,
    phase: str,
    loss: float,
    recon: float,
    boundary_loss: float = 0.0,
    rank_penalty: float = 0.0,
    diffusion_loss: float = 0.0,
    sigma_mean: float = 0.0,
    sigma_min: float = 0.0,
    lr: float = 0.0,
    fid: float | None = None,
    ssim: float | None = None,
    psnr: float | None = None,
) -> str:
    fid_s = f"{fid:>8.2f}" if fid is not None else f"{'-':>8s}"
    if model_type == "sccd":
        ssim_s = f"{ssim:>6.4f}" if ssim is not None else f"{'-':>6s}"
        psnr_s = f"{psnr:>7.2f}" if psnr is not None else f"{'-':>7s}"
        return (
            f"{epoch + 1:>4d}/{total_epochs:<5d}  {SEP}{phase:>6s}{SEP}"
            f"{loss:>12.4f}{SEP}{recon:>12.4f}{SEP}{boundary_loss:>10.4f}{SEP}{rank_penalty:>10.4f}{SEP}"
            f"{sigma_mean:>10.4f}{SEP}{sigma_min:>10.4f}{SEP}{lr:>10.2e}{SEP}{fid_s}{SEP}{ssim_s}{SEP}{psnr_s}"
        )
    return (
        f"{epoch + 1:>4d}/{total_epochs:<5d}  {SEP}{phase:>6s}{SEP}"
        f"{loss:>12.4f}{SEP}{diffusion_loss:>12.4f}{SEP}{lr:>10.2e}{SEP}{fid_s}"
    )


def append_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
    fid: float | None = None,
    ssim: float | None = None,
    psnr: float | None = None,
) -> None:
    log_header = _log_header_for(model_type)
    log_divider = _log_divider_for(model_type)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    if not write_header:
        try:
            with open(log_path, "r", encoding="utf-8") as rf:
                write_header = not any(log_header in line for line in rf)
        except OSError:
            write_header = True

    with open(log_path, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(f"{log_divider}\n")
            fh.write(f"{log_header}\n")
            fh.write(f"{log_divider}\n")
        fid_s = f"{fid:>10.2f}" if fid is not None else f"{'-':>10s}"
        if model_type == "sccd":
            ssim_s = f"{ssim:>8.4f}" if ssim is not None else f"{'-':>8s}"
            psnr_s = f"{psnr:>8.2f}" if psnr is not None else f"{'-':>8s}"
            row = (
                f"{model_type:<25s}{_LOG_SEP}"
                f"{epoch + 1:>4d}/{total_epochs:<5d}{_LOG_SEP}"
                f"{metrics['loss']:>12.4f}{_LOG_SEP}"
                f"{metrics['recon_loss']:>12.4f}{_LOG_SEP}"
                f"{metrics.get('boundary_loss', 0.0):>10.4f}{_LOG_SEP}"
                f"{metrics.get('rank_penalty', 0.0):>10.4f}{_LOG_SEP}"
                f"{metrics.get('sigma_mean', 0.0):>10.4f}{_LOG_SEP}"
                f"{metrics.get('sigma_min', 0.0):>10.4f}{_LOG_SEP}"
                f"{fid_s}{_LOG_SEP}{ssim_s}{_LOG_SEP}{psnr_s}"
            )
        else:
            row = (
                f"{model_type:<25s}{_LOG_SEP}"
                f"{epoch + 1:>4d}/{total_epochs:<5d}{_LOG_SEP}"
                f"{metrics['loss']:>12.4f}{_LOG_SEP}"
                f"{metrics.get('diffusion_loss', metrics.get('noise_loss', 0.0)):>12.4f}{_LOG_SEP}"
                f"{fid_s}"
            )
        fh.write(f"{row}\n")


def append_train_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
) -> None:
    append_stats(log_path, model_type, epoch, total_epochs, train_metrics)


def append_val_stats(
    log_path: Path,
    model_type: str,
    epoch: int,
    total_epochs: int,
    val_metrics: dict[str, float],
    fid: float | None,
    ssim: float | None,
    psnr: float | None,
) -> None:
    append_stats(
        log_path,
        model_type,
        epoch,
        total_epochs,
        val_metrics,
        fid=fid,
        ssim=ssim,
        psnr=psnr,
    )
