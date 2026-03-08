# RevGate CLI entry point.
# Commands: validate, classify, survival, sensitivity, download

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to sys.path when invoked directly (python main.py)
# pyproject.toml handles this automatically when installed via pip
sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio
from typing import Optional

import typer

app = typer.Typer(
    name="revgate",
    help="Empirical validation of the DNV-TC hypothesis via DepMap, TCGA, STRING and gnomAD.",
    no_args_is_help=True,
)


@app.command()
def validate(
    cancers: str = typer.Option(
        "LAML,SKCM,BRCA,KIRC,PAAD,LUAD",
        "--cancers",
        help="Comma-separated TCGA cancer type codes",
    ),
    top_n: int = typer.Option(20, "--top-n", help="Top dependency genes for NV-Score"),
    output: Path = typer.Option(Path("results/"), "--output", help="Output directory"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to settings.yaml"),
) -> None:
    """Run the full DNV-TC hypothesis validation pipeline."""
    from revgate.application.use_cases.validate_hypothesis import ValidationRequestDTO
    from revgate.config.settings import RevGateSettings
    from revgate.di.dependencies import Container

    settings = RevGateSettings.from_yaml(config) if config else RevGateSettings.from_yaml()
    container = Container.build(settings=settings)

    cancer_ids = [c.strip() for c in cancers.split(",")]

    request = ValidationRequestDTO(
        cancer_ids=cancer_ids,
        top_n=top_n,
        output_path=str(output),
    )

    result = container.validate_use_case.execute(request)

    if result.validation_result:
        typer.echo(result.validation_result.overall_summary())
    else:
        typer.echo("Pipeline failed. Errors:")
        for cancer_id, error in result.errors.items():
            typer.echo(f"  {cancer_id}: {error}")
        raise typer.Exit(code=1)


@app.command()
def classify(
    cancers: str = typer.Option(
        "LAML,SKCM",
        "--cancers",
        help="Comma-separated TCGA cancer type codes",
    ),
    top_n: int = typer.Option(20, "--top-n", help="Top dependency genes for NV-Score"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to settings.yaml"),
) -> None:
    """Run 4-axis classification only (no survival analysis)."""
    from revgate.application.dto.classification_dto import ClassificationRequestDTO
    from revgate.config.settings import RevGateSettings
    from revgate.di.dependencies import Container

    settings = RevGateSettings.from_yaml(config) if config else RevGateSettings.from_yaml()
    container = Container.build(settings=settings)

    cancer_ids = [c.strip() for c in cancers.split(",")]
    request = ClassificationRequestDTO(cancer_ids=cancer_ids, top_n=top_n)
    result = container.classify_use_case.execute(request)

    for axis in result.axes:
        typer.echo(
            f"{axis.cancer_id}: NV={axis.nv_class} | RDP={axis.rdp_class} | "
            f"CEP={axis.cep_class} | MP={axis.mp_class} | "
            f"NV-score={axis.nv_composite_score:.4f}"
        )

    if result.errors:
        typer.echo("Errors:")
        for cancer_id, error in result.errors.items():
            typer.echo(f"  {cancer_id}: {error}")


@app.command()
def download(
    source: str = typer.Option(
        "all",
        "--source",
        help="Data source: depmap | string | gnomad | all",
    ),
    cancer: Optional[str] = typer.Option(
        None,
        "--cancer",
        help="TCGA cancer code for TCGA downloads (e.g. SKCM)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download even if already cached",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to settings.yaml"),
) -> None:
    """Download and cache required data sources.

    Sources: depmap, string, gnomad, all.
    TCGA data requires manual download -- see instructions printed by --cancer flag.

    Examples:
        revgate download --source all
        revgate download --source depmap
        revgate download --source gnomad
        revgate download --cancer SKCM
    """
    from revgate.config.settings import RevGateSettings
    from revgate.di.dependencies import Container
    from revgate.infrastructure.external.downloader import (
        Downloader,
        SOURCE_GROUPS,
        DATA_SOURCES,
    )

    settings = RevGateSettings.from_yaml(config) if config else RevGateSettings.from_yaml()
    container = Container.build(settings=settings)
    downloader = Downloader(cache=container.cache)

    # TCGA download -- separate path
    if cancer is not None:
        typer.echo(f"TCGA download: {cancer.upper()}")
        try:
            downloader.download_tcga(cancer.upper(), force=force)
        except Exception as exc:
            typer.echo(str(exc))
        return

    # Validate source group
    known_groups = list(SOURCE_GROUPS.keys())
    if source not in known_groups:
        typer.echo(f"Unknown source: {source!r}")
        typer.echo(f"Known sources: {', '.join(known_groups)}")
        raise typer.Exit(code=1)

    # Print cache status before downloading
    typer.echo("Cache status:")
    status = downloader.status()
    for source_id, cached in status.items():
        marker = "OK" if cached else "MISSING"
        typer.echo(f"  [{marker}] {source_id}: {DATA_SOURCES[source_id]['cache_key']}")

    typer.echo(f"\nDownloading source group: {source}")

    try:
        paths = downloader.download_group(source, force=force)
        typer.echo(f"\nDownloaded {len(paths)} file(s).")
    except Exception as exc:
        typer.echo(f"\nDownload failed: {exc}")
        raise typer.Exit(code=1)


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to settings.yaml"),
) -> None:
    """Show cache status for all data sources."""
    from revgate.config.settings import RevGateSettings
    from revgate.di.dependencies import Container
    from revgate.infrastructure.external.downloader import Downloader, DATA_SOURCES

    settings = RevGateSettings.from_yaml(config) if config else RevGateSettings.from_yaml()
    container = Container.build(settings=settings)
    downloader = Downloader(cache=container.cache)

    typer.echo(f"Cache directory: {settings.cache_dir}")
    typer.echo("")

    cache_status = downloader.status()
    all_cached = all(cache_status.values())

    for source_id, cached in cache_status.items():
        marker = "OK" if cached else "MISSING"
        desc = DATA_SOURCES[source_id]["description"]
        typer.echo(f"  [{marker}] {source_id}: {desc}")

    typer.echo("")
    if all_cached:
        typer.echo("All data sources cached. Pipeline ready.")
    else:
        missing = [sid for sid, cached in cache_status.items() if not cached]
        typer.echo(f"Missing {len(missing)} source(s). Run: revgate download --source all")


if __name__ == "__main__":
    app()
