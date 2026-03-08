# Dependency injection -- wires all layers together.
# Single point of assembly: repositories -> use cases -> pipeline.

from __future__ import annotations

from pathlib import Path

from revgate.application.use_cases.analyze_survival import AnalyzeSurvivalUseCase
from revgate.application.use_cases.classify_tumors import ClassifyTumorUseCase
from revgate.application.use_cases.run_sensitivity import RunSensitivityUseCase
from revgate.application.use_cases.validate_hypothesis import ValidateHypothesisUseCase
from revgate.config.settings import RevGateSettings
from revgate.infrastructure.external.file_cache import FileCache
from revgate.infrastructure.persistence.repositories.depmap_repository import DepMapRepository
from revgate.infrastructure.persistence.repositories.gnomad_repository import GnomADRepository
from revgate.infrastructure.persistence.repositories.string_repository import STRINGRepository
from revgate.infrastructure.persistence.repositories.tcga_repository import TCGARepository
from revgate.infrastructure.processing.pipeline import PipelineOrchestrator
from revgate.infrastructure.processing.stages.classification_stage import ClassificationStage
from revgate.infrastructure.processing.stages.data_loading_stage import DataLoadingStage
from revgate.infrastructure.processing.stages.hypothesis_stage import HypothesisStage
from revgate.infrastructure.processing.stages.sensitivity_stage import SensitivityStage


class Container:
    """Dependency injection container.

    Assembles the full object graph:
        Settings -> Cache -> Repositories -> Use Cases -> Pipeline

    Usage:
        container = Container.build()
        pipeline  = container.pipeline
        result    = await pipeline.run(context)
    """

    def __init__(self, settings: RevGateSettings) -> None:
        self._settings = settings
        self._cache: FileCache | None = None
        self._depmap_repo: DepMapRepository | None = None
        self._tcga_repo: TCGARepository | None = None
        self._string_repo: STRINGRepository | None = None
        self._gnomad_repo: GnomADRepository | None = None
        self._classify_use_case: ClassifyTumorUseCase | None = None
        self._analyze_use_case: AnalyzeSurvivalUseCase | None = None
        self._sensitivity_use_case: RunSensitivityUseCase | None = None
        self._validate_use_case: ValidateHypothesisUseCase | None = None
        self._pipeline: PipelineOrchestrator | None = None

    @classmethod
    def build(cls, settings: RevGateSettings | None = None) -> "Container":
        """Build container with default or provided settings.

        Args:
            settings: RevGateSettings instance. Loads from YAML if None.

        Returns:
            Fully assembled Container.
        """
        if settings is None:
            settings = RevGateSettings.from_yaml()
        return cls(settings)

    @property
    def cache(self) -> FileCache:
        if self._cache is None:
            self._cache = FileCache(cache_dir=self._settings.cache_dir)
        return self._cache

    @property
    def depmap_repo(self) -> DepMapRepository:
        if self._depmap_repo is None:
            self._depmap_repo = DepMapRepository(cache=self.cache)
        return self._depmap_repo

    @property
    def tcga_repo(self) -> TCGARepository:
        if self._tcga_repo is None:
            self._tcga_repo = TCGARepository(cache=self.cache)
        return self._tcga_repo

    @property
    def string_repo(self) -> STRINGRepository:
        if self._string_repo is None:
            self._string_repo = STRINGRepository(cache=self.cache)
        return self._string_repo

    @property
    def gnomad_repo(self) -> GnomADRepository:
        if self._gnomad_repo is None:
            self._gnomad_repo = GnomADRepository(cache=self.cache)
        return self._gnomad_repo

    @property
    def classify_use_case(self) -> ClassifyTumorUseCase:
        if self._classify_use_case is None:
            self._classify_use_case = ClassifyTumorUseCase(
                dependency_repo=self.depmap_repo,
                clinical_repo=self.tcga_repo,
                network_repo=self.string_repo,
                constraint_repo=self.gnomad_repo,
            )
        return self._classify_use_case

    @property
    def analyze_use_case(self) -> AnalyzeSurvivalUseCase:
        if self._analyze_use_case is None:
            self._analyze_use_case = AnalyzeSurvivalUseCase(
                clinical_repo=self.tcga_repo,
            )
        return self._analyze_use_case

    @property
    def sensitivity_use_case(self) -> RunSensitivityUseCase:
        if self._sensitivity_use_case is None:
            self._sensitivity_use_case = RunSensitivityUseCase(
                classify_use_case=self.classify_use_case,
            )
        return self._sensitivity_use_case

    @property
    def validate_use_case(self) -> ValidateHypothesisUseCase:
        if self._validate_use_case is None:
            self._validate_use_case = ValidateHypothesisUseCase(
                classify_use_case=self.classify_use_case,
                analyze_survival_use_case=self.analyze_use_case,
            )
        return self._validate_use_case

    @property
    def pipeline(self) -> PipelineOrchestrator:
        if self._pipeline is None:
            self._pipeline = PipelineOrchestrator(
                stages=[
                    DataLoadingStage(cache=self.cache),
                    ClassificationStage(classify_use_case=self.classify_use_case),
                    HypothesisStage(analyze_use_case=self.analyze_use_case),
                    SensitivityStage(sensitivity_use_case=self.sensitivity_use_case),
                ],
                stop_on_failure=True,
            )
        return self._pipeline
