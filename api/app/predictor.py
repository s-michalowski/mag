import pickle
import time
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from app.queue_state import (
    QueueJob,
    QueueState,
    RunningJob
)


HOEFFDING_MODEL_PATH = "data/models/hoeffding_model.pkl"


@dataclass
class JobSimulationDetails:
    queue_position: int
    job_id: str
    owner_id: str
    worker_id: int
    model_name: str
    model_group: str
    predicted_start_offset_seconds: float
    predicted_duration_seconds: float
    predicted_end_offset_seconds: float


@dataclass
class OwnerEtaPrediction:
    owner_id: str
    active_jobs_count: int
    predicted_owner_finish_offset_seconds: float
    predicted_owner_finish_job_id: Optional[str]
    details: List[JobSimulationDetails]


class QueuePredictor:
    def __init__(self):
        self.hoeffding_model = self._load_model()

    @staticmethod
    def _load_model():
        with open(
            HOEFFDING_MODEL_PATH,
            "rb"
        ) as file:
            return pickle.load(file)

    def save(self) -> None:
        with open(
            HOEFFDING_MODEL_PATH,
            "wb"
        ) as file:
            pickle.dump(
                self.hoeffding_model,
                file
            )

    def update(
        self,
        row: pd.Series,
        y_true: float
    ) -> None:
        self.hoeffding_model.update(
            row=row,
            y_true=y_true
        )

    def predict_job_duration_from_features(
        self,
        model_group: str,
        pixel_count: float,
        compression_ratio: float,
        complexity: float
    ) -> float:
        row = pd.Series({
            "model_group": model_group,
            "pixel_count": pixel_count,
            "compression_ratio": compression_ratio,
            "complexity": complexity,
        })

        predicted_duration = float(
            self.hoeffding_model.predict(row)
        )

        return max(
            0.0,
            predicted_duration
        )

    def predict_job_duration(
        self,
        job: QueueJob
    ) -> float:
        return self.predict_job_duration_from_features(
            model_group=job.model_group,
            pixel_count=job.pixel_count,
            compression_ratio=job.compression_ratio,
            complexity=job.complexity
        )

    @staticmethod
    def predict_running_job_remaining(
        running_job: RunningJob,
        now: Optional[float] = None
    ) -> float:
        if now is None:
            now = time.time()

        predicted_remaining = (
            running_job.predicted_free_at - now
        )

        return max(
            0.0,
            float(predicted_remaining)
        )

    def predict_owner_eta(
        self,
        queue_state: QueueState,
        owner_id: str
    ) -> Optional[OwnerEtaPrediction]:
        queued_jobs = (
            queue_state.get_queued_jobs()
        )

        running_jobs = (
            queue_state.get_running_jobs()
        )

        owner_queued_jobs = [
            job
            for job in queued_jobs
            if job.owner_id == owner_id
        ]

        owner_running_jobs = [
            job
            for job in running_jobs
            if job.owner_id == owner_id
        ]

        if (
            not owner_queued_jobs
            and not owner_running_jobs
        ):
            return None

        worker_count = (
            queue_state.current_worker_count
        )

        if worker_count <= 0:
            raise ValueError(
                "Liczba workerów musi być większa od 0"
            )

        now = time.time()

        worker_availability = (
            queue_state.get_worker_availability_offsets(
                now=now
            )
        )

        details: List[JobSimulationDetails] = []

        owner_finish_offset = None
        owner_finish_job_id = None
        queue_position = 1

        # Najpierw uwzględniane są zadania,
        # które są już wykonywane przez workerów.
        sorted_running_jobs = sorted(
            running_jobs,
            key=lambda job: job.worker_id
        )

        for running_job in sorted_running_jobs:
            predicted_remaining = (
                self.predict_running_job_remaining(
                    running_job=running_job,
                    now=now
                )
            )

            predicted_start = 0.0
            predicted_end = predicted_remaining

            details.append(
                JobSimulationDetails(
                    queue_position=queue_position,
                    job_id=running_job.job_id,
                    owner_id=running_job.owner_id,
                    worker_id=running_job.worker_id,
                    model_name=running_job.model_name,
                    model_group=running_job.model_group,
                    predicted_start_offset_seconds=(
                        predicted_start
                    ),
                    predicted_duration_seconds=(
                        predicted_remaining
                    ),
                    predicted_end_offset_seconds=(
                        predicted_end
                    ),
                )
            )

            queue_position += 1

            if running_job.owner_id == owner_id:
                if (
                    owner_finish_offset is None
                    or predicted_end
                    > owner_finish_offset
                ):
                    owner_finish_offset = predicted_end
                    owner_finish_job_id = (
                        running_job.job_id
                    )

        # Następnie symulowane jest wykonanie
        # zadań oczekujących zgodnie z FIFO.
        for job in queued_jobs:
            worker_index = min(
                range(len(worker_availability)),
                key=worker_availability.__getitem__
            )

            predicted_start = (
                worker_availability[worker_index]
            )

            predicted_duration = (
                self.predict_job_duration(job)
            )

            predicted_end = (
                predicted_start
                + predicted_duration
            )

            worker_availability[worker_index] = (
                predicted_end
            )

            details.append(
                JobSimulationDetails(
                    queue_position=queue_position,
                    job_id=job.job_id,
                    owner_id=job.owner_id,
                    worker_id=worker_index + 1,
                    model_name=job.model_name,
                    model_group=job.model_group,
                    predicted_start_offset_seconds=float(
                        predicted_start
                    ),
                    predicted_duration_seconds=float(
                        predicted_duration
                    ),
                    predicted_end_offset_seconds=float(
                        predicted_end
                    ),
                )
            )

            queue_position += 1

            if job.owner_id == owner_id:
                if (
                    owner_finish_offset is None
                    or predicted_end
                    > owner_finish_offset
                ):
                    owner_finish_offset = predicted_end
                    owner_finish_job_id = job.job_id

        return OwnerEtaPrediction(
            owner_id=owner_id,
            active_jobs_count=(
                len(owner_queued_jobs)
                + len(owner_running_jobs)
            ),
            predicted_owner_finish_offset_seconds=float(
                owner_finish_offset
            ),
            predicted_owner_finish_job_id=(
                owner_finish_job_id
            ),
            details=details
        )