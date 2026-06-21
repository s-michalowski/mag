import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from app.model_grouping import get_model_group


@dataclass
class QueueJob:
    job_id: str
    owner_id: str
    created_at: str
    model_name: str
    model_group: str
    pixel_count: float
    compression_ratio: float
    complexity: float


@dataclass
class IncomingJob:
    job_id: str
    owner_id: str
    created_at: str
    model_name: str
    pixel_count: float
    compression_ratio: float
    complexity: float


@dataclass
class RunningJob:
    worker_id: int
    job_id: str
    owner_id: str
    created_at: str
    model_name: str
    model_group: str
    pixel_count: float
    compression_ratio: float
    predicted_free_at: float
    complexity: float


class QueueState:
    def __init__(self):
        self.queued_jobs: List[QueueJob] = []

        self.running_jobs: Dict[
            str,
            RunningJob
        ] = {}

        self.current_worker_count: int = 0

    def reset(self) -> None:
        self.queued_jobs = []
        self.running_jobs = {}
        self.current_worker_count = 0

    def set_worker_count(
        self,
        worker_count: int
    ) -> None:
        if worker_count < 0:
            raise ValueError(
                "worker_count nie może być ujemne"
            )

        self.current_worker_count = (
            worker_count
        )

    def create_queue_job(
        self,
        incoming_job: IncomingJob
    ) -> QueueJob:
        model_group = get_model_group(
            incoming_job.model_name
        )

        return QueueJob(
            job_id=incoming_job.job_id,
            owner_id=incoming_job.owner_id,
            created_at=incoming_job.created_at,
            model_name=incoming_job.model_name,
            model_group=model_group,
            pixel_count=incoming_job.pixel_count,
            compression_ratio=(
                incoming_job.compression_ratio
            ),
            complexity=incoming_job.complexity
        )

    def add_jobs(
        self,
        jobs: List[IncomingJob]
    ) -> None:
        queue_jobs = [
            self.create_queue_job(job)
            for job in jobs
        ]

        self.queued_jobs.extend(
            queue_jobs
        )

    def pop_next_queued_job(
        self
    ) -> Optional[QueueJob]:
        if not self.queued_jobs:
            return None

        return self.queued_jobs.pop(0)

    def add_running_job(
        self,
        job: QueueJob,
        worker_id: int,
        predicted_free_at: float
    ) -> RunningJob:
        running_job = RunningJob(
            worker_id=worker_id,
            job_id=job.job_id,
            owner_id=job.owner_id,
            created_at=job.created_at,
            model_name=job.model_name,
            model_group=job.model_group,
            pixel_count=job.pixel_count,
            compression_ratio=(
                job.compression_ratio
            ),
            predicted_free_at=predicted_free_at,
            complexity=job.complexity
        )

        self.running_jobs[job.job_id] = (
            running_job
        )

        return running_job

    def remove_running_job(
        self,
        job_id: str
    ) -> Optional[RunningJob]:
        return self.running_jobs.pop(
            job_id,
            None
        )

    def get_running_job_by_id(
        self,
        job_id: str
    ) -> Optional[RunningJob]:
        return self.running_jobs.get(
            job_id
        )

    def get_queued_job_by_id(
        self,
        job_id: str
    ) -> Optional[QueueJob]:
        for job in self.queued_jobs:
            if job.job_id == job_id:
                return job

        return None

    def get_job_by_id(
        self,
        job_id: str
    ):
        running_job = (
            self.get_running_job_by_id(
                job_id
            )
        )

        if running_job is not None:
            return running_job

        return self.get_queued_job_by_id(
            job_id
        )

    def has_job(
        self,
        job_id: str
    ) -> bool:
        return (
            self.get_job_by_id(job_id)
            is not None
        )

    def remove_job(
        self,
        job_id: str
    ):
        running_job = (
            self.remove_running_job(
                job_id
            )
        )

        if running_job is not None:
            return running_job

        for index, job in enumerate(
            self.queued_jobs
        ):
            if job.job_id == job_id:
                return self.queued_jobs.pop(
                    index
                )

        return None

    def get_queued_jobs(
        self
    ) -> List[QueueJob]:
        return list(
            self.queued_jobs
        )

    def get_running_jobs(
        self
    ) -> List[RunningJob]:
        return list(
            self.running_jobs.values()
        )

    def get_all_owner_ids(
        self
    ) -> List[str]:
        owner_ids = {
            job.owner_id
            for job in self.queued_jobs
        }

        owner_ids.update(
            job.owner_id
            for job
            in self.running_jobs.values()
        )

        return sorted(owner_ids)

    def get_worker_availability_offsets(
        self,
        now: Optional[float] = None
    ) -> List[float]:
        if now is None:
            now = time.time()

        offsets = []

        for worker_id in range(
            1,
            self.current_worker_count + 1
        ):
            worker_running_jobs = [
                job
                for job
                in self.running_jobs.values()
                if job.worker_id == worker_id
            ]

            if not worker_running_jobs:
                offsets.append(0.0)
                continue

            running_job = (
                worker_running_jobs[0]
            )

            remaining = max(
                0.0,
                float(
                    running_job.predicted_free_at
                    - now
                )
            )

            offsets.append(remaining)

        return offsets

    def queue_size(self) -> int:
        return len(
            self.queued_jobs
        )

    def running_size(self) -> int:
        return len(
            self.running_jobs
        )

    def to_dict(self) -> dict:
        return {
            "current_worker_count": (
                self.current_worker_count
            ),
            "queue_size": self.queue_size(),
            "running_size": self.running_size(),
            "worker_offsets": (
                self.get_worker_availability_offsets()
            ),
            "queued_jobs": [
                asdict(job)
                for job in self.queued_jobs
            ],
            "running_jobs": [
                asdict(job)
                for job
                in self.running_jobs.values()
            ],
        }