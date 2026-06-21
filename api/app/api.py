import time
from threading import Lock
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.predictor import QueuePredictor
from app.queue_state import IncomingJob, QueueJob, QueueState, RunningJob


app = FastAPI(title="Queue Prediction API")

queue_state = QueueState()
predictor = QueuePredictor()
state_lock = Lock()

owner_eta_cache = {}
completed_jobs_buffer = []

AUTO_UPDATE_MIN_RECORDS = 10
AUTO_UPDATE_INTERVAL_SECONDS = 300  # 5 minut
last_model_update_time = time.time()



# MODELE WEJŚCIA / WYJŚCIA
class SetWorkersRequest(BaseModel):
    worker_count: int = Field(..., ge=0)


class AddJobItem(BaseModel):
    job_id: str
    owner_id: str
    created_at: str
    model_name: str
    pixel_count: float
    compression_ratio: float
    complexity: float


class AddBatchRequest(BaseModel):
    jobs: List[AddJobItem]


class JobFinishedRequest(BaseModel):
    job_id: str
    real_duration_seconds: float = Field(..., ge=0)
    status: str = "NEW"


class ModelUpdateRequest(BaseModel):
    clear_buffer_after_update: bool = True


class PredictionDetailItem(BaseModel):
    owner_id: str
    job_id: str
    model_group: str
    predicted_duration_seconds: float


class OwnerEtaResponse(BaseModel):
    owner_id: str
    active_jobs_count: int
    predicted_owner_finish_offset_seconds: float
    predicted_owner_finish_job_id: Optional[str]
    details: List[PredictionDetailItem] = Field(default_factory=list)



# FUNKCJE POMOCNICZE
def recompute_all_owner_predictions() -> None:
    global owner_eta_cache

    new_cache = {}

    for owner_id in queue_state.get_all_owner_ids():
        prediction = predictor.predict_owner_eta(
            queue_state=queue_state,
            owner_id=owner_id
        )

        if prediction is None:
            new_cache[owner_id] = {
                "owner_id": owner_id,
                "active_jobs_count": 0,
                "predicted_owner_finish_offset_seconds": 0.0,
                "predicted_owner_finish_job_id": None,
                "details": [],
            }
            continue

        details = [
            {
                "owner_id": item.owner_id,
                "job_id": item.job_id,
                "model_group": item.model_group,
                "predicted_duration_seconds": (
                    item.predicted_duration_seconds
                ),
            }
            for item in prediction.details
        ]

        new_cache[owner_id] = {
            "owner_id": prediction.owner_id,
            "active_jobs_count": prediction.active_jobs_count,
            "predicted_owner_finish_offset_seconds": (
                prediction.predicted_owner_finish_offset_seconds
            ),
            "predicted_owner_finish_job_id": (
                prediction.predicted_owner_finish_job_id
            ),
            "details": details,
        }

    owner_eta_cache = new_cache


def register_finished_job(
    job_id: str,
    real_duration_seconds: float,
    status: str
):
    finished_job = queue_state.remove_job(job_id)

    if finished_job is not None:
        completed_jobs_buffer.append({
            "job_id": finished_job.job_id,
            "owner_id": finished_job.owner_id,
            "model_name": finished_job.model_name,
            "model_group": finished_job.model_group,
            "pixel_count": finished_job.pixel_count,
            "compression_ratio": finished_job.compression_ratio,
            "complexity": finished_job.complexity,
            "real_duration_seconds": real_duration_seconds,
            "status": status,
        })

    return finished_job


def should_auto_update_model() -> bool:
    if not completed_jobs_buffer:
        return False

    if len(completed_jobs_buffer) >= AUTO_UPDATE_MIN_RECORDS:
        return True

    time_since_last_update = time.time() - last_model_update_time

    return time_since_last_update >= AUTO_UPDATE_INTERVAL_SECONDS


def run_model_update(
    clear_buffer_after_update: bool = True
) -> dict:
    global completed_jobs_buffer, last_model_update_time

    if not completed_jobs_buffer:
        return {
            "updated_records": 0,
            "skipped_records": 0,
        }

    updated_records = 0
    skipped_records = 0

    for item in completed_jobs_buffer:
        if str(item.get("status", "NEW")).upper() != "NEW":
            skipped_records += 1
            continue

        row = pd.Series({
            "model_group": item["model_group"],
            "pixel_count": item["pixel_count"],
            "compression_ratio": item["compression_ratio"],
            "complexity": item["complexity"],
        })

        predictor.update(
            row=row,
            y_true=float(item["real_duration_seconds"])
        )

        updated_records += 1

    if updated_records > 0:
        predictor.save()

    if clear_buffer_after_update:
        completed_jobs_buffer = []

    last_model_update_time = time.time()

    return {
        "updated_records": updated_records,
        "skipped_records": skipped_records,
    }


def get_busy_worker_ids() -> set[int]:
    return {
        job.worker_id
        for job in queue_state.get_running_jobs()
    }


def get_idle_worker_ids() -> List[int]:
    busy_worker_ids = get_busy_worker_ids()

    return [
        worker_id
        for worker_id in range(
            1,
            queue_state.current_worker_count + 1
        )
        if worker_id not in busy_worker_ids
    ]


def assign_job_to_worker(
    job: QueueJob,
    worker_id: int
):
    predicted_duration = predictor.predict_job_duration(job)

    predicted_free_at = (
        time.time() + predicted_duration
    )

    return queue_state.add_running_job(
        job=job,
        worker_id=worker_id,
        predicted_free_at=predicted_free_at
    )


def dispatch_jobs_to_idle_workers() -> None:
    if queue_state.current_worker_count <= 0:
        return

    idle_worker_ids = get_idle_worker_ids()

    for worker_id in idle_worker_ids:
        next_job = queue_state.pop_next_queued_job()

        if next_job is None:
            break

        assign_job_to_worker(
            job=next_job,
            worker_id=worker_id
        )


def refresh_schedule_and_predictions() -> None:
    dispatch_jobs_to_idle_workers()
    recompute_all_owner_predictions()



# ENDPOINTY
@app.get("/")
def root():
    return {
        "message": "API działa"
    }


@app.post("/reset")
def reset_state():
    global owner_eta_cache
    global completed_jobs_buffer
    global last_model_update_time

    with state_lock:
        queue_state.reset()

        owner_eta_cache = {}
        completed_jobs_buffer = []
        last_model_update_time = time.time()

        return {
            "message": "State reset",
            "queue_size": queue_state.queue_size(),
            "running_size": queue_state.running_size(),
            "worker_count": queue_state.current_worker_count,
            "completed_jobs_buffer_size": (
                len(completed_jobs_buffer)
            ),
        }


@app.post("/workers/set")
def set_workers(request: SetWorkersRequest):
    with state_lock:
        queue_state.set_worker_count(
            request.worker_count
        )

        refresh_schedule_and_predictions()

        return {
            "message": "Worker count updated",
            "worker_count": queue_state.current_worker_count,
            "queue_size": queue_state.queue_size(),
            "running_size": queue_state.running_size(),
            "owners_in_cache": len(owner_eta_cache),
        }


@app.post("/queue/add-batch")
def add_batch(request: AddBatchRequest):
    with state_lock:
        incoming_jobs = []
        duplicate_job_ids = []

        for job in request.jobs:
            if queue_state.has_job(job.job_id):
                duplicate_job_ids.append(job.job_id)
                continue

            incoming_jobs.append(
                IncomingJob(
                    job_id=job.job_id,
                    owner_id=job.owner_id,
                    created_at=job.created_at,
                    model_name=job.model_name,
                    pixel_count=job.pixel_count,
                    compression_ratio=job.compression_ratio,
                    complexity=job.complexity,
                )
            )

        queue_state.add_jobs(incoming_jobs)
        refresh_schedule_and_predictions()

        return {
            "message": "Batch added",
            "received_jobs": len(request.jobs),
            "added_jobs": len(incoming_jobs),
            "duplicate_jobs_skipped": len(
                duplicate_job_ids
            ),
            "duplicate_job_ids": duplicate_job_ids,
            "queue_size": queue_state.queue_size(),
            "running_size": queue_state.running_size(),
            "owners_in_cache": len(owner_eta_cache),
        }


@app.post("/queue/job-finished")
def job_finished(request: JobFinishedRequest):
    with state_lock:
        finished_job = register_finished_job(
            job_id=request.job_id,
            real_duration_seconds=(
                request.real_duration_seconds
            ),
            status=request.status
        )

        # Jeżeli zakończyło się zadanie wykonywane przez workera,
        # worker może od razu otrzymać kolejne zadanie z kolejki.
        if isinstance(finished_job, RunningJob):
            worker_is_still_active = (
                finished_job.worker_id
                <= queue_state.current_worker_count
            )

            if worker_is_still_active:
                next_job = (
                    queue_state.pop_next_queued_job()
                )

                if next_job is not None:
                    assign_job_to_worker(
                        job=next_job,
                        worker_id=finished_job.worker_id
                    )

        model_auto_updated = False
        updated_records = 0
        skipped_records = 0

        if should_auto_update_model():
            update_result = run_model_update(
                clear_buffer_after_update=True
            )

            model_auto_updated = True
            updated_records = (
                update_result["updated_records"]
            )
            skipped_records = (
                update_result["skipped_records"]
            )

        recompute_all_owner_predictions()

        return {
            "message": "Job finished registered",
            "job_found": finished_job is not None,
            "job_id": request.job_id,
            "real_duration_seconds": (
                request.real_duration_seconds
            ),
            "status": request.status,
            "finished_job_was_running": isinstance(
                finished_job,
                RunningJob
            ),
            "queue_size": queue_state.queue_size(),
            "running_size": queue_state.running_size(),
            "owners_in_cache": len(owner_eta_cache),
            "completed_jobs_buffer_size": (
                len(completed_jobs_buffer)
            ),
            "model_auto_updated": model_auto_updated,
            "updated_records": updated_records,
            "skipped_records": skipped_records,
        }


@app.post("/models/update")
def update_model(request: ModelUpdateRequest):
    with state_lock:
        result = run_model_update(
            clear_buffer_after_update=(
                request.clear_buffer_after_update
            )
        )

        recompute_all_owner_predictions()

        return {
            "message": "Model updated",
            "updated_records": result["updated_records"],
            "skipped_records": result["skipped_records"],
            "completed_jobs_buffer_size": (
                len(completed_jobs_buffer)
            ),
            "owners_in_cache": len(owner_eta_cache),
        }


@app.get(
    "/predict/owner/{owner_id}",
    response_model=OwnerEtaResponse
)
def predict_owner(owner_id: str):
    with state_lock:
        cached_prediction = owner_eta_cache.get(
            owner_id
        )

        if cached_prediction is None:
            return {
                "owner_id": owner_id,
                "active_jobs_count": 0,
                "predicted_owner_finish_offset_seconds": 0.0,
                "predicted_owner_finish_job_id": None,
                "details": [],
            }

        return cached_prediction


@app.get("/state")
def get_state():
    with state_lock:
        return queue_state.to_dict()


@app.get("/state/owner-cache")
def get_owner_cache():
    with state_lock:
        return {
            "owner_eta_cache": owner_eta_cache
        }


@app.get("/state/completed-buffer")
def get_completed_buffer():
    with state_lock:
        return {
            "completed_jobs_buffer_size": (
                len(completed_jobs_buffer)
            ),
            "completed_jobs_buffer": (
                completed_jobs_buffer
            ),
        }


@app.get("/state/update-policy")
def get_update_policy():
    with state_lock:
        seconds_since_last_update = (
            time.time() - last_model_update_time
        )

        return {
            "auto_update_min_records": (
                AUTO_UPDATE_MIN_RECORDS
            ),
            "auto_update_interval_seconds": (
                AUTO_UPDATE_INTERVAL_SECONDS
            ),
            "completed_jobs_buffer_size": (
                len(completed_jobs_buffer)
            ),
            "seconds_since_last_model_update": round(
                seconds_since_last_update,
                4
            ),
        }