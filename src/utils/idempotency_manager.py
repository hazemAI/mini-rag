import hashlib
import json
from datetime import datetime, timezone, timedelta
from models.db_schemas.minirag.schemas.celery_task_execution import CeleryTaskExecution
from sqlalchemy import select, delete


class IdempotencyManager:

    def __init__(self, db_client, db_engine):
        self.db_client = db_client
        self.db_engine = db_engine

    def create_args_hash(self, task_name: str, task_args: dict):

        combined_data = {
            **task_args,
            "task_name": task_name,
        }

        json_string = json.dumps(combined_data, sort_keys=True, default=str)

        return hashlib.sha256(json_string.encode()).hexdigest()

    async def create_task_record(self, task_name: str, task_args: dict, celery_task_id: str = None) -> CeleryTaskExecution:
        """
        Create a new task execution record
        """

        args_hash = self.create_args_hash(task_name, task_args)

        task_record = CeleryTaskExecution(
            task_name=task_name,
            task_args_hash=args_hash,
            task_args=task_args,
            celery_task_id=celery_task_id,
            status="PENDING",
            started_at=datetime.now(timezone.utc),
        )

        async with self.db_client() as session:
            session.add(task_record)
            await session.commit()
            await session.refresh(task_record)
            return task_record

    async def update_task_status(self, execution_id: int, status: str, result: dict = None):

        async with self.db_client() as session:
            task_record = await session.get(CeleryTaskExecution, execution_id)
            if task_record:
                task_record.status = status
                if result:
                    task_record.result = result
                if status in ["SUCCESS", "FAILURE"]:
                    task_record.completed_at = datetime.now(timezone.utc)
                await session.commit()

    async def get_existing_task(self, task_name: str, task_args: dict, celery_task_id: str) -> CeleryTaskExecution:

        args_hash = self.create_args_hash(task_name, task_args)

        async with self.db_client() as session:
            stat = select(CeleryTaskExecution).where(
                CeleryTaskExecution.celery_task_id == celery_task_id,
                CeleryTaskExecution.task_name == task_name,
                CeleryTaskExecution.task_args_hash == args_hash,
            )
            result = await session.execute(stat)
            return result.scalar_one_or_none()

    async def should_execute_task(self, task_name: str, task_args: dict, celery_task_id: str,
                                  task_time_limit: int = 600) -> tuple[bool, CeleryTaskExecution]:
        """
        Check if task should be executed or return existing result.
        Args:
            task_name: Name of the task
            task_args: Arguments of the task
            task_time_limit: Time limit in seconds after which a stuck task can be re-executed
        Returns:
            tuple[bool, CeleryTaskExecution]:
                bool: True if task should be executed, False otherwise
                CeleryTaskExecution: Existing task record if task should not be executed
        """

        existing_task = await self.get_existing_task(task_name, task_args, celery_task_id)

        if not existing_task:
            return True, None

        if existing_task.status == "SUCCESS":
            return False, existing_task

        if existing_task.status in ["PENDING", "STARTED", "RETRY"]:
            if existing_task.started_at:
                now_utc = datetime.now(timezone.utc)
                time_elapsed = (
                    now_utc - existing_task.started_at).total_seconds()
                time_gap = 60
                if time_elapsed > (task_time_limit + time_gap):
                    return True, existing_task
            return False, existing_task

        return True, existing_task

    async def cleanup_old_task(self, time_retention: int = 86400) -> int:
        """
        Delete old task records older than time_retention seconds
        Args:
            time_retention: Time retention in seconds
        Returns:
            int: Number of deleted records
        """

        cutoff_time = datetime.now(timezone.utc) - \
            timedelta(seconds=time_retention)

        async with self.db_client() as session:
            stmt = delete(CeleryTaskExecution).where(
                CeleryTaskExecution.created_at < cutoff_time)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
