# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
    ):
        if not has_ec_transfer():
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)

    def maybe_get_ec_connector_output(
        self,
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:
        return (
            self._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()
            else nullcontext()
        )

    @staticmethod
    def maybe_wait_for_ec_load_caches(
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        if not has_ec_transfer():
            return
        connector = get_ec_transfer()
        if connector.is_consumer:
            connector.wait_for_load_caches(encoder_cache=encoder_cache, **kwargs)

    # This context manager must be used within an active forward context.
    # It encapsulates the entire EC connector lifecycle within execute_model
    @contextmanager
    def _get_ec_connector_output(
        self,
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()

        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        # Load caches for consumer or both roles
        if ec_connector.is_consumer:
            ec_connector.start_load_caches(encoder_cache, **kwargs)

        try:
            yield output
        finally:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
            self._ec_connector_output_this_step = output
            self._ec_polled_this_step = True

            ec_connector.clear_connector_metadata()

    @staticmethod
    def ec_connector_no_forward(
        scheduler_output: "SchedulerOutput",
    ) -> ECConnectorOutput | None:
        if not has_ec_transfer():
            return None

        output = ECConnectorOutput()
        ec_connector = get_ec_transfer()
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)
        try:
            output.finished_sending, output.finished_recving = (
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
        finally:
            ec_connector.clear_connector_metadata()

        if not output.finished_sending and not output.finished_recving:
            return None
        return output
