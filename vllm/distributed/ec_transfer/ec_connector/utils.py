# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase


class ECOutputAggregator:
    """Aggregate EC connector completion signals across workers."""

    def __init__(self, expected_finished_count: int):
        self._send_remaining_count: dict[str, int] = {}
        self._recv_remaining_count: dict[str, int] = {}
        self._expected_finished_count = expected_finished_count

    @classmethod
    def from_connector(cls, connector: "ECConnectorBase", world_size: int):
        return cls(connector.get_finished_count() or world_size)

    def aggregate(
        self,
        outputs: list[ModelRunnerOutput | None],
        primary: ModelRunnerOutput,
    ) -> ModelRunnerOutput:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()

        def update_finished_set(
            mm_hashes: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            for mm_hash in mm_hashes or ():
                remaining_count = remaining_count_dict.get(
                    mm_hash, self._expected_finished_count
                )
                remaining_count_dict[mm_hash] = remaining_count - 1
                if remaining_count_dict[mm_hash] == 0:
                    finished_set.add(mm_hash)
                    del remaining_count_dict[mm_hash]

        for model_runner_output in outputs:
            if model_runner_output is None:
                continue
            ec_output = model_runner_output.ec_connector_output
            if ec_output is None:
                continue
            update_finished_set(
                ec_output.finished_sending,
                self._send_remaining_count,
                finished_sending,
            )
            update_finished_set(
                ec_output.finished_recving,
                self._recv_remaining_count,
                finished_recving,
            )

        # Always replace the primary EC output. Otherwise raw output-rank
        # completions can leak to the scheduler before every worker reports.
        primary.ec_connector_output = ECConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
        )
        return primary
