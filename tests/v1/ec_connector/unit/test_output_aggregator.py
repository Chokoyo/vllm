# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.v1.outputs import ECConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


def _make_output(
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        ec_connector_output=ECConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        ),
    )


def test_aggregate_workers_output():
    aggregator = ECOutputAggregator(expected_finished_count=2)

    primary = _make_output(finished_sending={"send1"}, finished_recving={"recv1"})
    worker_output = _make_output()

    aggregated = aggregator.aggregate([primary, worker_output], primary)

    assert aggregated is primary
    ec_output = aggregated.ec_connector_output
    assert ec_output is not None
    assert ec_output.finished_sending is None
    assert ec_output.finished_recving is None
    assert aggregator._send_remaining_count == {"send1": 1}
    assert aggregator._recv_remaining_count == {"recv1": 1}

    primary = _make_output()
    worker_output = _make_output(
        finished_sending={"send1"}, finished_recving={"recv1"}
    )

    aggregated = aggregator.aggregate([primary, worker_output], primary)

    assert aggregated is primary
    ec_output = aggregated.ec_connector_output
    assert ec_output is not None
    assert ec_output.finished_sending == {"send1"}
    assert ec_output.finished_recving == {"recv1"}
    assert not aggregator._send_remaining_count
    assert not aggregator._recv_remaining_count
