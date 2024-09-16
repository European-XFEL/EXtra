
from extra.components.utils import _identify_instrument, identify_sase


def test_identify_instrument(mock_spb_aux_run, mock_sqs_remi_run):
    assert _identify_instrument(mock_sqs_remi_run) == 'SQS'

    # Too high threshold due to SA3 sources.
    assert _identify_instrument(mock_sqs_remi_run, 0.9) is None

    # Selecting SA3 also yields SA3.
    assert _identify_instrument(
        mock_sqs_remi_run.select('SA3*')) == 'SA3'

    # Also contains domain-less device IDs.
    assert _identify_instrument(mock_spb_aux_run) == 'SPB'


def test_identity_sase(mock_spb_aux_run, mock_sqs_remi_run):
    assert identify_sase(mock_spb_aux_run) == 1
    assert identify_sase(mock_sqs_remi_run) == 3
