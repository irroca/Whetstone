import pytest

from datatools.cleaners import CLEANERS, clean, strip_starcoder_metadata

BODY = "import os\n\n\ndef main():\n    print(os.getcwd())\n"


@pytest.mark.parametrize(
    "header",
    [
        "<reponame>MTES-MCT/sparte\n",
        "<filename>PyDSTool/core/context_managers.py\n",
        "<gh_stars>1-10\n",
        "<reponame>steven-lang/rational_activations<filename>lib/utils.py<gh_stars>10-100\n",
        "<filename>app/views.py\r\n",
    ],
)
def test_starcoder_metadata_line_is_removed(header):
    assert strip_starcoder_metadata(header + BODY) == BODY


def test_a_marker_inside_the_code_is_content():
    text = "import optparse\nparser = optparse.OptionParser('usage %prog -f <filename>')\n"
    assert strip_starcoder_metadata(text) == text
    assert strip_starcoder_metadata("<filename>a.py\n" + text) == text


def test_only_a_line_made_of_metadata_segments_is_removed():
    """A first line that merely starts like a marker but carries code stays."""
    text = '<filename>"a.py" < limit\n' + BODY
    assert strip_starcoder_metadata(text) == text


def test_clean_applies_cleaners_in_order():
    assert clean("<gh_stars>0\n" + BODY, ["starcoder_metadata"]) == BODY
    assert clean(BODY, []) == BODY
    assert set(CLEANERS) == {"starcoder_metadata"}
