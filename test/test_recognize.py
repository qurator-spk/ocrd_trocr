import os
import shutil
import subprocess
import tempfile
import urllib.request
from lxml import etree
from glob import glob

import pytest
import logging
from ocrd.resolver import Resolver

from ocrd_trocr import TrOcrRecognize
from .base import assets

METS_KANT = assets.url_of('kant_aufklaerung_1784-page-region-line-word_glyph/data/mets.xml')
WORKSPACE_DIR = tempfile.mkdtemp(prefix='test-ocrd-trocr-')
TROCR_MODEL = os.getenv('TROCR_MODEL')


def page_namespace(tree):
    """Return the PAGE content namespace used in the given ElementTree.

    This relies on the assumption that, in any given PAGE content file, the root element has the local name "PcGts". We
    do not check if the files uses any valid PAGE namespace.
    """
    root_name = etree.QName(tree.getroot().tag)
    if root_name.localname == "PcGts":
        return root_name.namespace
    else:
        raise ValueError("Not a PAGE tree")

def assertFileContains(fn, text):
    """Assert that the given file contains a given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert text in f.read()

def assertFileDoesNotContain(fn, text):
    """Assert that the given file does not contain given string."""
    with open(fn, "r", encoding="utf-8") as f:
        assert not text in f.read()


@pytest.fixture
def workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)

    resolver = Resolver()
    # due to core#809 this does not always work:
    #workspace = resolver.workspace_from_url(METS_KANT, dst_dir=WORKSPACE_DIR)
    # workaround:
    shutil.rmtree(WORKSPACE_DIR)
    shutil.copytree(os.path.dirname(METS_KANT), WORKSPACE_DIR)
    workspace = resolver.workspace_from_url(os.path.join(WORKSPACE_DIR, 'mets.xml'))

    # The binarization options I have are:
    #
    # a. ocrd_kraken which tries to install cltsm, whose installation is borken on my machine (protobuf)
    # b. ocrd_olena which 1. I cannot fully install via pip and 2. whose dependency olena doesn't compile on my
    #    machine
    # c. just fumble with the original files
    #
    # So I'm going for option c.
    for imgf in workspace.mets.find_files(fileGrp="OCR-D-IMG"):
        imgf = workspace.download_file(imgf)
        path = os.path.join(workspace.directory, imgf.local_filename)
        subprocess.call(['mogrify', '-threshold', '50%', path])

    # Remove GT Words and TextEquivs, to not accidently check GT text instead of the OCR text
    # XXX Review data again
    for of in workspace.mets.find_files(fileGrp="OCR-D-GT-SEG-WORD-GLYPH"):
        workspace.download_file(of)
        path = os.path.join(workspace.directory, of.local_filename)
        tree = etree.parse(path)
        nsmap_gt = { "pc": page_namespace(tree) }
        for to_remove in ["//pc:Word", "//pc:TextEquiv"]:
            for e in tree.xpath(to_remove, namespaces=nsmap_gt):
                e.getparent().remove(e)
        tree.write(path, xml_declaration=True, encoding="utf-8")
        assertFileDoesNotContain(path, "TextEquiv")

    yield workspace

    shutil.rmtree(WORKSPACE_DIR)


def test_recognize(workspace):
    TrOcrRecognize(
        workspace,
        input_file_grp="OCR-D-GT-SEG-WORD-GLYPH",
        output_file_grp="OCR-D-OCR-TROCR",
        parameter={
            "model": TROCR_MODEL,
        }
    ).process()
    workspace.save_mets()

    #page1 = os.path.join(workspace.directory, "OCR-D-OCR-TROCR/OCR-D-OCR-TROCR_0001.xml")
    page1 = os.path.join(workspace.directory, "OCR-D-OCR-TROCR/OCR-D-OCR-TROCR_phys_0001.xml")
    assert os.path.exists(page1)
    assertFileContains(page1, "TextEquiv")
    # FIXME assertFileContains(page1, "verſchuldeten")


# vim:tw=120:
