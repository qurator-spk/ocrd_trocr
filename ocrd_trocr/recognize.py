from __future__ import absolute_import

import os
import itertools
from glob import glob

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
        LabelType, LabelsType,
        MetadataItemType,
        TextEquivType,
        WordType, GlyphType, CoordsType,
        to_xml
)
from ocrd_utils import (
        getLogger, concat_padded,
        coordinates_for_segment, points_from_polygon, polygon_from_x0y0x1y1,
        make_file_id, assert_file_grp_cardinality,
        MIMETYPE_PAGE
)

from ocrd_trocr.config import OCRD_TOOL

TOOL = 'ocrd-trocr-recognize'


class TrOcrRecognize(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        # FIXME kwargs['version'] = '%s (TrOCR %s, PyTorch %s)' % (OCRD_TOOL['version'], XXX, XXX)
        super(TrOcrRecognize, self).__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        """
        Set up the model prior to processing.
        """
        #TODO resolved = self.resolve_resource(self.parameter['checkpoint_dir'])

        self.model = VisionEncoderDecoderModel.from_pretrained(self.parameter['model'])
        self.processor = TrOCRProcessor.from_pretrained(self.parameter['model'])

    def process(self):
        """
        Perform text recognition with TrOCR on the workspace.
        """
        log = getLogger('processor.TrOcrRecognize')

        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        for (n, input_file) in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            log.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))

            page = pcgts.get_Page()
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id, feature_selector=self.features)

            for region in page.get_AllRegions(classes=['Text']):
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords, feature_selector=self.features)

                textlines = region.get_TextLine()
                log.info("About to recognize %i lines of region '%s'", len(textlines), region.id)
                line_images_np = []
                line_coordss = []
                for line in textlines:
                    log.debug("Recognizing line '%s' in region '%s'", line.id, region.id)

                    line_image, line_coords = self.workspace.image_from_segment(line, region_image, region_coords, feature_selector=self.features)
                    if ('binarized' not in line_coords['features'] and
                            'grayscale_normalized' not in line_coords['features']):
                        # TODO review
                        # We cannot use a feature selector for this since we don't
                        # know whether the model expects (has been trained on)
                        # binarized or grayscale images; but raw images are likely
                        # always inadequate:
                        log.warning("Using raw image for line '%s' in region '%s'", line.id, region.id)

                    if (not all(line_image.size) or
                        line_image.height <= 8 or line_image.width <= 8 or
                        'binarized' in line_coords['features'] and line_image.convert('1').getextrema()[0] == 255):
                        # empty size or too tiny or no foreground at all: skip
                        log.warning("Skipping empty line '%s' in region '%s'", line.id, region.id)
                        line_image_np = np.array([[0]], dtype=np.uint8)
                    else:
                        line_image_np = np.array(line_image, dtype=np.uint8)
                    line_images_np.append(line_image_np)
                    line_coordss.append(line_coords)
                #raw_results_all = self.predictor.predict_raw(line_images_np, progress_bar=False)

                for line, line_coords, raw_results in zip(textlines, line_coordss, raw_results_all):

                    for i, p in enumerate(raw_results):
                        p.prediction.id = "fold_{}".format(i)

                    prediction = self.voter.vote_prediction_result(raw_results)
                    prediction.id = "voted"


                    # Delete existing results
                    if line.get_TextEquiv():
                        log.warning("Line '%s' already contained text results", line.id)
                    line.set_TextEquiv([])
                    if line.get_Word():
                        log.warning("Line '%s' already contained word segmentation", line.id)
                    line.set_Word([])

                    # Save line results
                    line_conf = prediction.avg_char_probability
                    line.set_TextEquiv([TextEquivType(Unicode=line_text, conf=line_conf)])


            _page_update_higher_textequiv_levels('line', pcgts)


            # Add metadata about this operation and its runtime parameters:
            self.add_metadata(pcgts)
            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))


# TODO: This is a copy of ocrd_tesserocr's function, and should probably be moved to a ocrd lib
def _page_update_higher_textequiv_levels(level, pcgts):
    """Update the TextEquivs of all PAGE-XML hierarchy levels above `level` for consistency.

    Starting with the hierarchy level chosen for processing,
    join all first TextEquiv (by the rules governing the respective level)
    into TextEquiv of the next higher level, replacing them.
    """
    regions = pcgts.get_Page().get_TextRegion()
    if level != 'region':
        for region in regions:
            lines = region.get_TextLine()
            if level != 'line':
                for line in lines:
                    words = line.get_Word()
                    if level != 'word':
                        for word in words:
                            glyphs = word.get_Glyph()
                            word_unicode = u''.join(glyph.get_TextEquiv()[0].Unicode
                                                    if glyph.get_TextEquiv()
                                                    else u'' for glyph in glyphs)
                            word.set_TextEquiv(
                                [TextEquivType(Unicode=word_unicode)])  # remove old
                    line_unicode = u' '.join(word.get_TextEquiv()[0].Unicode
                                             if word.get_TextEquiv()
                                             else u'' for word in words)
                    line.set_TextEquiv(
                        [TextEquivType(Unicode=line_unicode)])  # remove old
            region_unicode = u'\n'.join(line.get_TextEquiv()[0].Unicode
                                        if line.get_TextEquiv()
                                        else u'' for line in lines)
            region.set_TextEquiv(
                [TextEquivType(Unicode=region_unicode)])  # remove old

# vim:tw=120:
