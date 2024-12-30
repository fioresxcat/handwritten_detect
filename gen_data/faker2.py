from utils import *
from gen_words import WordsImageGen
from copy import deepcopy
from config import GenDataConfig
from image_faker import ImageFaker
from config import Type2Config


class ImageFakerType2(ImageFaker):
    def __init__(self, bg_dir, word_dir, save_dir):
        super().__init__(bg_dir, word_dir, save_dir)
    

    def _init_config(self):
        # set all config attribute to this class's attribute
        for k in dir(Type2Config):
            if not k.startswith('__'):  # Ignore dunder and methods
                v = getattr(Type2Config, k)
                setattr(self, k, v)

    def find_large_leading_space_rows(self, bb_rows, mean_row_h, mean_word_dist, im_w):
        cand_rows = []
        for row_idx, row in enumerate(bb_rows):
            if not is_valid_row(row, mean_row_h, dist_threshold=mean_word_dist):
                continue
            first_bb = row[0]
            if first_bb[0] >= 1/4 * im_w:
                cand_rows.append(row_idx)
        return cand_rows


    def paste_one_row_leading_space(self, bg, row, all_pr_hw_bbs):
        hw_bbs = []
        # calc num words to paste
        mean_word_width = np.mean([bb[2]-bb[0] for bb in row])
        leading_space = row[0][0]
        max_words = int(leading_space/mean_word_width)
        num_words = np.random.randint(self.MIN_WORD_TO_PASTE, max_words)
        # get rois to paste
        rois, texts = self.word_gen.get_multi_word_images(num_words)
        rois = rois[::-1] # reverse
        texts = texts[::-1]
        if len(rois) == 0:
            print('Skipping: No rois found')
            return bg, hw_bbs
        biggest_roi = self.find_highest_normal_roi(rois, texts)
        # get row h
        row_h = max([bb[3]-bb[1] for bb in row])
        scale_ratio = row_h * self.ROW_HEIGHT_SCALE_RATIO / biggest_roi.height
        # resize all rois
        rois = [roi.resize((int(roi.width * scale_ratio), int(roi.height * scale_ratio))) for roi in rois]
        smallest_roi = min(rois, key=lambda im: im.height)
        if smallest_roi.height <= self.MIN_WORD_HEIGHT:  # skip if boxes to paste are too small
            print('Skipping: Words too small')
            return bg, hw_bbs
        # get word dist
        ls_dists = [row[idx][0] - row[idx-1][2] for idx in range(1, len(row))]
        default_offset = np.mean(ls_dists) if len(ls_dists) > 0 else self.DEFAULT_WORD_DIST
        if default_offset / self.DEFAULT_WORD_DIST > 2:
            default_offset = int(self.DEFAULT_WORD_DIST*1.2)
        # get row_ymax
        row_ymax = max([bb[3] for bb in row])
        # paste
        first_pr_bb = row[0]  # last printing box
        paste_xmax = first_pr_bb[0] - default_offset
        gray_bg = np.array(bg.convert('L'))
        for roi_idx, (roi, text) in enumerate(zip(rois, texts)):
            paste_xmax += np.random.randint(-2, 2)
            paste_ymax = row_ymax if self.is_normal_height_text(text) else row_ymax + int(roi.height*self.HIGH_TEXT_OFFSET_RATIO)
            paste_ymin = paste_ymax - roi.height
            paste_xmin = paste_xmax - roi.width
            new_word_bb = [paste_xmin, paste_ymin, paste_xmax, paste_ymax]
            new_word_bb = list(map(int, new_word_bb))
            # check if box2paste overlaps with existing boxes or is out of bounds
            if new_word_bb[0] < 0 or is_region_black(gray_bg, new_word_bb) or self.is_overlap_with_existing_boxes(new_word_bb, all_pr_hw_bbs):
                break
            bg = self.paste(bg, roi, pos=(int(paste_xmin), int(paste_ymin)))
            paste_xmax = paste_xmin - default_offset
            hw_bbs.append(new_word_bb)
            all_pr_hw_bbs.append(new_word_bb)
        
        return bg, hw_bbs


    def fake(
        self, save_name, bg=None, bbs=None, visualize=False,
    ):
        """
            Paste hw boxes to leading space of rows
        """
        
        # get background and general text infos
        if bg is None:
            bg, bbs = self.get_background(doc_type=None)
        im_w, im_h = bg.size
        all_pr_hw_bbs = deepcopy(bbs)
        orig_all_pr_hw_bbs = deepcopy(bbs)
        all_hw_bbs = []
        if len(bbs) == 0:
            print('Skipping: No text found')
            return None, [], orig_all_pr_hw_bbs
            
        # get all text bbs
        text_xmin, text_xmax = min([bb[0] for bb in bbs]), max([bb[2] for bb in bbs])
        text_ymin, text_ymax = min([bb[1] for bb in bbs]), max([bb[3] for bb in bbs])
        bb_rows = row_bbs(bbs)
        # mean_row_h = np.mean([max([bb[3]-bb[1] for bb in row]) for row in bb_rows])
        mean_row_h = np.mean([max(bb[3] for bb in row) - min(bb[1] for bb in row) for row in bb_rows])
        med_row_h = np.median([max(bb[3] for bb in row) - min(bb[1] for bb in row) for row in bb_rows])
        mean_word_dist = get_mean_word_dist(bb_rows)

        # 2. find rows with large leading space
        cand_rows = self.find_large_leading_space_rows(bb_rows, mean_row_h, mean_word_dist, im_w)
        if len(cand_rows) == 0:
            print('Skipping: No rows with large leading space')
            return None, [], orig_all_pr_hw_bbs
            
        num_rows_to_paste = min(self.MAX_ROW_TO_PASTE, len(cand_rows)*1//3)
        paste_row_indexes = np.random.choice(cand_rows, size=num_rows_to_paste)
        for paste_row_idx in paste_row_indexes:
            paste_row = bb_rows[paste_row_idx]
            new_bg, hw_bbs = self.paste_one_row_leading_space(deepcopy(bg), paste_row, deepcopy(all_pr_hw_bbs))
            # print('NUM NEW BOXES:', len(hw_bbs))
            if len(hw_bbs) >= self.MIN_WORD_TO_PASTE:
                bg = new_bg
                all_hw_bbs.extend(hw_bbs)
                all_pr_hw_bbs.extend(hw_bbs)

        if visualize:
            draw = ImageDraw.Draw(bg)
            for bb in all_hw_bbs:
                draw.rectangle(bb, outline='red', width=2)
            
        if len(all_hw_bbs) >= self.MIN_WORD_TO_PASTE:
            if save_name is not None:
                self.save(bg, all_hw_bbs, save_name)
            return bg, all_hw_bbs, all_pr_hw_bbs
        else:
            return None, [], orig_all_pr_hw_bbs
