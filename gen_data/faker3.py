from utils import *
from gen_words import WordsImageGen
from copy import deepcopy
from config import GenDataConfig
from image_faker import ImageFaker
from config import Type3Config


class ImageFakerType3(ImageFaker):
    def __init__(self, bg_dir, word_dir, save_dir):
        super().__init__(bg_dir, word_dir, save_dir)
    

    def _init_config(self):
        # set all config attribute to this class's attribute
        for k in dir(Type3Config):
            if not k.startswith('__'):  # Ignore dunder and methods
                v = getattr(Type3Config, k)
                setattr(self, k, v)


    def find_empty_rows(self, bb_rows, mean_row_h, med_row_h, im_w):
        cand_rows = []
        for row_idx in range(1, len(bb_rows)):
            row = bb_rows[row_idx]
            prev_row = bb_rows[row_idx-1]
            row_ymin = min([bb[1] for bb in row])
            prev_row_ymax = max([bb[3] for bb in prev_row])
            k = int((row_ymin - prev_row_ymax) // mean_row_h)
            if k >= 2:
                xmin = np.random.randint(0, im_w//10)
                xmax = np.random.randint(im_w-im_w//10, im_w)
                ymin = prev_row_ymax
                for i in range(k-1):
                    ymin += mean_row_h * 1 // 3
                    ymax = ymin + int(med_row_h * self.ROW_HEIGHT_SCALE_RATIO)
                    cand_rows.append((xmin, ymin, xmax, ymax))
        return cand_rows


    def paste_one_empty_row(self, bg, row_bb, mean_word_dist, all_pr_hw_bbs):
        hw_bbs = []
        # get rois to paste
        rois, texts = self.word_gen.get_multi_word_images(num_words=10)
        if len(rois) == 0:
            print('Skipping: No rois found')
            return bg, hw_bbs
        biggest_roi = self.find_highest_normal_roi(rois, texts)
        # get row h
        row_h = row_bb[3] - row_bb[1]
        scale_ratio = row_h * self.ROW_HEIGHT_SCALE_RATIO / biggest_roi.height
        # resize all rois
        rois = [roi.resize((int(roi.width * scale_ratio), int(roi.height * scale_ratio))) for roi in rois]
        smallest_roi = min(rois, key=lambda im: im.height)
        if smallest_roi.height < self.MIN_WORD_HEIGHT:
            print('Skipping: Words too small')
            return bg, hw_bbs
        # get word dist
        default_offset = mean_word_dist
        row_ymax = row_bb[3]
        # paste
        paste_xmin = row_bb[0]
        gray_bg = np.array(bg.convert('L'))
        for roi_idx, (roi, text) in enumerate(zip(rois, texts)):
            paste_xmin += np.random.randint(-2, 2)
            paste_ymax = row_ymax if self.is_normal_height_text(text) else row_ymax + int(roi.height*self.HIGH_TEXT_OFFSET_RATIO)
            paste_ymin = paste_ymax - roi.height
            new_word_bb = [paste_xmin, paste_ymin, paste_xmin + roi.width, paste_ymax]
            new_word_bb = list(map(int, new_word_bb))
            # check if box2paste overlaps with existing boxes or is out of bounds
            if new_word_bb[2] > bg.width or is_region_black(gray_bg, new_word_bb) or self.is_overlap_with_existing_boxes(new_word_bb, all_pr_hw_bbs):
                break
            bg = self.paste(bg, roi, pos=(int(paste_xmin), int(paste_ymin)))
            paste_xmin += roi.width + default_offset + np.random.randint(0, 3)
            hw_bbs.append(new_word_bb)
            all_pr_hw_bbs.append(new_word_bb)
        
        return bg, hw_bbs
    

    def fake(
        self, save_name, bg=None, bbs=None, visualize=False,
    ):
        """
            Paste hw boxes to empty rows
        """
        
        # get background and general text infos
        if bg is None and bbs is None:
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
        mean_row_h = np.mean([max(bb[3] for bb in row) - min(bb[1] for bb in row) for row in bb_rows])
        med_row_h = np.median([max(bb[3] for bb in row) - min(bb[1] for bb in row) for row in bb_rows])
        # med_row_h = np.median([max([bb[3]-bb[1] for bb in row]) for row in bb_rows])
        mean_word_dist = get_mean_word_dist(bb_rows)
        # print('MEAN ROW HEIGHT: ', mean_row_h)
        # print('MEDIAN ROW HEIGHT: ', med_row_h)

        # 2. find rempty rows
        cand_rows = self.find_empty_rows(bb_rows, mean_row_h, mean_word_dist, im_w)
        if len(cand_rows) == 0:
            print('Skipping: No empty rows found')
            return None, [], orig_all_pr_hw_bbs
        num_rows_to_paste = min(self.MAX_ROW_TO_PASTE, len(cand_rows)*1//2)
        paste_row_indexes = np.random.choice(range(len(cand_rows)), size=num_rows_to_paste)
        for paste_row_idx in paste_row_indexes:
            paste_row = cand_rows[paste_row_idx]
            new_bg, hw_bbs = self.paste_one_empty_row(deepcopy(bg), paste_row, mean_word_dist, deepcopy(all_pr_hw_bbs))
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