import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from omegaconf import OmegaConf
from collections import Counter
import cv2
import json
import pdb
import unidecode
from pathlib import Path
import shutil
from typing_extensions import List, Tuple, Dict, Any, Union, Literal



def is_image(fp):
    fp = str(fp)
    return fp.endswith('.jpg') or fp.endswith('.png') or fp.endswith('.jpeg') or fp.endswith('.JPG') or fp.endswith('.JPEG') or fp.endswith('.PNG')


def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


def row_bbs(bbs):
    polys = []
    poly2bb = {}
    for bb in bbs:
        poly = [bb[0], bb[1], bb[2], bb[1], bb[2], bb[3], bb[0], bb[3]]
        polys.append(poly)
        poly2bb[tuple(poly)] = bb
    poly_rows = row_polys(polys)
    bb_rows = []
    for row in poly_rows:
        bb_row = []
        for poly in row:
            bb_row.append(poly2bb[tuple(poly)])
        bb_rows.append(bb_row)
    return bb_rows


def sort_bbs(bbs):
    bb2idx_original = {tuple(bb): i for i, bb in enumerate(bbs)}
    bb_rows = row_bbs(bbs)
    sorted_bbs = [bb for row in bb_rows for bb in row]
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly2idx_original = {tuple(poly): i for i, poly in enumerate(polys)}
    poly_clusters = row_polys(polys)
    sorted_polys = []
    for row in poly_clusters:
        sorted_polys.extend(row)
    sorted_indices = [poly2idx_original[tuple(poly)] for poly in sorted_polys]
    return polys, sorted_indices


def sort_json(json_data):
    polys = []
    poly2label = {}
    poly2text = {}
    poly2idx_original = {}
    poly2row = {}
    for i, shape in enumerate(json_data['shapes']):
        if 'type' in shape and shape['type'] == 'rectangle':
            continue

        if len(shape['points']) != 4:
            raise ValueError('Json contains shape with more than 4 points')
        
        poly = shape['points']
        poly = [int(coord) for pt in poly for coord in pt]
        polys.append(poly)
        poly2label[tuple(poly)] = shape['label']
        poly2text[tuple(poly)] = shape['text']
        poly2idx_original[tuple(poly)] = i
    rows = row_polys(polys)
    for row_idx, row in enumerate(rows):
        for poly in row:
            poly2row[tuple(poly)] = row_idx
    return poly2label, poly2text, rows, poly2idx_original, poly2row



def poly2box(poly):
    poly = np.array(poly).flatten().tolist()
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return [xmin, ymin, xmax, ymax]



def filter_text_detect_boxes(polys, im_size):
    im_w, im_h = im_size
    remove_indexes = []
    for index, poly in enumerate(polys):
        is_remove = False
        bb = poly2box(poly)
        bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]

        if not bb_w/bb_h >= 0.8:
            is_remove = True
        
        if bb_h <= 5 or bb_w < 5:
            is_remove = True
        
        if is_remove:
            remove_indexes.append(index)
    
    return remove_indexes


def iou_poly(poly1, poly2):
    poly1 = np.array(poly1).flatten().tolist()
    poly2 = np.array(poly2).flatten().tolist()

    xmin1, xmax1 = min(poly1[::2]), max(poly1[::2])
    ymin1, ymax1 = min(poly1[1::2]), max(poly1[1::2])
    xmin2, xmax2 = min(poly2[::2]), max(poly2[::2])
    ymin2, ymax2 = min(poly2[1::2]), max(poly2[1::2])

    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return 0, 0, 0

    if len(poly1) == 4:  # if poly1 is a box
        x1, y1, x2, y2 = poly1
        poly1 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    if len(poly2) == 4:  # if poly2 is a box
        x1, y1, x2, y2 = poly2
        poly2 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if len(poly1) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly1
        poly1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    if len(poly2) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly2
        poly2 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    ratio1 = intersect / poly1.area
    ratio2 = intersect / poly2.area
    iou = intersect / union
    
    return ratio1, ratio2, iou



def write_to_xml(boxes, labels, size, xml_path):
    w, h = size
    root = ET.Element('annotations')
    filename = ET.SubElement(root, 'filename')
    filename.text = Path(xml_path).stem + '.jpg'
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin, ymin = ET.SubElement(bndbox, 'xmin'), ET.SubElement(bndbox, 'ymin')
        xmax, ymax = ET.SubElement(bndbox, 'xmax'), ET.SubElement(bndbox, 'ymax')
        xmin.text, ymin.text, xmax.text, ymax.text = map(str, box)
    ET.ElementTree(root).write(xml_path)



def is_valid_row(row, mean_row_h, dist_threshold, min_consecutive_words=3):
    """
        Row is valid if:
        + there're at least "min_consecutive_words" bbs that is near each other
    """
    def has_consecutive_words():
        cnt = 0
        for idx in range(1, len(row)):
            bb = row[idx]
            prev_bb = row[idx-1]
            dist = bb[0] - prev_bb[2]
            if dist < dist_threshold:
                cnt += 1
                if cnt >= min_consecutive_words:
                    return True
            else:
                cnt = 0
        return False
    
    def has_valid_height():
        row_h = max([bb[3]-bb[1] for bb in row])
        return row_h >= 0.7 * mean_row_h
    
    return has_consecutive_words() and has_valid_height()


def get_mean_word_dist(bb_rows, top_row=3):
    bb_rows = sorted(bb_rows, key=lambda row: len(row), reverse=True)
    bb_rows = bb_rows[:top_row]
    dists = []
    for row in bb_rows:
        row_dists = []
        for idx in range(1, len(row)):
            word_dist = row[idx][0]-row[idx-1][2]
            if word_dist > row[idx][2] - row[idx][0]:
                continue
            row_dists.append(word_dist)
        dists.extend(row_dists)
    return np.median(dists) if len(dists) > 0 else 4


def is_region_black(image: np.ndarray, bb):
    """
    Determines whether a bounding box region in an image has more black or white pixels.

    Parameters:
        image_path (str): Path to the black-and-white or grayscale image.
        bbox (tuple): A tuple (x, y, width, height) defining the bounding box.

    Returns:
        str: "black" if the region contains more black pixels, "white" if more white pixels.
    """
    roi = image[bb[1]:bb[3], bb[0]:bb[2]]

    # Threshold the ROI to binary if it's not already binary
    _, binary_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

    # Count black (pixel value 0) and white (pixel value 255) pixels
    black_pixels = np.sum(binary_roi == 0)
    white_pixels = np.sum(binary_roi == 255)
    total_pixels = black_pixels + white_pixels

    return black_pixels / total_pixels > 0.1


if __name__ == '__main__':
    im = Image.open('test.png').convert('L')
    im = np.array(im)
    print(is_region_black(im, (40, 30, 58, 60)))
