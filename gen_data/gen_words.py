from utils import *


class WordsImageGen:
    """
        This class is to generate handwritten (hw) word/segment images to paste into background images:
        1. Get random handwritten (hw) single-word images
        2. Gen random hw segment images
    """
    def __init__(
        self, word_dir
    ):
        self.word_dir = word_dir
        if os.path.exists('resources/word_lines_dict.json'):
            with open('resources/word_lines_dict.json') as f:
                d = json.load(f)
            self.word_lines = [[Path(ip) for ip in ipaths] for k, ipaths in d.items()]
        else:
            d = self.get_word_lines_dict(word_dir)
            with open('resources/word_lines_dict.json', 'w') as f:
                json.dump(d, f)
            self.word_lines = [[Path(ip) for ip in ipaths] for k, ipaths in d.items()]
        self.all_word_paths = [ip for line in self.word_lines for ip in line]


    def get_word_lines_dict(self, word_dir):
        d = {}
        for ip in Path(word_dir).rglob('*'):
            if not is_image(ip) or 'checkpoint' in str(ip):
                continue
            xml_name, word_id, word_text, _ = ip.stem.split('-')
            _, p, l, w = word_id.split('_')
            line_id = f'{xml_name}-{p}-{l}'
            if line_id not in d:
                d[line_id] = [str(ip)]
            else:
                d[line_id].append(str(ip))
        return d
    

    def get_single_word_image(self):
        ip = np.random.choice(self.all_word_paths)
        xml_name, word_id, word_text, _ = ip.stem.split('-')
        im = Image.open(ip)
        return im, word_text


    def get_multi_word_images(self, num_words):
        images, texts = [], []
        cand_lines = [line for line in self.word_lines if len(line) >= num_words]
        if len(cand_lines) > 0:
            line = cand_lines[np.random.randint(len(cand_lines))]
            start_index = np.random.randint(max(len(line) - num_words, 1))
            word_ipaths = line[start_index:start_index + num_words]
            for ip in word_ipaths:
                im = Image.open(ip)
                images.append(im)
                xml_name, word_id, word_text, _ = ip.stem.split('-')
                texts.append(word_text)
        return images, texts