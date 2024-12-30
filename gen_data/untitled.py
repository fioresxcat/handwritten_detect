from utils import *


def visulaize_text_detect(ip, jp):
    im = cv2.imread(str(ip))
    with open(jp) as f:
        data = json.load(f)
    polys = [shape['points'] for shape in data['shapes']]
    remove_indexes = filter_text_detect_boxes(polys, im.shape[:2])
    data['shapes'] = [shape for i, shape in enumerate(data['shapes']) if i not in remove_indexes]
    for shape in data['shapes']:
        if 'shape_type' in shape and shape['shape_type'] == 'rectangle':
            continue
        poly = shape['points']
        poly = [int(coord) for pt in poly for coord in pt]
        cv2.polylines(im, [np.array(poly).reshape(-1, 2)], True, (0, 0, 255), 2)
    cv2.imwrite('test.jpg', im)

    
def get_doclaynet_data():
    type2files = {
        'financial_reports': [],
        'scientific_articles': [],
        'laws_and_regulations': [],
        'government_tenders': [],
        'manuals': [],
        'patents': [],
        'unknown': []
    }
    src_dir = '/data/tungtx2/nfs_data/layout_analysis/data/doclaynet'
    doc_types = []
    for ip in Path(src_dir).rglob('*'):
        if not is_image(ip):
            continue
        found = False
        for doc_type in type2files:
            if doc_type == 'unknown':
                continue
            if ip.name.startswith(doc_type):
                type2files[doc_type].append(ip)
                found = True
        if not found:
            type2files['unknown'].append(ip)

    for k, v in type2files.items():
        print(k, ':', len(v))
    pdb.set_trace()

    num_sample = 500
    out_dir = 'raw_data/doclaynet_filtered'
    os.makedirs(out_dir, exist_ok=True)
    for doc_type, list_files in type2files.items():
        if doc_type == 'unknown':
            continue
        for _ in range(100):
            np.random.shuffle(list_files)
        save_dir = os.path.join(out_dir, doc_type)
        os.makedirs(save_dir, exist_ok=True)
        for ip in list_files[:num_sample]:
            shutil.copy(ip, save_dir)
            print(f'done {ip}')



def get_D4LA_data():
    type2files = {
        'budget': [],
        'email': [],
        'form': [],
        'invoice': [],
        'letter': [],
        'memo': [],
        'news_article': [],
        'presentation': [],
        'resume': [],
        'scientific_publication': [],
        'scientific_report': [],
        'specification': [],
        'unknown': []
    }
    src_dir = '/data/tungtx2/nfs_data/layout_analysis/data/D4LA/D4LA/train_images'
    for ip in Path(src_dir).rglob('*'):
        if not is_image(ip):
            continue
        found = False
        for doc_type in type2files:
            if doc_type == 'unknown':
                continue
            if ip.name.startswith(doc_type):
                type2files[doc_type].append(ip)
                found = True
        if not found:
            type2files['unknown'].append(ip)

    for k, v in type2files.items():
        print(k, ':', len(v))
    pdb.set_trace()

    num_sample = 200
    out_dir = 'raw_data/D4LA_filtered'
    os.makedirs(out_dir, exist_ok=True)
    for doc_type, list_files in type2files.items():
        if doc_type == 'unknown':
            continue
        for _ in range(100):
            np.random.shuffle(list_files)
        save_dir = os.path.join(out_dir, doc_type)
        os.makedirs(save_dir, exist_ok=True)
        for ip in list_files[:num_sample]:
            shutil.copy(ip, save_dir)
            print(f'done {ip}')



def nothing():
    # Load the document background image
    background = Image.open("raw_data/D4LA_filtered/email/email_0011981840.png")  # Replace with your document image path

    # Load the transparent text image
    text_image = Image.open("raw_data/InkData_word/trans/20140603_0041_KQBDVN-tg_0_0_5-VIỆT-transparent.png")  # Replace with your transparent PNG path

    # Ensure both images are in RGBA mode
    background = background.convert("RGBA")
    text_image = text_image.convert("RGBA")

    # Resize or position the text image (optional)
    # text_image = text_image.resize((desired_width, desired_height))  # Resize if needed
    position = (300, 50)  # Coordinates where you want to paste the text image

    # Paste the text image onto the background using itself as a mask
    background.paste(text_image, position, text_image)
    background = background.convert('RGB')
    # Save or display the result
    background.save("test.jpg")
    


if __name__ == '__main__':
    pass
    nothing()
    # get_doclaynet_data()
    # get_D4LA_data()
    # visulaize_text_detect(
    #     'raw_data/D4LA_filtered/email/email_0011981840.png',
    #     'raw_data/D4LA_filtered/email/email_0011981840.json'
    # )