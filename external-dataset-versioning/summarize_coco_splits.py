import glob
import sys

from pycocotools.coco import COCO

def summarize_categories(cats):
    print('  # categories: %s' % len(cats))
    # print(cats)
    supers = {}
    for cat in cats:
        supers.setdefault(cat['supercategory'], []).append(cat['name'])
    for supe, subs in supers.items():
        print('    %s: %s' % (supe, subs))

def summarize_annotations(anns):
    key_counts = {}
    for ann in anns:
        for k, v in ann.items():
            if k not in key_counts:
                key_counts[k] = 0
            key_counts[k] += 1
            # if k == 'area':
            #     print(k, v)
    print('  # annotations: %s' % len(anns))
    for ann_key, count in key_counts.items():
        print('    %s: %s' % (ann_key, count))
    

def summarize_coco_split(path):
    print('Ann file: %s' % path)
    coco_api = COCO(path)
    coco_api.info()
    has_categories = 'categories' in coco_api.dataset
    print('  # images: %s' % len(coco_api.getImgIds()))
    if has_categories:
        summarize_categories(coco_api.loadCats(coco_api.getCatIds()))
    summarize_annotations(coco_api.loadAnns(coco_api.getAnnIds()))

def main(argv):
    for path in glob.glob('./datasets/coco/annotations/*ins*val*'):
        summarize_coco_split(path)

if __name__ == '__main__':
    main(sys.argv)