from pycocotools.coco import COCO

# d = COCO('./datasets/coco/annotations/instances_val2017.json')
d = COCO('./datasets/coco/annotations/instances_val2017.json')

anns = d.loadAnns(d.getAnnIds())

ann0 = anns[0]

# print('categories', d.loadCats(d.getCatIds()))

# print('firstcat', d.loadCats(ann0['category_id']))

print(ann0.keys())
# print(ann0['bbox'])
print(ann0['image_id'])

print(d.loadImgs(ann0['image_id']))

# TODO:
#   - make create_dataset.py, takes a query (from a json file) and generates a dataset
#   - make some kind of artifact API that can use the W&B API to load a dataset
#     (including grabbing the external files)