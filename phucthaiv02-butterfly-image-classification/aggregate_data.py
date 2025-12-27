# dataset: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
# import kagglehub
# path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
# print(path)

import shutil
import os
import uuid

dir = ['test', 'train']

for i in dir:
    print(os.listdir(i))
    for j in os.listdir(i):
        this_uuid = uuid.uuid4()
        print(this_uuid)
        shutil.move(i+'/'+j, 'dataset/'+str(this_uuid)+'.jpg')
    os.rmdir(i)