import json
import os
import random

def json_update(args, cfg):
    with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, 'r') as file:
        dataset_taxonomy = json.loads(file.read())

    dataset_path = args.render_output_path
    for taxonomy in dataset_taxonomy:
        taxonomy_id = taxonomy['taxonomy_id']
        taxonomy_path = os.path.join(dataset_path, taxonomy_id)
        samples = []
        sample_num = 0
        for sample in os.listdir(taxonomy_path):
            samples.append(sample)
            sample_num+=1
        #print('{} num is: {}'.format(taxonomy['taxonomy_name'], sample_num))
        taxonomy['train'] = samples
        del taxonomy['val']
        del taxonomy['test']

    with open(cfg.DATASETS.SHAPENET.FINETUNE_TAXONOMY_FILE_PATH, 'w') as f:
            json.dump(dataset_taxonomy, f, indent=4, ensure_ascii=False)