from datasets import load_dataset
from openie import StanfordOpenIE
from tqdm import tqdm
import json


properties = {
    'openie.affinity_probability_cap': 2 / 3,
}


def openie_tool_demo():
    with StanfordOpenIE(properties=properties) as client:
        text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            print('|-', triple)


def process_mup_example(text, client):
    triples = []
    for triple in client.annotate(text):
        subject_entity = triple['subject']
        object_entity = triple['object']
        relation = triple['relation']
        triples.append((subject_entity, relation, object_entity))
    return triples


def process_mup(data):
    processed_data = []
    cnt = 0
    with StanfordOpenIE(properties=properties) as client:
        for example in tqdm(data):
            cnt += 1
            if cnt > 500:
                break
            sci_triples = process_mup_example(example['text'][:3000], client)
            example['sci_claims'] = sci_triples
            processed_data.append(example)

    return processed_data


if __name__ == '__main__':

    with open("data/processed_train.json", "r") as f:
        train_mup = json.load(f)

    _results = process_mup(train_mup[:5])
    results = []
    for i in range(5):
        results.append({'id': i, 'result': _results[i]['sci_claims']})

    with open("data/openie_results.json", "w") as f:
        json.dump(results, f)


