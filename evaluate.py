import json


def extract_triplet(preds, data):
    sentences = data['sentences']
    text = []
    for sent in sentences:
        text += sent
    triplets = []
    for pred in preds:
        idx = pred[0]
        rels = pred[1]
        for rel in rels:
            e1_start, e1_end = rel[0][0], rel[0][1] + 1
            e2_start, e2_end = rel[1][0], rel[1][1] + 1
            e1 = ' '.join(text[e1_start: e1_end])
            e2 = ' '.join(text[e2_start: e2_end])
            r = rel[2]
            triplets.append((e1.strip(), r, e2.strip()))

    return triplets


if __name__ == '__main__':

    with open("data/processed_train_short_0_1500.json", "r") as f:
        train_mup_first_half = json.load(f)

    with open("data/processed_train_short_1500_3000.json", "r") as f:
        train_mup_second_half = json.load(f)

    with open("data/rel_pred_0_1500.json", "r") as f:
        pred_first_half = json.load(f)

    with open('data/rel_pred_1500_3000.json', 'r') as f:
        pred_second_half = json.load(f)

    results = []
    for i in range(5):
        triplets_first_half = extract_triplet(pred_first_half[str(i)], train_mup_first_half[i])
        triplets_second_half = extract_triplet(pred_second_half[str(i)], train_mup_second_half[i])
        results.append({'id': i, 'result': triplets_first_half + triplets_second_half})

    with open("data/closedie_results.json", "w") as f:
        # for triplets in results:
        #     f.writelines([t[0] + ' ' + t[1] + ' ' + t[2] for t in triplets])
        #     f.write('\n')
        json.dump(results, f)
