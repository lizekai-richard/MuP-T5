import json
import matplotlib.pyplot as plt
import numpy as np
from evaluate import extract_triplet
from stanford_openie import process_mup


with open("data/processed_train_short_0_1500.json", "r") as f:
    train_mup_first_half = json.load(f)

with open("data/processed_train_short_1500_3000.json", "r") as f:
    train_mup_second_half = json.load(f)

with open("data/rel_pred_0_1500.json", "r") as f:
    pred_first_half = json.load(f)

with open("data/rel_pred_1500_3000.json", "r") as f:
    pred_second_half = json.load(f)

train_mup_first_half = train_mup_first_half[:100] + train_mup_first_half[-100:-1]
train_mup_second_half = train_mup_second_half[:100] + train_mup_second_half[-100:-1]

# total # of triplets
# avg # of triplets per paper
closedie_total_cnt = 0
paper_cnt = 199
used_for_cnt = 0
feature_of_cnt = 0
hyponym_of_cnt = 0
part_of_cnt = 0
compare_cnt = 0
conjunction_cnt = 0
evaluate_for_cnt = 0

for i in range(199):
    triplets_first_half = extract_triplet(pred_first_half[str(i)], train_mup_first_half[i])
    triplets_second_half = extract_triplet(pred_second_half[str(i)], train_mup_second_half[i])
    triplets = triplets_first_half + triplets_second_half
    closedie_total_cnt += len(triplets)

    for triplet in triplets:
        rel = triplet[1]
        if rel == "USED-FOR":
            used_for_cnt += 1
        if rel == "FEATURE-OF":
            feature_of_cnt += 1
        if rel == "HYPONYM-OF":
            hyponym_of_cnt += 1
        if rel == "PART-OF":
            part_of_cnt += 1
        if rel == "COMPARE":
            compare_cnt += 1
        if rel == "CONJUNCTION":
            conjunction_cnt += 1
        if rel == "EVALUATE-FOR":
            evaluate_for_cnt += 1


closedie_avg_cnt = 1.0 * closedie_total_cnt / paper_cnt
avg_used_for = 1.0 * used_for_cnt / paper_cnt
avg_feature_of = 1.0 * feature_of_cnt / paper_cnt
avg_hyponym_of = 1.0 * hyponym_of_cnt / paper_cnt
avg_part_of = 1.0 * part_of_cnt / paper_cnt
avg_compare = 1.0 * compare_cnt / paper_cnt
avg_conjunction = 1.0 * conjunction_cnt / paper_cnt
avg_evaluate_for = 1.0 * evaluate_for_cnt / paper_cnt

plt.figure()
data = [used_for_cnt, feature_of_cnt, hyponym_of_cnt, part_of_cnt, compare_cnt, conjunction_cnt,
        evaluate_for_cnt]
labels = ['used_for', 'feature_of', 'hyponym_of', 'part_of', 'compare', 'conjunction', 'evaluate_for']
plt.bar(range(len(data)), data, tick_label=labels)
plt.xticks(rotation=20)
plt.savefig('closed_ie_total.png')

plt.figure()
data = [avg_used_for, avg_feature_of, avg_hyponym_of, avg_part_of, avg_compare, avg_conjunction,
        avg_evaluate_for]
labels = ['used_for', 'feature_of', 'hyponym_of', 'part_of', 'compare', 'conjunction', 'evaluate_for']
plt.bar(range(len(data)), data, tick_label=labels)
plt.xticks(rotation=20)
plt.savefig('closed_ie_avg.png')


openie_total_cnt = 0
with open("data/processed_train.json", "r") as f:
    train_mup = json.load(f)

train_mup = train_mup[:100] + train_mup[-100:-1]
processed_data = process_mup(train_mup)
for example in processed_data:
    openie_total_cnt += len(example['sci_claims'])

openie_avg_cnt = 1.0 * openie_total_cnt / paper_cnt

plt.figure()
plt.bar(range(2), [openie_total_cnt, closedie_total_cnt], width=0.3, tick_label=['open_ie', 'closed_ie'])
plt.savefig("compare_total.png")

plt.figure()
plt.bar(range(2), [openie_avg_cnt, closedie_avg_cnt], width=0.3, tick_label=['open_ie', 'closed_ie'])
plt.savefig("compare_avg.png")

if __name__ == '__main__':
    print("done")
