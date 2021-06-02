import pickle
import os

results = pickle.load(open('/home/oscar/workspace/master/modules/m1/project/Team3/data/qst1_w5/results/result.pkl','rb'))

count_10 = 0
count_1 = 0
count_bad = 0

for result_img in results:
    for result_painting in result_img:
        if len(result_painting) == 10:
            count_10 += 1

        elif len(result_painting) == 1:
            if result_painting[0] == -1:
                count_1 += 1
            else:
                count_bad += 1
        else:
            count_bad += 1

print(f'Total: {len(results)}, Count_10: {count_10}, Count_1: {count_1}, Count_bad: {count_bad}')
