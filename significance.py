import numpy as np
import csv
import ast

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diffs = np.array(xs) - np.array(ys)
    diff = np.abs(np.mean(xs) - np.mean(ys))

    signs = np.random.randint(0, 2, (n, nmc)) * 2 - 1
    rand_avgs = (diffs.reshape(n, 1) * signs).mean(0)
    k = (diff < np.abs(rand_avgs)).sum()

    return k / nmc

def load_predictions(file_name):
    model_dic = {}
    per_label = {}
    per_label_sum = {}

    for i in range(31):
        per_label[i] = []

    with open(file_name, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            model_dic[row[0]] = sum(ast.literal_eval(row[1]))

            per_label_sum[row[0]] = [ast.literal_eval(row[1])[0], sum(ast.literal_eval(row[1])[1:19]), sum(ast.literal_eval(row[1])[19:])]

            for i, l in zip(range(len(ast.literal_eval(row[1]))), ast.literal_eval(row[1])):
                per_label[i].append(l)

    return model_dic, per_label, per_label_sum

def test_stat(one, two, one_key=None, two_key=None):

    list_one = []
    list_two = []

    if one_key != None:
        for v in one.keys():
            list_one.append(one[v][one_key])
            list_two.append(two[v][two_key])
    else:
        for v in one.keys():
            list_one.append(one[v])
            list_two.append(two[v])

    out = exact_mc_perm_test(list_one, list_two, 10000)
    print(out)

    return [out]

def make_csv(per_label_lists):

    label_names = ['0',
                 '10',
                 '11',
                 '12',
                 '13',
                 '14',
                 '18',
                 '2',
                 '3',
                 '34',
                 '38',
                 '4',
                 '41',
                 '46',
                 '5',
                 '6',
                 '7',
                 '8',
                 '9',
                 'P1-1',
                 'P1-2',
                 'P1-3',
                 'P12-1',
                 'P4-2',
                 'P4-4',
                 'P6-1',
                 'P6-3',
                 'P7-1',
                 'P7-2',
                 'P7-3',
                 'P7-4']

    with open("results/label_results.csv", "w") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(label_names)
        for label_list in per_label_lists:
            out_list = []
            for k in label_list.keys():
                out_list.append(np.array(label_list[k]).mean())
            writer.writerow(out_list)

def run():

    b16_prec_facts, b16_prec_facts_label, per_label_prec_facts = load_predictions("results/precedent_facts_preds.csv")
    b16_prec_args, b16_prec_args_label, per_label_prec_args = load_predictions("results/precedent_arguments_preds.csv")
    facts, facts_label, per_label_facts = load_predictions("results/facts_preds.csv")
    args, args_label, per_label_args = load_predictions("results/arguments_preds.csv")

    make_csv([b16_prec_args_label, b16_prec_facts_label, facts_label, args_label])

    out = []

    label_names = ['0',
                 '10',
                 '11',
                 '12',
                 '13',
                 '14',
                 '18',
                 '2',
                 '3',
                 '34',
                 '38',
                 '4',
                 '41',
                 '46',
                 '5',
                 '6',
                 '7',
                 '8',
                 '9',
                 'P1-1',
                 'P1-2',
                 'P1-3',
                 'P12-1',
                 'P4-2',
                 'P4-4',
                 'P6-1',
                 'P6-3',
                 'P7-1',
                 'P7-2',
                 'P7-3',
                 'P7-4']

    for i, label in zip(list(range(len(label_names))), label_names):
        mi_fact = np.array(facts_label[i]).mean() - np.array(b16_prec_facts_label[i]).mean()
        mi_args = np.array(facts_label[i]).mean() - np.array(b16_prec_args_label[i]).mean()
        print(label, mi_fact, mi_args, np.array(facts_label[i]).mean())

    for i, label in zip(list(range(len(label_names))), label_names):
        print(label)
        try:
            # out += test_stat(facts_label, b16_prec_facts_label, i, i)
            # out += test_stat(facts_label, b16_prec_args_label, i, i)
            test_stat(b16_prec_args_label, b16_prec_facts_label, i, i)
        except:
            pass

    print("End Test")

    print("precedent FACT vs ARGS")
    out += test_stat(b16_prec_facts, b16_prec_args)

    print("precedent FACT vs normal FACTS")
    out += test_stat(b16_prec_facts, facts)

    print("precedent ARGS vs normal FACTS")
    out += test_stat(b16_prec_args, facts)

    # benjaminy hochbach correction
    corrected = []
    p = 0.05
    for o, i in zip(out, [3, 2, 1]):
        corrected.append(o*6/i)

    print(corrected)

if __name__ == '__main__':
    run()