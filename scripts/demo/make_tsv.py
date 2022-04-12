'''
In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity.  What causes precipitation to fall?
'''

import csv

in_path = 'input.tsv'

with open(in_path, mode='w', encoding='utf-8') as o:
    writer = csv.writer(o, delimiter='\t')

    questions = ['told', 'confirm', 'broke']
    context = 'He told her to confirm that John broke the window .'

    for question in questions:
        input_seq = [question,context]
        writer.writerow(input_seq)
