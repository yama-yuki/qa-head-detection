import argparse
import csv
import json
import os
import sys
import torch
from transformers import BertForQuestionAnswering, AutoTokenizer, AutoConfig
from eval import evaluate

def main():
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='pred')
    parser.add_argument('--model_dir')
    parser.add_argument('--pretrained', default='bert-base-uncased')
    parser.add_argument('--input_path', hint='.json')
    parser.add_argument('--output_path', hint='.json')
    args = parser.parse_args()

    mode = args.mode
    model_directory = args.model_dir
    pretrained_model = args.pretrained

    ## load model & tokenizer
    config = AutoConfig.from_pretrained(os.path.join(model_directory, "config.json"))
    tokenizer_config = AutoConfig.from_pretrained(os.path.join(model_directory, "tokenizer_config.json"))

    model = BertForQuestionAnswering.from_pretrained(pretrained_model, config=config)
    model.load_state_dict(torch.load(os.path.join(model_directory, "pytorch_model.bin"), map_location=torch.device('cuda')))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, config=tokenizer_config)

    def predict(question, context):

        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        output = model(**inputs)
        answer_start = torch.argmax(output.start_logits)  
        answer_end = torch.argmax(output.end_logits) + 1 
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        return answer

    if mode == 'pred':
        ## predict & write out
        in_path = args.input_path
        out_path = args.output_path

        with open(in_path,mode='r',encoding='utf-8') as f:
            with open(out_path,mode='w',encoding='utf-8') as o:
                reader = csv.reader(f, delimiter='\t')
                writer = csv.writer(o, delimiter='\t')

                for row in reader:
                    context = row[0]
                    questions = row[1:]
                    answers = [predict(question, context) for question in questions]
                    writer.writerow(answers)
    
    elif mode == 'test':
        ## evaluate on test set
        test_path = args.test_path
        pred_path = 'pred.json'
        
        with open(test_path, mode='r', encoding='utf-8') as f:
            dataset_json = json.load(f)
            dataset = dataset_json['data']
        with open(pred_path, mode='r', encoding='utf-8') as g:
            predictions = json.load(g)

        print(json.dumps(evaluate(dataset, predictions)))

    else:
        sys.exit('SPECIFY MODE')



if __name__ == '__main__':
    main()