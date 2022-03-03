import argparse
import csv
import torch
from transformers import BertForQuestionAnswering, AutoTokenizer, AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir')
parser.add_argument('--pretrained', default='bert-base-uncased')
parser.add_argument('--input_path')
parser.add_argument('--output_path')
args = parser.parse_args() 
model_directory = args.model_dir
pretrained_model = args.pretrained
in_path = args.input_path
out_path = args.output_path

config = AutoConfig.from_pretrained(model_directory + "/config.json")
tokenizer_config = AutoConfig.from_pretrained(model_directory + "/tokenizer_config.json")

model = BertForQuestionAnswering.from_pretrained(pretrained_model, config=config)
model.load_state_dict(torch.load(model_directory + "/pytorch_model.bin", map_location=torch.device('cuda')))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, config=tokenizer_config)

def predict(question, context):
    ## predict
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  
    answer_end = torch.argmax(output.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def main():
    with open(in_path,mode='r',encoding='utf-8') as f:
        with open(out_path,mode='w',encoding='utf-8') as o:
            reader = csv.reader(f, delimiter='\t')
            writer = csv.writer(o, delimiter='\t')

            for row in reader:
                context = row[0]
                questions = row[1:]
                answers = [predict(question, context) for question in questions]
                writer.writerow(answers)

if __name__ == '__main__':
    main()