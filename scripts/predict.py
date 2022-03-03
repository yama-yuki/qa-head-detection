from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoConfig
import torch

model_directory = '../models/bert-base-uncased_tok_2_3e-5_24'
pretrained_model = 'bert-base-uncased'

config = AutoConfig.from_pretrained(model_directory + "/config.json")
tokenizer_config = AutoConfig.from_pretrained(model_directory + "/tokenizer_config.json")

model = BertForQuestionAnswering.from_pretrained(pretrained_model, config=config)
model.load_state_dict(torch.load(model_directory + "/pytorch_model.bin", map_location=torch.device('cuda')))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, config=tokenizer_config)

# 入力テキスト
context = "I ate sushi before I took a walk to the station."
question="ate"

# 推論の実行
inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]
output = model(**inputs)
answer_start = torch.argmax(output.start_logits)  
answer_end = torch.argmax(output.end_logits) + 1 
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

# 結果出力
print("Q: "+question)
print("A: "+answer)