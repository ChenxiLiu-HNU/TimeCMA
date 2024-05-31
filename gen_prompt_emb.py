import torch
import torch.nn as nn
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
from datetime import datetime, timedelta
import time
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'Exchange',
        model_name = "gpt2",
        model_path = "/data/cxliu/code/NeurIPS2023-One-Fits-All/Long-term_Forecasting/llama/llama/output",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1

        if self.model_name in ['gpt2']:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2Model.from_pretrained(model_name).to(self.device)

        elif self.model_name in ['llama2']:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self.model = LlamaModel.from_pretrained(model_path, output_hidden_states=True).to(self.device)
            self.model.layers = nn.ModuleList(self.model.layers[:layer])

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):

        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([f"{value:g}" for value in values])

        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():g}"

        start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
        end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        # print("in_prompt: ", in_prompt)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device) # [1, T, V]
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state # [1, T, E, V]
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
        # input_template=f"From [t1] to [t2], the values were value1, ..., valuen every day. The total trend value was Trends" # exc
        input_template=f"From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends" # fred
        tokenized_prompts = []
        max_token_count = 0
        for i in range(len(in_data)):
            for j in range(in_data.shape[2]):
                tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt.to(self.device),j))
        # print("Len of tokenized_prompts: ",len(tokenized_prompts)) # exc 321 | ili 7 | fred 107
        # print(max_token_count) # exc 583 0.3 -> 606 g | ili 216 ｜ fred 231

        in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

        for i, tokenized_prompt, j in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)

            else:
                prompt_embeddings_padded = prompt_embeddings
                       
            in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
            in_prompt_emb_1 = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
            in_prompt_emb_1 = in_prompt_emb_1.squeeze()


        #   Save and visualize
        #     embeddings = in_prompt_emb_1.t()
        #     norm_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        #     # cosine similarity
        #     last_similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

        # similarity_matrix_cpu = last_similarity_matrix.cpu().numpy()
        # similarity_df = pd.DataFrame(similarity_matrix_cpu)
        # similarity_df.to_csv('./Similarity_matrix/fred.csv', index=False)

        # plt.rcParams.update({'font.size': 20})

        # fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        # heatmap = sns.heatmap(similarity_df, annot=False, cmap='coolwarm')
        
        # # Set labels with a specific rotation and font size
        # # labels = ['WEIGHTED ILI', 'UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT']
        # # heatmap.set_xticklabels(labels, rotation=45, ha='right', fontsize=20)
        # # heatmap.set_yticklabels(labels, rotation=0, fontsize=20)

        # # Adding titles and axis labels with specific font size
        # plt.title('LLM Embedding Similarity Between Variables', fontsize=20)
        # plt.savefig('./Similarity_matrix/fred.pdf', format='pdf', dpi=300)
        # plt.close()

        return in_prompt_emb_1
