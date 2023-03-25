'''
@Time    : 2022/7/29 10:19
@Author  : leeguandon@gmail.com
'''
import torch
from transformers import GPT2LMHeadModel, GPT2Config

# fy = torch.load(r"E:\common_tools\CPM-main\CPM-main\model\epoch77\pytorch_model.bin", map_location=torch.device('cpu'))
# for i in fy.keys():
#     print(i + '   ' + str(list(fy[i].size())))

# model = GPT2LMHeadModel.from_pretrained("model/zuowen_epoch40")
model_config = GPT2Config.from_json_file("F:\gitlab\material\cpm\config\cpm-small.json")
model = GPT2LMHeadModel(config=model_config)
print(model)
