from transformers import BertTokenizer, BertModel

# 下载并保存模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 保存到本地目录
tokenizer.save_pretrained('./bert-base-chinese')
model.save_pretrained('./bert-base-chinese')
