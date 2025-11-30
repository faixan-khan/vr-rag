import torch
import os, json, random, pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


from qwen_vl_utils import process_vision_info
from utils import easy_metrics
from prompts import get_prompt

seed_value = 0
random.seed(seed_value)
torch.manual_seed(seed_value)

def read_file(path):
	if path.endswith('.pkl'):
		with open(path,'rb') as f:
			return pickle.load(f)
	elif path.endswith('.npy'):
		return np.load(path)
	else:
		with open(path, 'r') as json_file:
			return json.load(json_file)

def set_seed(seed):
	"""Set the seed for reproducibility."""
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser('VR-RAG Inference')
parser.add_argument('--root_dir', default='./feats/', type=str, help='Root directory of the datasets')
parser.add_argument('--dataset', default='cub', type=str, choices=['birdsnap', 'cub', 'inat', 'indian', 'nabirds'], help='Dataset name')
parser.add_argument('--summary_file', type=str, default='./data/first_summary.json',help='species summary file')
parser.add_argument('--ptype', type=str, default='gemini',help='prompt type')
parser.add_argument('--output', type=str, default='output',help='output folder')
parser.add_argument('--k', default=5, type=int, help='number of retrieved species')
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
parser.add_argument('--rerank_type', type=int, default=1, choices=[0, 1], help='rerank type')
parser.add_argument('--rerank_topk', type=int, default=100, help='number of re-ranked candidates')
parser.add_argument('--visual_fusion', type=int, default=1, help='visual fusion')
parser.add_argument('--img_lamda', default=0.7, type=float, help='image lambda')
args = parser.parse_args()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
	"Qwen/Qwen2.5-VL-7B-Instruct",
	torch_dtype=torch.bfloat16,
	attn_implementation="flash_attention_2",
	device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",  revision="refs/pr/24")


messages = [
	{
		"role": "user",
		"content": [
			{
				"type": "image",
				"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
				"min_pixels": 50176, "max_pixels": 50176,
			},
			{
				"type": "text", "text": "Describe the image in detail."
			},
		],
	}
]

def get_visual_rerank(scores, vec_labels, args):
	args.model = 'dino'
	interested = torch.topk(scores, args.rerank_topk, dim=0).indices.cpu().numpy()
	dino_q = read_file('./{}/{}/{}/q.npy'.format(args.root_dir, args.model, args.dataset))
	dino_v = read_file('./{}/{}/visual_anchors.pkl'.format(args.root_dir, args.model))
	query_features = torch.tensor(dino_q, dtype=torch.float32)  
	reranked_scores = torch.zeros_like(scores[:args.rerank_topk, :])  
	ck = 0
	for img_idx in tqdm(range(interested.shape[1])):
		top30_indices = interested[:, img_idx]  
		top30_gt_keys = [vec_labels[idx] for idx in top30_indices]  
		top30_features_list = []
		for gt in top30_gt_keys:
			feats = torch.tensor(dino_v[gt])
			top30_features_list.append(feats)
		top30_features = torch.stack(top30_features_list)
		query_feature = query_features[img_idx].unsqueeze(0)  
		similarity_scores = (top30_features @ query_feature.T).squeeze().cuda() 
		reranked_scores[:, img_idx] = (scores[top30_indices, img_idx])*(args.img_lamda) + ((1-args.img_lamda)*similarity_scores)
	sorted_indices = torch.argsort(reranked_scores, dim=0, descending=True)  
	reranked_indices = interested[sorted_indices.cpu(), torch.arange(interested.shape[1]).unsqueeze(0)] 
	return reranked_indices


def rag_inf(args=None):

	args.split_file = f'./data/{args.dataset}_test.pkl'
	args.data_folder = f'../vr_rag/data/{args.dataset}/test'

	data_path = os.path.join(args.summary_file)	
	summary = read_file(data_path)
	models = ['clip','open','sig']
	save_name = os.path.join(args.dataset  + '_inference.json')
	print('Output: ',save_name)

	scores = {}
	for ret in models:
		print(ret)
		args.model = ret
		vecs = read_file(os.path.join(args.root_dir, args.model, 'spe_db.npy'))
		if args.visual_fusion == 1:
			v_vecs = read_file(os.path.join(args.root_dir, args.model, 'spe_db_visual.npy'))
			vecs = (vecs + v_vecs)/2
		vec_labels = read_file(os.path.join(args.root_dir, args.model, 'spe_db_labels.pkl'))
		q_vecs = read_file(os.path.join(args.root_dir, args.model, args.dataset, 'q.npy'))
		q_vec_labels = read_file(os.path.join(args.root_dir, args.model, args.dataset, 'q_labels.pkl'))
		if len(models) == 1:
			vecs = torch.from_numpy(vecs).float().cuda()
			q_vecs = torch.from_numpy(q_vecs).float().cuda()
			scores = torch.matmul(vecs, q_vecs.T).float().cuda()
		else:
			vecs = torch.from_numpy(vecs).float().cuda()
			q_vecs = torch.from_numpy(q_vecs).float().cuda()
			sc = torch.matmul(vecs, q_vecs.T)
			scores[ret] = sc
	if len(models) > 1:
		scores = torch.stack(list(scores.values())).mean(dim=0)


	interested = torch.topk(scores, 100, dim=0).indices.cpu().numpy()

	easy_metrics(interested, vec_labels, q_vec_labels, args.k)
	
	if args.rerank_type == 1:
		interested = get_visual_rerank(scores, vec_labels, args)	
		easy_metrics(interested, vec_labels, q_vec_labels, args.k)
	
	fams = {}
	prompt = get_prompt(args.ptype)
	print('prompt: ', prompt)
	ck = 0
	images = read_file(args.split_file)
	for i in tqdm(range(interested.shape[1])):
		fams[i] = {}
		indices = interested[:1000, i]
		preds = [vec_labels[idx] for idx in indices]
		preds = list(dict.fromkeys(preds))
		preds = preds[:args.k]
		fams[i] = preds
	images = read_file(args.split_file)
	resp = {}
	ck = 0

	if os.path.exists(save_name):
		resp = read_file(save_name)
		print(len(resp))
		ck = len(resp)

	failed = 0
	for idx, i in tqdm(enumerate(images)):
		img = os.path.join(args.data_folder, i)
		if not os.path.exists(img):
			failed += 1
			continue
		if img in resp:
			continue
		retreived = fams[ck]
		context = []
		for option_idx,ret in enumerate(retreived):
			text = str(ret) + ': ' + summary[ret] + '\n'
			context.append(text)
		context = ''.join(context)
			
		messages[0]['content'][0]['image'] = img
		messages[0]['content'][1]['text'] = prompt.format(context)
		text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		image_inputs, video_inputs = process_vision_info(messages)
		inputs = processor(
			text=[text],
			images=image_inputs,
			videos=video_inputs,
			padding=True,
			return_tensors="pt",
		)
		inputs = inputs.to("cuda")
		generated_ids = model.generate(**inputs, max_new_tokens=128)
		generated_ids_trimmed = [
			out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
		]
		output_text = processor.batch_decode(
			generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)[0]
		resp[img] = output_text
		ck+=1
		if ck%100==0:
			with open(save_name, 'w') as f:
				json.dump(resp, f)
	with open(save_name, 'w') as f:
		json.dump(resp, f)
	print("FAILED: ", failed)
	
if __name__ == '__main__':
	set_seed(args.seed)
	print(args)
	rag_inf(args)