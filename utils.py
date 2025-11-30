import numpy as np
import os, re, json
from PIL import Image
from tqdm import tqdm
from itertools import product

def matching(pred,gt):
	preds = generate_combinations(pred)
	gts = generate_combinations(gt)
	for p in preds:
		if p in gts:
			return True
	return False

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

def easy_metrics(interested, vec_labels, q_vec_labels, pnt):
	
	r_k = [1, 5, 10]
	for r in r_k:
		percls = {}
		total = {}
		mrr = 0
		for i in range(interested.shape[1]):
			percls[q_vec_labels[i]] = 0
			total[q_vec_labels[i]] = 0
			indices = interested[:r, i]
			preds = [vec_labels[idx] for idx in indices]
			positive = q_vec_labels[i]
			try:
				index = preds.index(positive)
				idx = 1/(int(index) + 1)
				mrr += idx
			except:
				pass
		print('MRR@: ',r,': ', mrr/interested.shape[1])
	return

def generate_combinations(string):

	strings = [string, string.replace("'s",""), string.replace("'s","s"), string.replace("_","'s"), string.replace("_","'s ")]
	combinations = []
	for string in strings:
		string = re.sub(r"['_\s-]", "", string)
		# string = re.sub(r"['_]", "", string)
		string = string.lower()
		# combinations = []
		underscores = [i for i, char in enumerate(string) if char == '_']
		replacements = product([' ', '-', '_'], repeat=len(underscores))
		for replace_pattern in replacements:
			new_string = list(string)
			for i, replace in zip(underscores, replace_pattern):
				if replace != '_':
					new_string[i] = replace
			combinations.append(''.join(new_string))
	return combinations