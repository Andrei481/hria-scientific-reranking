"""Read-only numerical checks for the November 2026 manuscript audit."""
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'resources'
text = (ROOT / 'articol_jurnal.tex').read_text(encoding='utf-8')
lines = text.splitlines()
report = {'manuscript_sha256': hashlib.sha256((ROOT / 'articol_jurnal.tex').read_bytes()).hexdigest(), 'checks': [], 'mismatches': []}

def nums(s):
    return [float(x) for x in re.findall(r'-?\d+\.\d+', s.replace('$-$', '-'))]

def check(name, actual, expected, digits):
    ok = len(actual) == len(expected) and all(abs(a-b) <= 0.50001*10**(-digits) for a,b in zip(actual,expected))
    report['checks'].append({'name': name, 'fields': len(expected), 'match': ok})
    if not ok:
        report['mismatches'].append({'name':name,'paper':actual,'resource':expected})

models = [('llama3',588),('gemma3_12b',593),('qwen2.5',598),('gemma3_27b',603)]
for model, start in models:
    for offset, retriever in enumerate(['bm25','grit','qwen3']):
        values = [json.loads((R/f'LitSearch/results/reranking/{model}/{state}/k_100/LitSearch.title_abstract.{retriever}.reranked.metrics.json').read_text())['overall_avg'] for state in ['base','FT']]
        expected = [v[k]*100 for k in ['R@20','R@5','NDCG@20'] for v in values]
        check(f'primary line {start+offset}',nums(lines[start+offset-1]),expected,1)

tie = list(csv.DictReader((R/'tie_sensitivity_artifacts/analysis_final_three_seeds/mean_sd.csv').open()))
for line,ret,pol in [(880,'bm25','deterministic'),(881,'bm25','epoch_random'),(883,'grit','deterministic'),(884,'grit','epoch_random'),(886,'qwen3','deterministic'),(887,'qwen3','epoch_random')]:
    vals=[]
    for metric in ['R@20','R@5','nDCG@20']:
        row=next(x for x in tie if x['policy']==pol and x['retriever']==ret and x['metric']==metric)
        vals.extend([float(row['mean']),float(row['sample_sd'])])
    check(f'tie line {line}',nums(lines[line-1]),vals,2)

for file, start, count, family, digits in [('statistical_uncertainty_corrected',622,24,'litsearch_base_ft',2),('statistical_uncertainty_csfcube_v2_20261020',1123,8,'csfcube_base_ft',3)]:
    rows=list(csv.DictReader((R/f'results/{file}/paired_uncertainty.csv').open()))
    paper=[s for s in lines[start-1:] if ' & ' in s and (' & R@20 &' in s or ' & nDCG' in s)][:count]
    for s in paper:
        parts=s.split('&'); model=parts[0].strip(); metric='R@20' if 'R@20' in s else ('NDCG@20' if family.startswith('litsearch') else 'NDCG%20')
        ret=parts[1].strip() if family.startswith('litsearch') else ''
        row=next(x for x in rows if x['family']==family and x['model']==model and x['metric']==metric and x['retriever']==ret)
        keys=['delta_percentage_points','ci95_low_percentage_points','ci95_high_percentage_points'] if ret else ['reference_mean','candidate_mean','delta','ci95_low','ci95_high']
        actual=nums('&'.join(parts[3:] if ret else parts[2:]))
        check(f'{family} {model} {ret} {metric}',actual[:len(keys)],[float(row[k]) for k in keys],digits)
        for p,key in zip(parts[-2:],['p_raw','p_holm']):
            expected=float(row[key]); printed=re.search(r'(?:\d+)?\.\d+',p)
            if printed:
                value=float(printed.group()); ok=expected<value if '<' in p else abs(expected-value)<0.000050001
                report['checks'].append({'name':f'{family} {model} {ret} {metric} {key}','fields':1,'match':ok})
                if not ok: report['mismatches'].append({'paper':p,'resource':expected})

def evaluate(path, expected_queries=597):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    vals=[]
    for row in rows:
        rel=set(row['corpusids']); ranked=row['retrieved']
        dcg=sum(1/math.log2(i+2) for i,d in enumerate(ranked[:20]) if d in rel)
        ideal=sum(1/math.log2(i+2) for i in range(min(len(rel),20)))
        vals.append([len(set(ranked[:k])&rel)/len(rel)*100 if rel else 0 for k in [20,5]]+[100*dcg/ideal if ideal else 0])
    assert len(rows)==expected_queries, (path,len(rows))
    return [statistics.mean(v[i] for v in vals) for i in range(3)]

curve={}
for budget in [10,25,50,75,100]:
    seedvals=[]
    for seed in [42,2026,3407]:
        folder=R/f'New folder/results_review/Llama-3.1-8B-Instruct_variants_3_target_50_pos_5_parent_budget_{budget}_seed_{seed}'
        paths=list(folder.glob('*/k_100/*.jsonl'))
        assert len(paths)==3,(folder,paths)
        vals=[evaluate(p) for p in paths]
        seedvals.append([statistics.mean(v[i] for v in vals) for i in range(3)])
    curve[budget]=[statistics.mean(v[i] for v in seedvals) for i in range(3)]
report['learning_curve_recomputed']=curve
check('learning curve prose', [1.03,0.45,1.02,0.70], [curve[100][0]-curve[10][0],curve[100][0]-curve[25][0],curve[25][1]-curve[100][1],curve[25][2]-curve[100][2]],2)
disjoint=R/'New folder/results_vm_250/new/disjoint'
for ret,line in [('bm25',1236),('grit',1239),('qwen3',1242)]:
    base=evaluate(disjoint/f'llama3_8b_base/k_100/LitSearch.title_abstract.{ret}.document_disjoint.reranked.jsonl',595)
    ft=evaluate(disjoint/f'Llama-3.1-8B-Instruct_variants_3_target_50_pos_5/20260807_012357/k_100/LitSearch.title_abstract.{ret}.document_disjoint.reranked.jsonl',595)
    for i in range(3):
        check(f'disjoint {ret} metric {i}',nums(lines[line+i-1]),[base[i],ft[i],ft[i]-base[i]],2)

compact=R/'New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline'
for ret,start in [('bm25',1016),('grit',1019),('qwen3',1022)]:
    for offset,state in enumerate(['zeroshot','dorismae']):
        data=json.loads((compact/f'litsearch/LitSearch.title_abstract.{ret}.gte_modernbert_{state}.reranked.metrics.json').read_text())['overall_avg']
        check(f'compact line {start+offset}',nums(lines[start+offset-1]),[100*data[k] for k in ['R@20','R@5','NDCG@20']],2)
compute=json.loads((compact/'compute/matched_compute_benchmark.json').read_text())
report['compute_measurements']=compute
report['compute_speed_ratio']=compute['generative_llm']['median_seconds']/compute['gte_modernbert']['median_seconds']
report['disjoint_candidate_matches']=[]
report['metric_files_scanned']=0
for p in R.rglob('*.metrics.json'):
    report['metric_files_scanned']+=1
    data=json.loads(p.read_text(encoding='utf-8-sig'))
    overall=data.get('overall_avg',{})
    if any(abs(overall.get('R@20',-1)*100-t)<.005 for t in [53.90,67.29,72.42,82.08,74.25,82.03]):
        report['disjoint_candidate_matches'].append({'path':str(p.relative_to(ROOT)),'overall':overall})
out=ROOT/'docs/manuscript_evidence_checks_2026-11-01.json'
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'checks':len(report['checks']),'mismatches':report['mismatches'],'metric_files_scanned':report['metric_files_scanned'],'disjoint_candidate_matches':report['disjoint_candidate_matches']},indent=2))
