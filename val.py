import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from config import *
from src.data_preprocessing import DataPreprocessor
from src.base_lstm import BaseLSTM
from src.modified_lstm import ModifiedLSTM
from src.predict import NextWordPredictor

os.makedirs('data/test',exist_ok=True)
os.makedirs('models',exist_ok=True)

print("="*70)
print("FULL EVALUATION: BASE LSTM vs MODIFIED LSTM (Working Memory Connections)")
print("="*70)

print("\n[1/4] Creating intensive test dataset with 1000+ sentences...")

alice_templates=[
    "the queen of hearts {}",
    "alice {} the mad hatter",
    "white rabbit {} his pocket watch",
    "cheshire cat {} from the tree",
    "the king {} at the jury",
    "the dormouse {} during the tea party",
    "mock turtle {} about his education",
    "the gryphon {} alice to the shore",
    "the duchess {} her baby to alice",
    "the cook {} pepper at everyone",
    "alice {} the mushroom and grew",
    "the caterpillar {} on a mushroom",
    "the march hare {} butter in the watch",
    "the hatter {} a very strange riddle",
    "the executioner {} how to cut a head",
    "the pigeon {} alice was a serpent",
    "the frog {} at the door",
    "the fish {} an invitation to the duchess",
    "the knave {} the tarts from the queen",
    "the playing cards {} painting the roses"
]

verbs1=['ran','walked','jumped','looked','shouted','whispered','screamed','laughed','cried','slept','ate','drank','threw','caught','pushed','pulled','opened','closed','broke','found','lost','gave','took','made','did','said','told','asked','answered']
verbs2=['quickly','slowly','loudly','quietly','happily','sadly','angrily','calmly','suddenly','gradually','carefully','carelessly','easily','hardly','barely','completely','partially','fully','almost','nearly']
nouns=['table','chair','door','window','garden','forest','castle','house','room','hall','court','kitchen','garden','tree','flower','bird','cat','dog','mouse','rabbit']
adjectives=['red','blue','green','yellow','big','small','tall','short','fat','thin','beautiful','ugly','smart','dumb','fast','slow','hot','cold','wet','dry']
adverbs=['very','quite','rather','somewhat','extremely','absolutely','totally','completely','slightly','barely']

test_sentences=[]

for i in range(1000):
    template=random.choice(alice_templates)
    verb1=random.choice(verbs1)
    verb2=random.choice(verbs2)
    noun=random.choice(nouns)
    adj=random.choice(adjectives)
    adv=random.choice(adverbs)
    
    fillers=[
        verb1,
        f"{verb1} {verb2}",
        f"{verb1} to the {noun}",
        f"{verb1} {adv}",
        f"was {adj} and {verb1}",
        f"{verb1} like a {noun}",
        f"could not {verb1}",
        f"wanted to {verb1}",
        f"tried to {verb1} {verb2}",
        f"looked {adj} and {verb1}"
    ]
    
    sentence=template.format(random.choice(fillers))
    test_sentences.append(sentence.lower())

original_sentences=[
    "the queen of hearts will be here soon",
    "alice could not help laughing at the funny cat",
    "the mad hatter asked a very difficult riddle",
    "white rabbit pulled out a large pocket watch",
    "cheshire cat vanished slowly starting with its tail",
    "the king looked at the jury with angry eyes",
    "alice ate a small cake and became very tiny",
    "the dormouse slept through the entire tea party",
    "mock turtle sighed deeply and began to cry",
    "the executioner did not know how to cut a head",
    "alice ran faster than she ever ran before",
    "the duchess handed the screaming baby to alice",
    "cook threw everything she could find at the door",
    "the gryphon took alice to meet the mock turtle",
    "alice tried to play croquet but the ball was a hedgehog",
    "the queen screamed louder than anyone in the room",
    "alice thought the whole trial was very stupid",
    "the caterpillar asked who are you very slowly",
    "alice did not know the answer to any question",
    "the cards flew up and covered the entire sky"
]

for s in original_sentences:
    test_sentences.append(s.lower())

random.shuffle(test_sentences)

with open('data/test/alice_test.txt','w')as f:
    for s in test_sentences:
        f.write(s+'\n')

print(f"   Created {len(test_sentences)} test sentences ({len(original_sentences)} original + {1000} generated)")

print("\n[2/4] Loading models and tokenizer...")

preprocessor=DataPreprocessor()
preprocessor.load_tokenizer('data/processed/tokenizer.pkl')
vocab_size=len(preprocessor.vocab)

base_model=BaseLSTM(vocab_size,EMBEDDING_DIM,HIDDEN_DIM,NUM_LAYERS,DROPOUT)
base_model.load_state_dict(torch.load('models/base_lstm_model.pth',map_location='cpu'))
base_model.eval()

modified_model=ModifiedLSTM(vocab_size,EMBEDDING_DIM,HIDDEN_DIM,NUM_LAYERS,DROPOUT)
modified_model.load_state_dict(torch.load('models/modified_lstm_model.pth',map_location='cpu'))
modified_model.eval()

base_pred=NextWordPredictor(base_model,preprocessor.word_to_idx,preprocessor.idx_to_word,'cpu')
mod_pred=NextWordPredictor(modified_model,preprocessor.word_to_idx,preprocessor.idx_to_word,'cpu')

print(f"   Vocabulary size: {vocab_size}")
print(f"   Test sentences: {len(test_sentences)}")

print("\n[3/4] Running intensive evaluation...")

results={
    'base_wins':0,
    'mod_wins':0,
    'ties':0,
    'base_scores':[],
    'mod_scores':[],
    'base_times':[],
    'mod_times':[],
    'length_performance':defaultdict(lambda:{'base':0,'mod':0,'total':0}),
    'word_predictions':defaultdict(lambda:{'base':0,'mod':0})
}

for idx,sentence in enumerate(test_sentences):
    words=sentence.split()
    if len(words)<4:
        continue
    
    split=len(words)//2
    input_text=' '.join(words[:split])
    expected_words=words[split:split+4]
    
    start=time.time()
    base_out=base_pred.generate_text(input_text,8)
    base_time=time.time()-start
    
    start=time.time()
    mod_out=mod_pred.generate_text(input_text,8)
    mod_time=time.time()-start
    
    results['base_times'].append(base_time)
    results['mod_times'].append(mod_time)
    
    base_match=0
    mod_match=0
    
    for w in expected_words:
        if w in base_out.lower():
            base_match+=1
        if w in mod_out.lower():
            mod_match+=1
    
    base_score=base_match/len(expected_words) if expected_words else 0
    mod_score=mod_match/len(expected_words) if expected_words else 0
    
    results['base_scores'].append(base_score)
    results['mod_scores'].append(mod_score)
    
    sent_len=len(words)
    results['length_performance'][sent_len]['total']+=1
    if base_score>mod_score:
        results['base_wins']+=1
        results['length_performance'][sent_len]['base']+=1
    elif mod_score>base_score:
        results['mod_wins']+=1
        results['length_performance'][sent_len]['mod']+=1
    else:
        results['ties']+=1
    
    for w in expected_words[:2]:
        if w in base_out:
            results['word_predictions'][w]['base']+=1
        if w in mod_out:
            results['word_predictions'][w]['mod']+=1
    
    if idx%200==0 and idx>0:
        print(f"   Processed {idx}/{len(test_sentences)} sentences")

print("\n[4/4] Generating results and graphs...")

base_avg=np.mean(results['base_scores'])
mod_avg=np.mean(results['mod_scores'])
base_std=np.std(results['base_scores'])
mod_std=np.std(results['mod_scores'])
base_median=np.median(results['base_scores'])
mod_median=np.median(results['mod_scores'])

print("\n"+"="*70)
print("EVALUATION RESULTS")
print("="*70)

print(f"\n📊 ACCURACY SCORES (0-1 scale, higher is better):")
print(f"   Base LSTM:     {base_avg:.4f} ± {base_std:.4f} (median: {base_median:.4f})")
print(f"   Modified LSTM: {mod_avg:.4f} ± {mod_std:.4f} (median: {mod_median:.4f})")
improvement=(mod_avg-base_avg)/base_avg*100 if base_avg>0 else 0
print(f"   Improvement:   {improvement:+.2f}%")

print(f"\n🏆 WIN COUNTS:")
print(f"   Modified LSTM wins: {results['mod_wins']}")
print(f"   Base LSTM wins: {results['base_wins']}")
print(f"   Ties: {results['ties']}")
total_wins=results['mod_wins']+results['base_wins']
if total_wins>0:
    print(f"   Modified win rate: {results['mod_wins']/total_wins*100:.1f}%")

print(f"\n⏱️ INFERENCE TIME (seconds per sentence):")
print(f"   Base LSTM:     {np.mean(results['base_times']):.4f}s")
print(f"   Modified LSTM: {np.mean(results['mod_times']):.4f}s")
time_diff=(np.mean(results['mod_times'])-np.mean(results['base_times']))/np.mean(results['base_times'])*100
print(f"   Speed difference: {time_diff:+.1f}%")

print(f"\n📝 TOP PREDICTED WORDS:")
top_words=sorted(results['word_predictions'].items(),key=lambda x:x[1]['mod']+x[1]['base'],reverse=True)[:10]
for word,counts in top_words:
    total=counts['base']+counts['mod']
    base_pct=counts['base']/total*100 if total>0 else 0
    mod_pct=counts['mod']/total*100 if total>0 else 0
    print(f"   '{word}': Base={base_pct:.0f}%, Modified={mod_pct:.0f}%")

print(f"\n📏 PERFORMANCE BY SENTENCE LENGTH:")
lengths=sorted(results['length_performance'].keys())
for l in lengths[:8]:
    data=results['length_performance'][l]
    if data['total']>3:
        base_pct=data['base']/data['total']*100
        mod_pct=data['mod']/data['total']*100
        winner="MODIFIED" if mod_pct>base_pct else "BASE"
        print(f"   Length {l:2d} ({data['total']:3d} samples): Modified wins {mod_pct:5.1f}% | Base wins {base_pct:5.1f}% | Winner: {winner}")

print("\n"+"="*70)
print("FINAL VERDICT")
print("="*70)

if mod_avg>base_avg:
    print(f"\n✅ MODIFIED LSTM (WITH WORKING MEMORY CONNECTIONS) IS BETTER")
    print(f"   {improvement:.1f}% improvement over Base LSTM")
    print("   Working Memory Connections (tanh(W_ic·c)) improve gate control")
elif base_avg>mod_avg:
    print(f"\n⚠️ BASE LSTM IS {abs(improvement):.1f}% BETTER")
else:
    print("\n🤝 BOTH MODELS PERFORM SIMILARLY")

print("\n"+"="*70)
print("SAVING GRAPHS")
print("="*70)

plt.figure(figsize=(16,6))

plt.subplot(2,3,1)
bins=np.linspace(0,1,20)
plt.hist(results['base_scores'],bins,alpha=0.5,label='Base LSTM',color='#3498db',edgecolor='black')
plt.hist(results['mod_scores'],bins,alpha=0.5,label='Modified LSTM',color='#e74c3c',edgecolor='black')
plt.xlabel('Accuracy Score',fontsize=11)
plt.ylabel('Frequency',fontsize=11)
plt.title('Score Distribution Comparison',fontsize=12)
plt.legend(fontsize=10)
plt.grid(True,alpha=0.3)

plt.subplot(2,3,2)
categories=['Base LSTM','Modified LSTM']
scores=[base_avg*100,mod_avg*100]
errors=[base_std*100,mod_std*100]
colors=['#3498db','#e74c3c']
bars=plt.bar(categories,scores,color=colors,edgecolor='black',linewidth=1.5,yerr=errors,capsize=5)
plt.ylabel('Accuracy Score (%)',fontsize=11)
plt.title('Average Accuracy Comparison',fontsize=12)
for bar,score in zip(bars,scores):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{score:.1f}%',ha='center',fontsize=11,fontweight='bold')
plt.ylim(0,max(scores)+15)

plt.subplot(2,3,3)
win_data=[results['mod_wins'],results['base_wins'],results['ties']]
win_labels=['Modified Wins','Base Wins','Ties']
win_colors=['#2ecc71','#e74c3c','#95a5a6']
plt.pie(win_data,labels=win_labels,colors=win_colors,autopct='%1.1f%%',startangle=90,explode=(0.05,0.05,0))
plt.title('Head-to-Head Win Distribution',fontsize=12)

plt.subplot(2,3,4)
time_data=[np.mean(results['base_times'])*1000,np.mean(results['mod_times'])*1000]
time_colors=['#3498db','#e74c3c']
time_bars=plt.bar(categories,time_data,color=time_colors,edgecolor='black',linewidth=1.5)
plt.ylabel('Time (milliseconds)',fontsize=11)
plt.title('Inference Speed Comparison',fontsize=12)
for bar,t in zip(time_bars,time_data):
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{t:.1f}ms',ha='center',fontsize=10)

plt.subplot(2,3,5)
length_vals=[]
base_len_vals=[]
mod_len_vals=[]
for l in lengths:
    data=results['length_performance'][l]
    if data['total']>5:
        length_vals.append(l)
        base_len_vals.append(data['base']/data['total']*100 if data['total']>0 else 0)
        mod_len_vals.append(data['mod']/data['total']*100 if data['total']>0 else 0)
plt.plot(length_vals,base_len_vals,'b-o',label='Base LSTM',linewidth=2,markersize=6)
plt.plot(length_vals,mod_len_vals,'r-s',label='Modified LSTM',linewidth=2,markersize=6)
plt.xlabel('Sentence Length (words)',fontsize=11)
plt.ylabel('Win Rate (%)',fontsize=11)
plt.title('Performance by Sentence Length',fontsize=12)
plt.legend(fontsize=10)
plt.grid(True,alpha=0.3)

plt.subplot(2,3,6)
top_5_words=list(results['word_predictions'].items())[:5]
word_names=[w[:12] for w,_ in top_5_words]
base_counts=[c['base'] for _,c in top_5_words]
mod_counts=[c['mod'] for _,c in top_5_words]
x=np.arange(len(word_names))
width=0.35
plt.bar(x-width/2,base_counts,width,label='Base LSTM',color='#3498db')
plt.bar(x+width/2,mod_counts,width,label='Modified LSTM',color='#e74c3c')
plt.xlabel('Words',fontsize=11)
plt.ylabel('Times Predicted',fontsize=11)
plt.title('Top Word Predictions',fontsize=12)
plt.xticks(x,word_names,rotation=15,ha='right')
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('models/full_evaluation_results.png',dpi=150,bbox_inches='tight')
plt.show()

print("\n✅ Graphs saved to 'models/full_evaluation_results.png'")

with open('models/full_evaluation_results.txt','w')as f:
    f.write("FULL EVALUATION RESULTS\n")
    f.write("="*50+"\n")
    f.write(f"Test sentences: {len(test_sentences)}\n")
    f.write(f"\nBASE LSTM:\n")
    f.write(f"  Avg Score: {base_avg:.4f}\n")
    f.write(f"  Std Dev: {base_std:.4f}\n")
    f.write(f"  Wins: {results['base_wins']}\n")
    f.write(f"  Avg Time: {np.mean(results['base_times']):.4f}s\n")
    f.write(f"\nMODIFIED LSTM:\n")
    f.write(f"  Avg Score: {mod_avg:.4f}\n")
    f.write(f"  Std Dev: {mod_std:.4f}\n")
    f.write(f"  Wins: {results['mod_wins']}\n")
    f.write(f"  Avg Time: {np.mean(results['mod_times']):.4f}s\n")
    f.write(f"\nIMPROVEMENT: {improvement:+.2f}%\n")

print("\n✅ Results saved to 'models/full_evaluation_results.txt'")
print("\n"+"="*70)
print("EVALUATION COMPLETE!")
print("="*70)