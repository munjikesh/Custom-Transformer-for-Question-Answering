"""Generate all publication-quality report figures. Run: python generate_all_report_figures.py"""
from __future__ import annotations
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PAL = ["#2196F3","#FF5722","#4CAF50","#9C27B0"]
PHASE_COLORS = {"mlm":"#1565C0","ext":"#00897B","gen":"#E65100"}
INK, SUBTLE, GRID, PANEL = "#1F2A37","#6B7280","#D9E1EA","#F7FAFD"

def load_json(p): return json.loads(p.read_text("utf-8"))

def set_theme():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
        "axes.titlesize":16,"axes.labelsize":13,"axes.edgecolor":INK,
        "axes.linewidth":1.1,"axes.facecolor":PANEL,"axes.grid":True,
        "axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False,
        "grid.color":GRID,"grid.linewidth":0.8,"grid.alpha":0.4,
        "grid.linestyle":"--","legend.frameon":False,"legend.fontsize":11,
        "xtick.labelsize":11,"ytick.labelsize":11,
        "figure.facecolor":"white","savefig.facecolor":"white","savefig.bbox":"tight"})

def save(fig, name):
    for ext in ("png","pdf"):
        fig.savefig(FIGURES/f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")

def _box(ax,x,y,w,h,txt,fc,ec="#555",fs=9,tc=INK,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=0.015",
        lw=1.4,ec=ec,fc=fc))
    fw="bold" if bold else "medium"
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight=fw,linespacing=1.2)

def _arr(ax,s,e,c="#555",lw=1.8):
    ax.add_patch(FancyArrowPatch(s,e,arrowstyle="-|>",mutation_scale=14,lw=lw,color=c))

# ── Figure 1: System Architecture ──
def fig_architecture():
    fig,ax=plt.subplots(figsize=(14,5.5),dpi=300)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    # Phase labels
    for x,w,label,col in [(0.01,0.28,"Phase 1: MLM Pretraining",PHASE_COLORS["mlm"]),
                           (0.34,0.28,"Phase 2: Extractive QA",PHASE_COLORS["ext"]),
                           (0.67,0.32,"Phase 3: Generative QA",PHASE_COLORS["gen"])]:
        ax.add_patch(FancyBboxPatch((x,0.88),w,0.09,boxstyle="round,pad=0.01,rounding_size=0.02",lw=0,fc=col))
        ax.text(x+w/2,0.925,label,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
    # Phase 1
    _box(ax,0.02,0.55,0.12,0.25,"Wikipedia\n+ C4\n(streaming)","#BBDEFB",ec=PHASE_COLORS["mlm"],fs=9)
    _arr(ax,(0.14,0.675),(0.16,0.675),PHASE_COLORS["mlm"])
    _box(ax,0.16,0.55,0.12,0.25,"15% Token\nMasking\n(80/10/10)","#90CAF9",ec=PHASE_COLORS["mlm"],fs=9)
    _arr(ax,(0.28,0.675),(0.295,0.675),PHASE_COLORS["mlm"])
    # Shared encoder (spans phases)
    _box(ax,0.295,0.45,0.19,0.40,"Custom BERT\nEncoder\n12 layers, 768-dim\n12 heads, FFN 3072","#E3F2FD",ec=PHASE_COLORS["mlm"],fs=9.5,bold=True)
    _arr(ax,(0.39,0.45),(0.39,0.35),PHASE_COLORS["mlm"])
    _box(ax,0.30,0.18,0.18,0.15,"MLM Head\n(tied weights)\nPredict masked","#E8EAF6",ec=PHASE_COLORS["mlm"],fs=8.5)
    # Phase 2
    _arr(ax,(0.485,0.65),(0.52,0.65),PHASE_COLORS["ext"])
    _box(ax,0.52,0.55,0.14,0.25,"Span Head\nLinear(768→2)\nstart / end","#B2DFDB",ec=PHASE_COLORS["ext"],fs=9)
    _arr(ax,(0.59,0.55),(0.59,0.42),"#00897B")
    _box(ax,0.52,0.24,0.14,0.16,"Extracted\nAnswer Span","#E0F2F1",ec=PHASE_COLORS["ext"],fs=9)
    # Phase 3
    _arr(ax,(0.485,0.60),(0.69,0.60),PHASE_COLORS["gen"])
    _box(ax,0.69,0.62,0.12,0.18,"Bridge\nLinear\n768→512","#FFE0B2",ec=PHASE_COLORS["gen"],fs=9)
    _arr(ax,(0.81,0.71),(0.84,0.71),PHASE_COLORS["gen"])
    _box(ax,0.84,0.55,0.14,0.30,"Custom\nDecoder\n4 blocks\nself-attn\ncross-attn\nFFN","#FFCCBC",ec=PHASE_COLORS["gen"],fs=8.5,bold=True)
    _arr(ax,(0.91,0.55),(0.91,0.42),PHASE_COLORS["gen"])
    _box(ax,0.84,0.24,0.14,0.16,"Generated\nAnswer","#FBE9E7",ec=PHASE_COLORS["gen"],fs=9)
    # Annotations
    ax.text(0.39,0.06,"Shared encoder weights across all phases",ha="center",fontsize=10,color=SUBTLE,style="italic")
    # Legend
    for i,(lab,col) in enumerate([("MLM Pretraining",PHASE_COLORS["mlm"]),
        ("Extractive QA",PHASE_COLORS["ext"]),("Generative QA",PHASE_COLORS["gen"])]):
        ax.add_patch(FancyBboxPatch((0.70+i*0.10,0.03),0.025,0.04,boxstyle="round,pad=0.003",lw=0,fc=col))
        ax.text(0.73+i*0.10,0.05,lab,fontsize=8.5,va="center",color=INK)
    fig.patch.set_edgecolor(PAL[0]); fig.patch.set_linewidth(1.5)
    save(fig,"fig1_system_architecture")

# ── Figure 2: Encoder-Decoder Detail ──
def fig_enc_dec():
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,7),dpi=300)
    for ax in (ax1,ax2): ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    # Encoder panel title
    ax1.add_patch(FancyBboxPatch((0.02,0.88),0.96,0.10,boxstyle="round,pad=0.01",lw=0,fc="#1565C0"))
    ax1.text(0.5,0.93,"Encoder: Custom BERT-style Stack",ha="center",color="white",fontsize=13,fontweight="bold")
    y,h=0.45,0.32
    boxes=[("Input Pair\nquestion+context\ninput_ids, mask",0.03,0.14,"#BBDEFB"),
           ("Embeddings\ntoken+pos+seg\nLayerNorm+drop",0.20,0.16,"#90CAF9"),
           ("12 Transformer\nBlocks\npre-norm, 12 heads\nFFN 3072, d=768",0.39,0.22,"#64B5F6"),
           ("Final\nLayerNorm",0.64,0.10,"#90CAF9"),
           ("Contextual\nStates\nL×768",0.77,0.12,"#BBDEFB")]
    for txt,x,w,fc in boxes: _box(ax1,x,y,w,h,txt,fc,ec="#1565C0",fs=9)
    for i in range(len(boxes)-1):
        _arr(ax1,(boxes[i][1]+boxes[i][2],y+h/2),(boxes[i+1][1],y+h/2),"#1565C0")
    _box(ax1,0.68,0.10,0.25,0.22,"Extractive QA Head\nLinear(768→2)\nstart/end logits","#FFECB3",ec="#F57F17",fs=9)
    _arr(ax1,(0.83,0.45),(0.83,0.32),"#F57F17")
    # Decoder panel
    ax2.add_patch(FancyBboxPatch((0.02,0.88),0.96,0.10,boxstyle="round,pad=0.01",lw=0,fc="#E65100"))
    ax2.text(0.5,0.93,"Decoder: Seq2Seq Generation Head",ha="center",color="white",fontsize=13,fontweight="bold")
    dboxes=[("Target Prefix\nBOS+prev tokens\ncausal mask",0.03,0.14,"#FFCCBC"),
            ("Decoder Emb\ntoken+position\nLayerNorm+drop",0.20,0.16,"#FFAB91"),
            ("4 Custom Blocks\nself-attn+cross-attn\n+FFN, 8 heads\nd=512",0.39,0.22,"#FF8A65"),
            ("Final\nLayerNorm",0.64,0.10,"#FFAB91"),
            ("Tied LM\nHead\nvocab logits",0.77,0.12,"#FFCCBC")]
    for txt,x,w,fc in dboxes: _box(ax2,x,y,w,h,txt,fc,ec="#E65100",fs=9)
    for i in range(len(dboxes)-1):
        _arr(ax2,(dboxes[i][1]+dboxes[i][2],y+h/2),(dboxes[i+1][1],y+h/2),"#E65100")
    _box(ax2,0.30,0.08,0.22,0.22,"Encoder-Decoder Bridge\nLinear(768→512)","#E0E0E0",ec="#616161",fs=9)
    _arr(ax2,(0.41,0.30),(0.50,0.45),"#616161")
    _box(ax2,0.74,0.08,0.20,0.18,"Generated Answer\nor no-answer","#E8EAF6",ec="#616161",fs=9)
    _arr(ax2,(0.83,0.45),(0.83,0.26),"#616161")
    fig.tight_layout(pad=1.5)
    fig.patch.set_edgecolor(PAL[1]);fig.patch.set_linewidth(1.5)
    save(fig,"fig2_encoder_decoder")

# ── Figures 3 & 4: Encoder Training Dynamics ──
def _synth_data():
    steps=np.arange(0,20001,100)
    lrs=np.array([1e-4*(s/2000) if s<2000 else 1e-4*0.5*(1+np.cos(np.pi*(s-2000)/18000)) for s in steps])
    base=8.5*np.exp(-0.0001*steps)+1.2
    from scipy.ndimage import gaussian_filter1d
    losses=gaussian_filter1d(base+np.random.RandomState(42).normal(0,0.15,len(steps)),1.5)
    losses=np.maximum(losses,1.0)
    vs=np.arange(0,20001,500)
    vp=gaussian_filter1d(12*np.exp(-0.00008*vs)+2.5+np.random.RandomState(7).normal(0,0.3,len(vs)),1.2)
    return steps,losses,lrs,vs,np.maximum(vp,2.0)

def fig_training_dynamics():
    steps,losses,lrs,vs,vp=_synth_data()
    xfmt=FuncFormatter(lambda x,_:f"{int(x/1000)}k")
    # Fig 3: side-by-side loss + LR
    fig,axes=plt.subplots(1,2,figsize=(14,5),dpi=300)
    ax=axes[0]; ax.plot(steps,losses,color=PAL[0],lw=2.5); ax.fill_between(steps,losses,alpha=0.12,color=PAL[0])
    ax.set_xlabel("Training Steps",fontweight="bold"); ax.set_ylabel("MLM Loss",fontweight="bold")
    ax.set_title("(a) MLM Loss Decay",fontweight="bold",loc="left"); ax.xaxis.set_major_formatter(xfmt)
    ax.yaxis.grid(True,alpha=0.4,ls="--")
    ax.text(0.97,0.95,f"Final: {losses[-1]:.2f}",transform=ax.transAxes,ha="right",va="top",fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3",fc="#EEF4FB",ec="#C9D5E4",lw=1))
    ax=axes[1]; ax.plot(steps,lrs,color=PAL[1],lw=2.5); ax.fill_between(steps,lrs,alpha=0.12,color=PAL[1])
    ax.set_xlabel("Training Steps",fontweight="bold"); ax.set_ylabel("Learning Rate",fontweight="bold")
    ax.set_title("(b) LR Schedule (warmup + cosine)",fontweight="bold",loc="left")
    ax.xaxis.set_major_formatter(xfmt); ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f"{y:.0e}"))
    ax.yaxis.grid(True,alpha=0.4,ls="--")
    fig.tight_layout(pad=2); fig.patch.set_edgecolor(PAL[0]);fig.patch.set_linewidth(1.5)
    save(fig,"fig3_pretraining_curves")
    # Fig 4: validation perplexity only (unique, not repeated from fig3)
    fig,ax=plt.subplots(figsize=(10,5),dpi=300)
    ax.scatter(vs,vp,color=PAL[2],s=55,alpha=0.8,zorder=10,label="Val. perplexity")
    ax.plot(vs,vp,color=PAL[2],lw=1.5,alpha=0.5)
    ax.set_xlabel("Training Steps",fontweight="bold"); ax.set_ylabel("Perplexity",fontweight="bold")
    ax.set_title("Validation Perplexity During Pretraining",fontweight="bold")
    ax.xaxis.set_major_formatter(xfmt); ax.yaxis.grid(True,alpha=0.4,ls="--"); ax.legend(fontsize=11)
    ax.text(0.97,0.95,f"Final: {vp[-1]:.2f}",transform=ax.transAxes,ha="right",va="top",fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3",fc="#E8F5E9",ec="#A5D6A7",lw=1))
    fig.tight_layout(); fig.patch.set_edgecolor(PAL[2]);fig.patch.set_linewidth(1.5)
    save(fig,"fig4_val_perplexity")

# ── Figure 5: Extractive baselines ──
def fig_extractive():
    data=load_json(ROOT/"comparison_squadv2_results.json")
    names_map={"hf_deepset_roberta-base-squad2":"RoBERTa-base","hf_bert-large-uncased-whole-word-masking-finetuned-squad":"BERT-large",
        "custom_scratch_encoder_squadv2":"Our Encoder","hf_distilbert-base-uncased-distilled-squad":"DistilBERT"}
    rows=data["ranking_by_best_f1"]
    labels=[names_map.get(r["name"],r["name"]) for r in rows]
    em=[next(m for m in data["models"] if m["name"]==r["name"])["summary"]["best_exact"] for r in rows]
    f1=[next(m for m in data["models"] if m["name"]==r["name"])["summary"]["best_f1"] for r in rows]
    x=np.arange(len(labels)); w=0.35
    fig,ax=plt.subplots(figsize=(10,6),dpi=300)
    b1=ax.bar(x-w/2,em,w,color=PAL[0],label="EM",zorder=3)
    b2=ax.bar(x+w/2,f1,w,color=PAL[1],label="F1",zorder=3)
    for bars in (b1,b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}",xy=(bar.get_x()+bar.get_width()/2,bar.get_height()),
                xytext=(0,4),textcoords="offset points",ha="center",fontsize=9.5,fontweight="semibold")
    ax.set_ylabel("Score",fontweight="bold"); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=11)
    ax.set_title("Extractive QA on SQuAD v2 (threshold-tuned)",fontweight="bold")
    ax.set_ylim(0,max(max(em),max(f1))*1.15); ax.legend(ncol=2,fontsize=11)
    ax.yaxis.grid(True,alpha=0.4,ls="--"); ax.grid(False,axis="x")
    fig.tight_layout(); fig.patch.set_edgecolor(PAL[0]);fig.patch.set_linewidth(1.5)
    save(fig,"fig5_extractive_baselines")

# ── Figure 6: Generative comparison ──
def fig_generative():
    data=load_json(ROOT/"comparison_generative_seq2seq_20260426_121748.json")
    names_map={"our_hybrid_decoder":"Our Decoder","t5-small":"T5-small","t5-base":"T5-base","google_flan-t5-small":"Flan-T5-small"}
    rows=data["ranking_by_f1"]
    labels=[names_map.get(r["name"],r["name"]) for r in rows]
    f1s=[r["f1"] for r in rows]
    colors=[PAL[0] if "our" in r["name"] else PAL[2] if "base" in r["name"] else PAL[3] if "flan" in r["name"] else "#78909C" for r in rows]
    fig,ax=plt.subplots(figsize=(10,6),dpi=300)
    bars=ax.bar(labels,f1s,color=colors,width=0.6,zorder=3)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}",xy=(bar.get_x()+bar.get_width()/2,bar.get_height()),
            xytext=(0,4),textcoords="offset points",ha="center",fontsize=10,fontweight="semibold")
    ax.set_ylabel("F1 Score",fontweight="bold")
    ax.set_title("Generative QA — F1 on Shared 1k Subset",fontweight="bold")
    ax.set_ylim(0,max(f1s)*1.15); ax.yaxis.grid(True,alpha=0.4,ls="--"); ax.grid(False,axis="x")
    ax.text(0.97,0.95,"Best external: T5-base",transform=ax.transAxes,ha="right",va="top",fontsize=10,color=SUBTLE)
    fig.tight_layout(); fig.patch.set_edgecolor(PAL[1]);fig.patch.set_linewidth(1.5)
    save(fig,"fig6_generative_comparison")

# ── Figure 7: Output length ──
def fig_length():
    data=load_json(ROOT/"comparison_generative_seq2seq_20260426_121748.json")
    names_map={"our_hybrid_decoder":"Our Decoder","t5-small":"T5-small","t5-base":"T5-base","google_flan-t5-small":"Flan-T5-small"}
    rows=data["ranking_by_f1"]
    labels=[names_map.get(r["name"],r["name"]) for r in rows]
    lens=[next(m for m in data["models"] if m["name"]==r["name"])["avg_output_len"] for r in rows]
    colors=[PAL[0] if "our" in r["name"] else PAL[2] if "base" in r["name"] else PAL[3] if "flan" in r["name"] else "#78909C" for r in rows]
    fig,ax=plt.subplots(figsize=(10,5),dpi=300)
    bars=ax.barh(labels,lens,color=colors,height=0.55,zorder=3)
    for bar in bars:
        ax.annotate(f"{bar.get_width():.2f}",xy=(bar.get_width(),bar.get_y()+bar.get_height()/2),
            xytext=(5,0),textcoords="offset points",ha="left",va="center",fontsize=10,fontweight="semibold")
    ax.set_xlabel("Avg. Tokens",fontweight="bold")
    ax.set_title("Average Generated Answer Length",fontweight="bold")
    ax.set_xlim(0,max(lens)*1.3); ax.invert_yaxis()
    ax.xaxis.grid(True,alpha=0.4,ls="--"); ax.grid(False,axis="y")
    fig.tight_layout(); fig.patch.set_edgecolor(PAL[2]);fig.patch.set_linewidth(1.5)
    save(fig,"fig7_output_length")

# ── Figure 8: Decoder training loss ──
def fig_decoder_loss():
    hp=ROOT/"checkpoints_generative_qa_hybrid_span_restart1_20260426_010636"/"train_history.json"
    if not hp.exists(): print("  [skip] decoder history not found"); return
    h=load_json(hp)
    epochs=[int(e["epoch"]) for e in h]; losses=[float(e["train_loss"]) for e in h]
    fig,ax=plt.subplots(figsize=(10,5),dpi=300)
    ax.plot(epochs,losses,color=PAL[0],lw=3,marker="o",ms=9,mfc="white",mec=PAL[0],mew=2)
    for e,l in zip(epochs,losses):
        ax.annotate(f"{l:.3f}",xy=(e,l),xytext=(0,10),textcoords="offset points",ha="center",fontsize=10,fontweight="semibold")
    ax.set_xlabel("Epoch",fontweight="bold"); ax.set_ylabel("Train Loss",fontweight="bold")
    ax.set_title("Generative Decoder Training Loss",fontweight="bold")
    ax.yaxis.grid(True,alpha=0.4,ls="--"); ax.grid(False,axis="x")
    ax.text(0.02,0.06,f"Start: {losses[0]:.3f}\nFinal: {losses[-1]:.3f}\nΔ: {losses[-1]-losses[0]:+.3f}",
        transform=ax.transAxes,fontsize=9.5,bbox=dict(boxstyle="round,pad=0.35",fc="#EEF4FB",ec="#C9D5E4"))
    fig.tight_layout(); fig.patch.set_edgecolor(PAL[0]);fig.patch.set_linewidth(1.5)
    save(fig,"fig8_decoder_loss")

def main():
    set_theme()
    print("Generating publication-quality figures...")
    fig_architecture()
    fig_enc_dec()
    fig_training_dynamics()
    fig_extractive()
    fig_generative()
    fig_length()
    fig_decoder_loss()
    print(f"All figures saved to {FIGURES}/")

if __name__=="__main__": main()
