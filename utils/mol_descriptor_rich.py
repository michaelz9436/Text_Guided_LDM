import os
import csv
import glob
import random
import argparse
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

import prompt_vocab as vocab

#
class Config:
    SDF_DIR = "/home/course_project/MolFLAE/Latent_Experiments/data/latent_experiment/ex2"
    OUTPUT_CSV = "text2mol_prompts_rich_upgraded.csv"
    
    N_VARIANTS = 5               
    SEED = 42                 
    FEATURE_DROPOUT_RATE = 0.1  
    ENFORCE_MIN_INFO = True      
    GRANULARITY = 0.6         
    LANGUAGE_RICHNESS = 0.8   
    MAX_FGS_TO_MENTION = 4     
    PROB_EXACT_STRUCTURAL_COUNT = 0.25 

    QUOTE_ALL_STRINGS = True    


FUNCTIONAL_GROUPS: Dict[str, str] = {
    "hydroxyl":    "[OX2H]",
    "carboxyl":    "[CX3](=O)[OX2H1]",
    "primary amine": "[NX3;H2;!$(NC=O)]",
    "secondary amine": "[NX3;H1;!$(NC=O)]",
    "amide bond":  "[NX3][CX3](=[OX1])[#6]",
    "ester group": "[#6][CX3](=O)[OX2H0][#6]",
    "ether linkage": "[OD2]([#6])[#6]",
    "aldehyde":    "[CX3H1](=O)[#6]",
    "ketone":      "[#6][CX3](=O)[#6]",
    "nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "cyano group": "[#6]C#N",
    "sulfonyl":    "[SX4](=O)(=O)",
    "thiol":       "[SX2H]",
    "thioether":   "[#6][SX2][#6]",
    "F":           "[F]",
    "Cl":          "[Cl]",
    "Br":          "[Br]",
    "I":           "[I]",
    "epoxide":     "C1OC1",
}

RING_SMARTS: Dict[str, str] = {
    "benzene":     "c1ccccc1",
    "pyridine":    "c1ccncc1",
    "pyrrole":     "c1cc[nH]c1",
    "furan":       "c1ccoc1",
    "thiophene":   "c1ccsc1",
    "cyclohexane": "C1CCCCC1",
    "cyclopentane":"C1CCCC1",
}

def _count(mol: Chem.Mol, smarts: str) -> int:
    pat = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(pat)) if pat else 0

def _extract_raw(mol: Chem.Mol) -> Dict:
    props = {}
    props["fg"] = {k: _count(mol, sma) for k, sma in FUNCTIONAL_GROUPS.items() if _count(mol, sma) > 0}
    props["ring_types"] = {k: _count(mol, sma) for k, sma in RING_SMARTS.items() if _count(mol, sma) > 0}
    props["n_aromatic"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    props["n_aliphatic"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    props["tpsa"] = Descriptors.TPSA(mol)
    props["logp"] = Descriptors.MolLogP(mol)
    props["mw"] = Descriptors.ExactMolWt(mol)
    props["rot_bonds"] = Lipinski.NumRotatableBonds(mol)
    return props


def _format_count(n: int, is_structural: bool, rng: random.Random) -> str:
    """
    智能格式化数量：
    - 如果是核心结构 (is_structural=True)，有概率给出完全确定的数字 (模拟人类精确设计)。
    - 如果不是或未命中概率，则生成模糊数量。
    """
    if is_structural and rng.random() < Config.PROB_EXACT_STRUCTURAL_COUNT:
        if n == 1: return rng.choice(["1", "one", "exactly one", "a single"])
        else: return rng.choice([str(n), f"exactly {n}"])

    # 模糊
    if Config.GRANULARITY > 0.5:
        if n == 1: return rng.choice(["a single", "one", "an", "a"])
        elif n == 2: return rng.choice(["two", "a couple of", "a pair of"])
        elif n <= 4: return rng.choice([f"about {n}", f"around {n}", str(n), "a few"])
        else: return rng.choice([str(n), "multiple", "several"])
    else:
        if n == 1: return rng.choice(["a", "an", "some"])
        elif n <= 3: return rng.choice(["a few", "some", "a couple of"])
        else: return rng.choice(["multiple", "numerous", "many"])

def get_physical_properties(props: Dict, rng: random.Random) -> List[str]:
    """物理与几何属性描述 (永远不使用具体数字，彻底依赖词汇库中的形容词)"""
    adjectives = []
    
    # 1. Size
    mw = props["mw"]
    if mw > 450: adjectives.append(rng.choice(vocab.ADJECTIVES["large"]))
    elif mw < 250: adjectives.append(rng.choice(vocab.ADJECTIVES["small"]))
    elif rng.random() < 0.3: adjectives.append(rng.choice(vocab.ADJECTIVES["medium"]))
    
    # 2. Polarity / Lipophilicity
    logp = props["logp"]
    tpsa = props["tpsa"]
    if logp > 3.0 or tpsa < 40:
        adjectives.append(rng.choice(vocab.ADJECTIVES["lipophilic"]))
    elif logp < 1.0 or tpsa > 80:
        adjectives.append(rng.choice(vocab.ADJECTIVES["hydrophilic"]))

    # 3. Flexibility
    rot = props["rot_bonds"]
    if rot <= 2:
        adjectives.append(rng.choice(vocab.ADJECTIVES["rigid"]))
    elif rot >= 7:
        adjectives.append(rng.choice(vocab.ADJECTIVES["flexible"]))

    rng.shuffle(adjectives)
    return adjectives

def get_ring_descriptions(props: Dict, rng: random.Random) -> str:
    """生成环系统描述 (加入精确计数的芳香环/脂肪环统计)"""
    na, nal = props["n_aromatic"], props["n_aliphatic"]
    specific_rings = [k for k, v in props["ring_types"].items()]
    
    # 判断基本骨架形态
    if na == 0 and nal == 0:
        base_scaffold = rng.choice(vocab.RING_SCAFFOLDS["acyclic"])
    elif na > 0 and nal == 0:
        base_scaffold = rng.choice(vocab.RING_SCAFFOLDS["fully_aromatic"])
    elif na == 0 and nal > 0:
        base_scaffold = rng.choice(vocab.RING_SCAFFOLDS["aliphatic_cyclic"])
    else:
        base_scaffold = rng.choice(vocab.RING_SCAFFOLDS["mixed"])
        
    # ✨ 核心特性：人类意图的特定数量 (比如明确提出含有几个芳香环)
    if rng.random() < Config.PROB_EXACT_STRUCTURAL_COUNT:
        intents = []
        if na > 0: intents.append(f"{_format_count(na, True, rng)} aromatic ring{'s' if na!=1 else ''}")
        if nal > 0: intents.append(f"{_format_count(nal, True, rng)} aliphatic ring{'s' if nal!=1 else ''}")
        
        if intents:
            connector = rng.choice([" containing ", " composed of ", " with "])
            base_scaffold += connector + " and ".join(intents)

    # 具体环的具体点名 (比如直接点名苯环)
    elif specific_rings and rng.random() < (1.0 - Config.FEATURE_DROPOUT_RATE):
        rng.shuffle(specific_rings)
        chosen = specific_rings[:2]
        
        ring_mentions = []
        for r in chosen:
            c = props["ring_types"][r]
            q = _format_count(c, is_structural=True, rng=rng)
            ring_mentions.append(f"{q} {r} ring{'s' if c>1 and str(c) in q else ''}")
            
        connector = rng.choice([" containing ", " that includes ", " featuring "])
        if len(ring_mentions) == 1:
            base_scaffold += connector + ring_mentions[0]
        else:
            base_scaffold += connector + " and ".join(ring_mentions)
            
    return base_scaffold

def get_fg_descriptions(props: Dict, rng: random.Random) -> List[str]:
    """生成官能团描述"""
    fgs = list(props["fg"].items())
    rng.shuffle(fgs)
    fg_texts = []
    
    halogens = {"F", "Cl", "Br", "I"}
    present_halogens = [(k, v) for k, v in fgs if k in halogens]
    other_fgs = [(k, v) for k, v in fgs if k not in halogens]
    
    # 卤素处理
    if present_halogens:
        if Config.GRANULARITY > 0.6 and rng.random() < 0.7:
            for k, v in present_halogens:
                q = _format_count(v, is_structural=True, rng=rng)
                fg_texts.append(f"{q} {k} atom{'s' if v>1 and str(v) in q else ''}")
        else:
            fg_texts.append(rng.choice(vocab.HALOGEN_DESCRIPTIONS))
            
    # 常规官能团
    for k, v in other_fgs[:Config.MAX_FGS_TO_MENTION]:
        q = _format_count(v, is_structural=True, rng=rng)
        
        # 为了语法通顺：如果前缀只是 "a" 或 "one" 并且单词以 s 结尾(少见)，或者已经带有修饰符
        plural_suffix = "s" if v > 1 and q not in ["a", "an", "one", "a single"] and not k.endswith('s') else ""
        fg_texts.append(f"{q} {k}{plural_suffix}")
            
    return fg_texts


def _assemble_rich_prompt(props: Dict, rng: random.Random) -> str:
    adjectives = get_physical_properties(props, rng)
    ring_desc = get_ring_descriptions(props, rng)
    fgs = get_fg_descriptions(props, rng)
    
    # Dropout
    if rng.random() < Config.FEATURE_DROPOUT_RATE: adjectives = []
    if rng.random() < Config.FEATURE_DROPOUT_RATE: ring_desc = ""
    if rng.random() < Config.FEATURE_DROPOUT_RATE: fgs = []
    
    # Enforce min info
    if Config.ENFORCE_MIN_INFO and sum([bool(adjectives), bool(ring_desc), bool(fgs)]) < 2:
        adjectives = get_physical_properties(props, rng)
        fgs = get_fg_descriptions(props, rng)
        ring_desc = get_ring_descriptions(props, rng)

    # 1. 组装形容词
    adj_str = ", ".join(adjectives) if adjectives else ""
    
    # 2. 组装官能团
    fg_str = ""
    if fgs:
        if len(fgs) == 1: fg_str = fgs[0]
        elif len(fgs) == 2: fg_str = f"{fgs[0]} and {fgs[1]}"
        else: fg_str = ", ".join(fgs[:-1]) + f", and {fgs[-1]}"
        
    noun = rng.choice(vocab.NOUNS)
    starter = rng.choice(vocab.SENTENCE_STARTERS)
    
    templates = []
    
    # 模式A：高复杂度的复合长句
    if Config.LANGUAGE_RICHNESS >= 0.5 and (adj_str or ring_desc or fg_str):
        if adj_str:
            # 解决 a/an 的问题
            first_letter = adj_str.lower()[0]
            if starter in ["A", "This is a", "It's a", "Looks like a", "Essentially, it is a"] and first_letter in "aeiou":
                starter = starter[:-1] + "an"
            p1 = f"{starter} {adj_str} {noun}"
        else:
            p1 = f"{starter} {noun}"
            
        p2 = f" characterized by its {ring_desc}" if ring_desc else ""
        
        p3 = ""
        if fg_str:
            # 从外部词典获取随机连接词
            transition = rng.choice(vocab.TRANSITIONS_TO_FG)
            p3 = f"{transition}{fg_str}"
            
        templates.append((p1 + p2 + p3).strip())
        
    # 模式B：分成两句
    if Config.LANGUAGE_RICHNESS >= 0.2:
        first_letter = adj_str.lower()[0] if adj_str else noun[0]
        art = "An" if first_letter in "aeiou" else "A"
        sent1 = f"{art} {adj_str} {noun}" if adj_str else f"The {noun}"
        if ring_desc: 
            sent1 += f" with a {ring_desc}"
        sent1 += "."
        
        sent2 = ""
        if fg_str:
            sent2 = rng.choice(vocab.TRANSITIONS_BETWEEN_SENTENCES) + fg_str + "."
            
        templates.append((sent1 + sent2).strip())
        
    # 模式C：极简标签流
    if Config.LANGUAGE_RICHNESS <= 0.6:
        tags = []
        if adj_str: tags.append(adj_str)
        if ring_desc: tags.append(ring_desc)
        if fg_str: tags.append(f"contains {fg_str}")
        rng.shuffle(tags)
        separator = rng.choice(vocab.TAG_SEPARATORS)
        templates.append(separator.join(tags).lower())
        
    # 容错
    if not templates:
        return "A chemical compound."

    # 选择、清理、润色
    chosen = rng.choice(templates)
    chosen = chosen.replace("a acyclic", "an acyclic").replace("a aromatic", "an aromatic").replace("a aliphatic", "an aliphatic")
    chosen = chosen.replace("  ", " ").strip()
    
    if "|" not in chosen and ";" not in chosen and chosen and chosen[0].islower():
        chosen = chosen[0].upper() + chosen[1:]
        
    if rng.random() < 0.2:
        chosen = chosen.rstrip('.')
        
    return chosen

def generate_prompt_variants(mol: Chem.Mol, n_variants: int, seed: int) -> List[str]:
    props = _extract_raw(mol)
    rng = random.Random(seed)
    return [_assemble_rich_prompt(props, rng) for _ in range(n_variants)]


def batch_process():
    print("=" * 60)
    print("🚀 Text2Mol 拟人化描述符生成器 (全参数配置 + 外部词库版)")
    print(f"📁 输入目录: {Config.SDF_DIR}")
    print(f"💾 输出文件: {Config.OUTPUT_CSV}")
    print(f"⚙️  配置预览:")
    print(f"    - N_VARIANTS: {Config.N_VARIANTS}")
    print(f"    - LANGUAGE_RICHNESS: {Config.LANGUAGE_RICHNESS}")
    print(f"    - GRANULARITY: {Config.GRANULARITY}")
    print(f"    - PROB_EXACT_STRUCTURAL_COUNT: {Config.PROB_EXACT_STRUCTURAL_COUNT} (保留精确结构数量的概率)")
    print("=" * 60)
    
    if not os.path.exists(Config.SDF_DIR):
        print(f"❌ 找不到输入目录: {Config.SDF_DIR}")
        return

    sdf_files = sorted(glob.glob(os.path.join(Config.SDF_DIR, "*.sdf")))
    if not sdf_files:
        print(f"⚠️ 目录中没有找到 .sdf 文件!")
        return

    header = ["filename", "smiles"] + [f"prompt_variant_{i+1}" for i in range(Config.N_VARIANTS)]
    rows = []
    success_count = 0

    for i, path in enumerate(sdf_files):
        fname = os.path.basename(path)
        try:
            supplier = Chem.SDMolSupplier(path, removeHs=True)
            mol = next((m for m in supplier if m is not None), None)
            
            if mol is None:
                rows.append([fname, "FAILED_TO_LOAD"] + [""] * Config.N_VARIANTS)
                continue
            
            smiles = Chem.MolToSmiles(mol)
            prompts = generate_prompt_variants(mol, n_variants=Config.N_VARIANTS, seed=Config.SEED + i)
            
            rows.append([fname, smiles] + prompts)
            success_count += 1
            
            # 打印样例展示
            if success_count <= 3:
                print(f"\n▶ 示例 {success_count} | 文件: {fname}")
                for v, p in enumerate(prompts[:3]):
                    print(f"   [{v+1}]: {p}")
                if Config.N_VARIANTS > 3:
                    print(f"   ... (已隐藏剩余 {Config.N_VARIANTS-3} 个)")

        except Exception as e:
            print(f"❌ 处理 {fname} 出错: {e}")
            rows.append([fname, f"ERROR: {e}"] + [""] * Config.N_VARIANTS)

    # ---------------- CSV 写入与强制包裹双引号 ----------------
    quoting_style = csv.QUOTE_ALL if Config.QUOTE_ALL_STRINGS else csv.QUOTE_MINIMAL
    
    with open(Config.OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=quoting_style)
        writer.writerow(header)
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"🎉 任务完成! 成功提取 {success_count}/{len(sdf_files)} 个分子。")
    print(f"💾 数据已保存至: {os.path.abspath(Config.OUTPUT_CSV)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成高信息量 Text2Mol 描述符")
    parser.add_argument("--dir", type=str, default=Config.SDF_DIR)
    parser.add_argument("--out", type=str, default=Config.OUTPUT_CSV)
    parser.add_argument("--variants", type=int, default=Config.N_VARIANTS)
    args = parser.parse_args()

    Config.SDF_DIR = args.dir
    Config.OUTPUT_CSV = args.out
    Config.N_VARIANTS = args.variants

    batch_process()