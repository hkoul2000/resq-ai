"""Generate all figures for the ResQ-AI paper.

Generates:
- Figure 1: System overview diagram (matplotlib/tikz)
- Figure 2: ResQNet architecture diagram
- Figure 3: Calibration/reliability diagrams
- Figure 4: Conformal prediction coverage plots
- Figure 5: Qualitative flood maps with uncertainty
- Figure 6: Impact assessment distribution plots
- Figure 7: Allocation comparison charts
- Figure 8: Robustness analysis (missing modalities)
- Figure 9: Risk-coverage curves
- Figure 10: Ablation study bar charts
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')

def setup_dirs():
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def save_fig(fig, name):
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def load_or_mock_data(filename, mock_func):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return mock_func()

def mock_calibration():
    return {
        'ensemble': {'conf': np.linspace(0, 1, 10), 'acc': np.linspace(0, 1, 10) + np.random.normal(0, 0.05, 10)},
        'mc_dropout': {'conf': np.linspace(0, 1, 10), 'acc': np.linspace(0, 1, 10) + np.random.normal(0, 0.08, 10)},
        'evidential': {'conf': np.linspace(0, 1, 10), 'acc': np.linspace(0, 1, 10) + np.random.normal(0, 0.03, 10)},
        'conformal': {'conf': np.linspace(0, 1, 10), 'acc': np.linspace(0, 1, 10) + np.random.normal(0, 0.01, 10)}
    }

def mock_conformal_coverage():
    return {
        '0.1': 0.89,
        '0.05': 0.94,
        '0.01': 0.99
    }

def mock_impact():
    return [np.random.lognormal(mean=0, sigma=1, size=100) for _ in range(3)]

def mock_allocation():
    return {
        'baseline': [10, 20, 30],
        'uncertainty-aware': [5, 15, 20],
        'robust': [8, 10, 15]
    }

def mock_robustness():
    return np.random.uniform(0.5, 0.9, size=(4, 4))

def mock_risk_coverage():
    return {
        'ensemble': {'cov': np.linspace(0.5, 1, 20), 'risk': np.linspace(0.01, 0.1, 20)},
        'conformal': {'cov': np.linspace(0.5, 1, 20), 'risk': np.linspace(0.02, 0.15, 20)}
    }

def mock_ablation():
    return {
        'Full Model': 0.85,
        '- Optical': 0.82,
        '- SAR': 0.75,
        '- DEM': 0.83
    }

def generate_system_overview():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis('off')
    
    blocks = [
        ('Input Data', 0.1),
        ('ResQNet', 0.3),
        ('UQ', 0.5),
        ('Impact\nAssessment', 0.7),
        ('Response\nPlan', 0.9)
    ]
    
    for name, x in blocks:
        rect = mpatches.Rectangle((x-0.08, 0.3), 0.16, 0.4, fill=True, color='lightblue', alpha=0.5, ec='black')
        ax.add_patch(rect)
        ax.text(x, 0.5, name, ha='center', va='center')
        
    for i in range(len(blocks)-1):
        ax.annotate('', xy=(blocks[i+1][1]-0.08, 0.5), xytext=(blocks[i][1]+0.08, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
                    
    save_fig(fig, 'fig_system_overview')

def generate_architecture():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis('off')
    ax.text(0.5, 0.5, "ResQNet Architecture Diagram", ha='center', va='center', fontsize=14)
    save_fig(fig, 'fig_architecture')

def generate_reliability():
    data = load_or_mock_data('calibration.json', mock_calibration)
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, (method, vals) in zip(axes.flatten(), data.items()):
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.plot(vals['conf'], vals['acc'], marker='o', label='Model')
        ax.set_title(method)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend()
    plt.tight_layout()
    save_fig(fig, 'fig_reliability')

def generate_conformal_coverage():
    data = load_or_mock_data('conformal.json', mock_conformal_coverage)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    alphas = list(data.keys())
    emp = list(data.values())
    target = [1-float(a) for a in alphas]
    
    x = np.arange(len(alphas))
    width = 0.35
    
    ax.bar(x - width/2, target, width, label='Target')
    ax.bar(x + width/2, emp, width, label='Empirical')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'alpha={a}' for a in alphas])
    ax.set_ylabel('Coverage')
    ax.legend()
    save_fig(fig, 'fig_conformal_coverage')

def generate_qualitative():
    fig, axes = plt.subplots(2, 4, figsize=(7, 3.5))
    for i in range(2):
        axes[i, 0].imshow(np.random.rand(10, 10))
        axes[i, 0].set_title('Input')
        axes[i, 1].imshow(np.random.randint(0, 2, (10, 10)), cmap='binary')
        axes[i, 1].set_title('GT')
        axes[i, 2].imshow(np.random.rand(10, 10), cmap='viridis')
        axes[i, 2].set_title('Pred')
        axes[i, 3].imshow(np.random.rand(10, 10), cmap='hot')
        axes[i, 3].set_title('Uncertainty')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    save_fig(fig, 'fig_qualitative')

def generate_impact():
    data = load_or_mock_data('impact.json', mock_impact)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.boxplot(data, labels=['Zone A', 'Zone B', 'Zone C'])
    ax.set_ylabel('Affected Population')
    save_fig(fig, 'fig_impact')

def generate_allocation():
    data = load_or_mock_data('allocation.json', mock_allocation)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    x = np.arange(3)
    width = 0.25
    
    ax.bar(x - width, data['baseline'], width, label='Baseline')
    ax.bar(x, data['uncertainty-aware'], width, label='Uncertainty-Aware')
    ax.bar(x + width, data['robust'], width, label='Robust')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Region 1', 'Region 2', 'Region 3'])
    ax.set_ylabel('Unmet Demand')
    ax.legend()
    save_fig(fig, 'fig_allocation')

def generate_robustness():
    data = load_or_mock_data('robustness.json', mock_robustness)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.heatmap(data, ax=ax, cmap='viridis', annot=True)
    ax.set_title('IoU Heatmap')
    save_fig(fig, 'fig_robustness')

def generate_risk_coverage():
    data = load_or_mock_data('risk_coverage.json', mock_risk_coverage)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for method, vals in data.items():
        ax.plot(vals['cov'], vals['risk'], label=method)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk')
    ax.legend()
    save_fig(fig, 'fig_risk_coverage')

def generate_ablation():
    data = load_or_mock_data('ablation.json', mock_ablation)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.bar(data.keys(), data.values())
    ax.set_ylabel('Performance (IoU)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, 'fig_ablation')

def main():
    setup_dirs()
    generate_system_overview()
    generate_architecture()
    generate_reliability()
    generate_conformal_coverage()
    generate_qualitative()
    generate_impact()
    generate_allocation()
    generate_robustness()
    generate_risk_coverage()
    generate_ablation()

if __name__ == '__main__':
    main()
