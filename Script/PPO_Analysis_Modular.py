"""
Enhanced PPO Learning Rate Analysis - Modular Version
====================================================

Complete end-to-end analysis pipeline for comparing PPO learning rate schedules
in HVAC control using Bayesian mixed-effects models.

Author: Amr Shebl
Date: 08 AUG 2025
"""

# ═══════════════════════════════════════════════════════════════════
# 1. IMPORTS AND SETUP
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az
import pathlib
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pathlib

# ═══════════════════════════════════════════════════════════════════
# 2. CONFIGURATION AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════

class AnalysisConfig:
    """Configuration settings for the analysis pipeline."""
    
    # File paths
    BASE = pathlib.Path(__file__).resolve().parents[1]
    RAW_DATA = BASE / "PPO_Analysis_Eval/AllEpisodes_raw.csv"
    OUT_DATA = BASE / "data"
    OUT_FIG = BASE / "figures"
    OUT_RES = BASE / "results"
    PARETO_CSV = BASE / "PPO_Analysis_Eval/grid_tidy_cells_reward_logcv.csv"
    # Learning rate mappings for visualization
    LR_MAPPING = {
        'LR00005': 'Fixed Moderate\n(η = 0.0005)',
        'LR0001': 'Fixed High\n(η = 0.001)', 
        'LateDecayLR': 'Adaptive Decay\n(η = 0.001 → 0.00001)'
    }
    
    LR_MAPPING_SHORT = {
        'LR00005': 'Moderate',
        'LR0001': 'High',
        'LateDecayLR': 'Decay'
    }
    
    # Color schemes
    CLIMATE_COLORS = {
        'Cool': '#4472C4',    # Professional blue
        'Hot': '#E15759',     # Warm red  
        'Mix': '#70AD47'      # Balanced green
    }
    
    LR_COLORS = ['#2E8B57', '#CD853F', '#4682B4']  # For win rates, etc.
    
    # Model parameters
    MCMC_PARAMS = {
        'draws': 3000,
        'chains': 4,
        'target_accept': 0.995,
        'max_treedepth': 15,
        'init': 'adapt_diag',
        'random_seed': 42
    }
    
    # Data columns
    RAW_COLUMNS = {
        "mean_reward": "reward",
        "mean_power_demand": "power",
        "comfort_violation_time(%)": "comfort",
        "std_reward": "reward_std"
    }
    
    ID_COLUMNS = ["PolicySeed", "EvalSeed", "LearningRate", "Climate"]

# ═══════════════════════════════════════════════════════════════════
# 3. DATA PROCESSING MODULE
# ═══════════════════════════════════════════════════════════════════

class DataProcessor:
    """Handles all data loading, processing, and transformation."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load raw episode data and process into analysis-ready format.
        
        Returns:
            pd.DataFrame: Processed episode means (90 rows)
        """
        print("1) Loading and processing raw data...")
        
        # Load raw data
        df = pd.read_csv(self.config.RAW_DATA)
        df[self.config.ID_COLUMNS] = df[self.config.ID_COLUMNS].astype("category")
        
        # Aggregate to episode means
        keep_cols = list(self.config.RAW_COLUMNS.keys())
        epi = (df.groupby(self.config.ID_COLUMNS, observed=True)[keep_cols]
               .agg(['mean', 'std'])
               .rename_axis(index=self.config.ID_COLUMNS, columns=['', '']))
        
        # Rename columns
        epi.columns = [f"{self.config.RAW_COLUMNS[b]}_{stat}" for b, stat in epi.columns]
        epi = epi.reset_index()
        
        # Create derived metrics
        epi = self._create_derived_metrics(epi)
        
        # Z-score standardization
        epi = self._standardize_targets(epi)
        
        # Diagnostics
        self._print_data_diagnostics(epi)
        
        return epi
    
    def _create_derived_metrics(self, epi: pd.DataFrame) -> pd.DataFrame:
        """Create derived metrics for analysis."""
        print("   - Creating derived metrics...")
        
        # Comfort proportion (for Beta modeling)
        eps = 1e-4
        epi["comfort_prop"] = epi["comfort_mean"] / 100.0
        epi["comfort_prop"] = epi["comfort_prop"].clip(eps, 1 - eps)
        
        # Reward coefficient of variation
        epi["reward_cv"] = epi["reward_std"] / epi["reward_mean"].abs()
        epi["log_reward_cv"] = np.log10(epi["reward_cv"])
        
        # Log-transform power demand
        epi["log_power_mean"] = np.log1p(epi["power_mean"])
        
        return epi
    
    def _standardize_targets(self, epi: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardize modeling targets for NUTS stability."""
        print("   - Z-score standardizing targets...")
        
        targets = ["reward_mean", "log_power_mean", "log_reward_cv"]
        for col in targets:
            epi[col] = (epi[col] - epi[col].mean()) / epi[col].std()
        
        return epi
    
    def _print_data_diagnostics(self, epi: pd.DataFrame) -> None:
        """Print data structure diagnostics."""
        print("\n=== DATA STRUCTURE DIAGNOSTICS ===")
        print(f"Final dataset shape: {epi.shape}")
        print(f"Learning rates: {epi['LearningRate'].unique().tolist()}")
        print(f"Climates: {epi['Climate'].unique().tolist()}")
        print(f"Policy seeds: {sorted(epi['PolicySeed'].unique().tolist())}")
        print(f"Eval seeds: {sorted(epi['EvalSeed'].unique().tolist())}")

# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZATION MODULE
# ═══════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """Handles all visualization creation with consistent styling."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        sns.set_theme(style="whitegrid")
        
    def apply_lr_mapping(self, data: pd.DataFrame, short: bool = False) -> pd.DataFrame:
        """Apply consistent learning rate labeling."""
        data_viz = data.copy()
        mapping = self.config.LR_MAPPING_SHORT if short else self.config.LR_MAPPING
        data_viz['LearningRate'] = data_viz['LearningRate'].map(mapping)
        return data_viz
    
    def create_enhanced_descriptives(self, epi: pd.DataFrame) -> None:
        """
        Create four separate descriptive analysis plots .
        
        Each plot is saved as a separate figure with its own legend and optimal spacing.
        
        """
        print("2) Creating enhanced descriptive visualizations ")
    
        epi_viz = self.apply_lr_mapping(epi)
    
    # Define plot specifications
        plot_specs = [
            {
                'y_var': 'reward_mean',
                'title': 'Mean Reward Performance',
                'subtitle': 'Higher values indicate better performance (±SD)',
                'ylabel': 'Z-scored Reward',
                'filename': 'reward_performance'
            },
            {
                'y_var': 'log_power_mean', 
                'title': 'Energy Consumption',
                'subtitle': 'Lower values indicate more efficient energy use (±SD)',
                'ylabel': 'Log Power Demand (Z-scored)',
                'filename': 'energy_consumption'
            },
            {
                'y_var': 'reward_cv',
                'title': 'Performance Stability',
                'subtitle': 'Lower values indicate more consistent performance (±SD)', 
                'ylabel': 'Reward Coefficient of Variation',
                'filename': 'performance_stability'
            },
            {
                'y_var': 'comfort_mean',
                'title': 'Comfort Violations',
                'subtitle': 'Lower values indicate better thermal comfort (±SD)',
                'ylabel': 'Comfort Violation Time (%)',
                'filename': 'comfort_violations'
            }
        ]
        
        # Create each plot as a separate figure
        for i, spec in enumerate(plot_specs):
            # Create individual figure with optimal size
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            
            # Create the bar plot
            sns.barplot(data=epi_viz, x="LearningRate", y=spec['y_var'],
                    hue="Climate", errorbar="sd", ax=ax,
                    palette=self.config.CLIMATE_COLORS)
            
            # Enhanced styling
            ax.set_title(spec['title'], fontsize=16, fontweight='bold', pad=20)
            ax.text(0.5, 0.98, spec['subtitle'], transform=ax.transAxes, 
                    ha='center', va='top', fontsize=11, style='italic',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            ax.set_xlabel("Learning Rate Schedule", fontweight='bold', fontsize=12)
            ax.set_ylabel(spec['ylabel'], fontweight='bold', fontsize=12)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            
            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Position legend optimally for individual plot
            ax.legend(title='Climate Condition', 
                    loc='upper right', 
                    frameon=True, fancybox=True, shadow=True,
                    fontsize=10, title_fontsize=11)
            
            # Adjust layout for clean appearance
            plt.tight_layout()
            
            # Save individual plot
            filename = f"descriptive_{spec['filename']}.png"
            plt.savefig(self.config.OUT_FIG/filename, 
                    dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"   ✓ Created: {filename}")
        
        print("   ✓ All 4 descriptive plots created successfully!")

    
    def create_marginal_means_plot(self, epi: pd.DataFrame) -> None:
        """
        Create marginal means visualization with proper error bars.
        
        Left panel: Overall LR performance with SEM error bars
        Right panel: Climate-specific performance patterns
        """
        print("   - Creating marginal means visualization...")
        
        epi_viz = self.apply_lr_mapping(epi)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall LR ranking with SEM
        lr_means = epi_viz.groupby('LearningRate', observed=True)['reward_mean'].agg(['mean', 'std', 'count'])
        
        ax[0].errorbar(range(len(lr_means)), lr_means['mean'],
                      yerr=lr_means['std']/np.sqrt(lr_means['count']),
                      marker='o', capsize=8, linewidth=3, markersize=12,
                      color='darkblue', markerfacecolor='lightblue', 
                      markeredgewidth=2, markeredgecolor='darkblue')
        ax[0].set_xticks(range(len(lr_means)))
        ax[0].set_xticklabels(lr_means.index, fontweight='bold')
        ax[0].set_title("Overall Learning Rate Performance", fontsize=15, fontweight='bold')
        ax[0].set_ylabel("Z-scored Reward (±SEM)", fontweight='bold')
        ax[0].grid(True, alpha=0.3)
        
        # Climate-specific performance
        climate_colors_list = [self.config.CLIMATE_COLORS[c] for c in ['Cool', 'Hot', 'Mix']]
        sns.pointplot(data=epi_viz, x='LearningRate', y='reward_mean',
                     hue='Climate', ax=ax[1],
                     markers=['o', 's', '^'],
                     linestyles=['-', '--', '-.'],
                     palette=climate_colors_list,
                     markersize=10, linewidth=3)
        ax[1].set_title("Performance by Climate Condition", fontsize=15, fontweight='bold')
        ax[1].set_ylabel("Z-scored Reward", fontweight='bold')
        ax[1].set_xlabel("Learning Rate Schedule", fontweight='bold')
        ax[1].tick_params(axis='x', rotation=0)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(title='Climate', loc='lower right', frameon=True, 
                    fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"marginal_means_enhanced.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_climate_results_table(self, epi: pd.DataFrame) -> pd.DataFrame:
        """
        Extract climate-specific results directly from processed data.
        """
        print(" - Creating climate-specific results table from processed data...")
        
        # Apply learning rate mapping for clean labels
        epi_clean = self.apply_lr_mapping(epi)
        
        # Extract mean values by Learning Rate and Climate
        results_list = []
        
        for lr in epi_clean['LearningRate'].unique():
            lr_data = epi_clean[epi_clean['LearningRate'] == lr]
            
            row = {'Learning Rate Schedule': lr}
            
            # Extract values for each climate condition
            for climate in ['Cool', 'Hot', 'Mix']:
                climate_data = lr_data[lr_data['Climate'] == climate]
                
                if len(climate_data) > 0:
                    # Mean reward performance (Z-scored)
                    row[f'{climate}_Reward'] = climate_data['reward_mean'].mean()
                    
                    # Energy consumption (Z-scored log power)
                    row[f'{climate}_Energy'] = climate_data['log_power_mean'].mean()
                    
                    # Performance stability (coefficient of variation)
                    row[f'{climate}_Stability'] = climate_data['reward_cv'].mean()
                    
                    # Comfort violations (percentage)
                    row[f'{climate}_Comfort'] = climate_data['comfort_mean'].mean()
                else:
                    # Handle missing data
                    for metric in ['Reward', 'Energy', 'Stability', 'Comfort']:
                        row[f'{climate}_{metric}'] = np.nan
            
            results_list.append(row)
        
        # Create DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Save raw data table
        df_results.to_csv(self.config.OUT_DATA/"climate_specific_results_extracted.csv", index=False)
        
        # Create formatted display table
        print("\n=== EXTRACTED CLIMATE-SPECIFIC RESULTS ===")
        
        # Display formatted table
        for _, row in df_results.iterrows():
            lr_name = row['Learning Rate Schedule']
            print(f"\n{lr_name}:")
            print(f"  Reward Performance: Cool={row['Cool_Reward']:.2f}, Hot={row['Hot_Reward']:.2f}, Mix={row['Mix_Reward']:.2f}")
            print(f"  Energy Consumption: Cool={row['Cool_Energy']:.2f}, Hot={row['Hot_Energy']:.2f}, Mix={row['Mix_Energy']:.2f}")
            print(f"  Stability (CV):     Cool={row['Cool_Stability']:.3f}, Hot={row['Hot_Stability']:.3f}, Mix={row['Mix_Stability']:.3f}")
            print(f"  Comfort Violations: Cool={row['Cool_Comfort']:.1f}%, Hot={row['Hot_Comfort']:.1f}%, Mix={row['Mix_Comfort']:.1f}%")
        
        # Create markdown table
        self._create_markdown_table(df_results)
        
        print(f" ✓ Extracted results saved to: {self.config.OUT_DATA/'climate_specific_results_extracted.csv'}")
        
        return df_results

    def _create_markdown_table(self, df_results: pd.DataFrame) -> None:
        """Create markdown table """
        
        with open(self.config.OUT_DATA/"climate_table.md", 'w') as f:
            f.write("| Learning Rate Schedule | **Mean Reward (Z-score)** | **Energy Consumption (Z-score)** | **Stability (CV)** | **Comfort Violations (%)** |\n")
            f.write("|------------------------|---------------------------|----------------------------------|--------------------|--------------------------|\n")
            f.write("| | Cool \\| Hot \\| Mix | Cool \\| Hot \\| Mix | Cool \\| Hot \\| Mix | Cool \\| Hot \\| Mix |\n")
            
            for _, row in df_results.iterrows():
                lr_name = row['Learning Rate Schedule']
                f.write(f"| **{lr_name}** | ")
                f.write(f"{row['Cool_Reward']:+.2f} \\| {row['Hot_Reward']:+.2f} \\| {row['Mix_Reward']:+.2f} | ")
                f.write(f"{row['Cool_Energy']:+.2f} \\| {row['Hot_Energy']:+.2f} \\| {row['Mix_Energy']:+.2f} | ")
                f.write(f"{row['Cool_Stability']:.3f} \\| {row['Hot_Stability']:.3f} \\| {row['Mix_Stability']:.3f} | ")
                f.write(f"{row['Cool_Comfort']:.1f} \\| {row['Hot_Comfort']:.1f} \\| {row['Mix_Comfort']:.1f} |\n")
        
        print(f" ✓ Markdown table saved to: {self.config.OUT_DATA/'climate_table.md'}")        

    def create_win_rates_analysis(self, epi: pd.DataFrame) -> pd.DataFrame:
        print("   - Creating CORRECTED win rates analysis...")
        
        # Aggregate across EvalSeeds first
        lr_performance = epi.groupby(['PolicySeed', 'LearningRate', 'Climate'], 
                                    observed=True)['reward_mean'].mean().reset_index()
        lr_performance.columns = ['PolicySeed', 'LearningRate', 'Climate', 'avg_reward']
        
        # Find winners for each seed-climate combination
        winners_list = []
        for (policy_seed, climate), group in lr_performance.groupby(['PolicySeed', 'Climate']):
            winner_idx = group['avg_reward'].idxmax()
            winner_lr = group.loc[winner_idx, 'LearningRate']
            winners_list.append({
                'PolicySeed': policy_seed,
                'Climate': climate,
                'Winner': winner_lr
            })
        
        winners = pd.DataFrame(winners_list)
        
        # Apply labeling
        winners_viz = winners.copy()
        winners_viz['Winner'] = winners_viz['Winner'].map(self.config.LR_MAPPING_SHORT)
        
        win_counts = winners_viz['Winner'].value_counts()
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left panel: Overall wins
        bars = win_counts.plot(kind='bar', ax=ax[0], color=self.config.LR_COLORS[:len(win_counts)])
        ax[0].set_title("Learning Rate 'Wins' Across All Conditions", fontsize=14, fontweight='bold')
        ax[0].set_ylabel("Number of Wins (out of 15)", fontweight='bold')
        ax[0].set_xlabel("Learning Rate Schedule", fontweight='bold')
        ax[0].tick_params(axis='x', rotation=45)
        ax[0].grid(True, alpha=0.3)
        
        total = len(winners)
        for i, v in enumerate(win_counts):
            ax[0].text(i, v + 0.2, f'{v/total*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
        
        # Right panel: Win rate by climate - FIXED
        win_by_climate = pd.crosstab(winners_viz['Climate'], winners_viz['Winner'])
        
        # Add missing learning rates with zeros
        all_lrs = ['Moderate', 'High', 'Decay']
        for lr in all_lrs:
            if lr not in win_by_climate.columns:
                win_by_climate[lr] = 0
        
        # Reorder columns
        win_by_climate = win_by_climate.reindex(columns=all_lrs, fill_value=0)
        
        # Create plot with all LRs
        win_by_climate.plot(kind='bar', ax=ax[1], color=self.config.LR_COLORS)
        ax[1].set_title("Win Rate by Climate", fontsize=14, fontweight='bold')
        ax[1].set_ylabel("Number of Wins (out of 5 per climate)", fontweight='bold')
        ax[1].set_xlabel("Climate Condition", fontweight='bold')
        ax[1].tick_params(axis='x', rotation=0)
        ax[1].legend(title="Learning Rate", loc='upper right')
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"win_rates_analysis_corrected.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return winners

    def create_win_rates_heatmap(self, epi: pd.DataFrame) -> pd.DataFrame:
        """Create heatmap visualization of win rates."""
        # Aggregate across EvalSeeds first
        lr_performance = epi.groupby(['PolicySeed', 'LearningRate', 'Climate'], 
                                    observed=True)['reward_mean'].mean().reset_index()
        lr_performance.columns = ['PolicySeed', 'LearningRate', 'Climate', 'avg_reward']
        # Find winners
        winners_list = []
        for (policy_seed, climate), group in lr_performance.groupby(['PolicySeed', 'Climate']):
            winner_lr = group.loc[group['avg_reward'].idxmax(), 'LearningRate']
            winners_list.append({
                'PolicySeed': policy_seed,
                'Climate': climate,
                'Winner': winner_lr
            })
        
        winners = pd.DataFrame(winners_list)
        winners['Winner'] = winners['Winner'].map(self.config.LR_MAPPING_SHORT)
        
        # Create pivot table for heatmap
        heatmap_data = pd.crosstab(winners['Climate'], winners['PolicySeed'], 
                                values=winners['Winner'], aggfunc='first')
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Climate vs PolicySeed heatmap
        # Create numeric mapping for colors
        lr_to_num = {'Moderate': 2, 'Decay': 1, 'High': 0}
        heatmap_numeric = heatmap_data.applymap(lambda x: lr_to_num.get(x, 0))
        
        sns.heatmap(heatmap_numeric, annot=heatmap_data, fmt='', 
                    cmap='RdYlGn', ax=ax[0], cbar_kws={'label': 'Learning Rate Performance'})
        ax[0].set_title("Win Pattern: Climate × Policy Seed", fontsize=14, fontweight='bold')
        ax[0].set_xlabel("Policy Seed")
        ax[0].set_ylabel("Climate")
        
        # Right: Summary pie chart
        win_counts = winners['Winner'].value_counts()
        colors = [self.config.LR_COLORS[i] for i in range(len(win_counts))]
        ax[1].pie(win_counts.values, labels=win_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax[1].set_title("Overall Win Distribution", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"win_rates_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return winners
    def create_win_rates_stacked(self, epi: pd.DataFrame) -> pd.DataFrame:
        """Create stacked area/bar chart for win rates."""
        lr_performance = epi.groupby(['PolicySeed', 'LearningRate', 'Climate'], 
                                    observed=True)['reward_mean'].mean().reset_index()
        lr_performance.columns = ['PolicySeed', 'LearningRate', 'Climate', 'avg_reward']
        
        winners_list = []
        for (policy_seed, climate), group in lr_performance.groupby(['PolicySeed', 'Climate']):
            winner_lr = group.loc[group['avg_reward'].idxmax(), 'LearningRate']
            winners_list.append({
                'PolicySeed': policy_seed,
                'Climate': climate,
                'Winner': winner_lr
            })
        
        winners = pd.DataFrame(winners_list)
        winners['Winner'] = winners['Winner'].map(self.config.LR_MAPPING_SHORT)
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Stacked bar by climate
        win_by_climate = pd.crosstab(winners['Climate'], winners['Winner'])
        
        # Ensure all LRs appear
        all_lrs = ['Moderate', 'High', 'Decay']
        for lr in all_lrs:
            if lr not in win_by_climate.columns:
                win_by_climate[lr] = 0
        win_by_climate = win_by_climate.reindex(columns=all_lrs, fill_value=0)
        
        # Create stacked bar as percentages
        win_percentages = win_by_climate.div(win_by_climate.sum(axis=1), axis=0) * 100
        
        win_percentages.plot(kind='bar', stacked=True, ax=ax[0], 
                            color=self.config.LR_COLORS, width=0.6)
        ax[0].set_title("Win Rate Distribution by Climate", fontsize=14, fontweight='bold')
        ax[0].set_ylabel("Percentage of Wins")
        ax[0].set_xlabel("Climate")
        ax[0].tick_params(axis='x', rotation=0)
        ax[0].legend(title="Learning Rate")
        
        # Right: Donut chart
        win_counts = winners['Winner'].value_counts()
        wedges, texts, autotexts = ax[1].pie(win_counts.values, labels=win_counts.index, 
                                            autopct='%1.1f%%', colors=self.config.LR_COLORS,
                                            wedgeprops=dict(width=0.5))
        ax[1].set_title("Overall Dominance", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"win_rates_stacked.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return winners
   
 
    def create_forest_plots(self, idata_dict: Dict[str, Any]) -> None:
        """
        Create forest plots comparing learning rate effects across all outcomes.
        
        NOW WITH PROPER LEARNING RATE LABELS!
        """
        print("   - Creating forest plots with enhanced LR labels...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        outcomes = ['reward', 'energy', 'comfort', 'cv']
        titles = ['Reward', 'Energy Consumption', 
                 'Comfort Violations', 'Stability (CV)']
        
        for i, (outcome, idata) in enumerate(idata_dict.items()):
            try:
                # Create forest plot
                az.plot_forest(idata, var_names=["LearningRate"], 
                              combined=True, ax=axes[i])
                
                # ENHANCEMENT: Update y-axis labels with proper LR names
                current_labels = axes[i].get_yticklabels()
                new_labels = []
                for label in current_labels:
                    label_text = label.get_text()
                    if 'LR0001' in label_text:
                        new_labels.append('Fixed High (η=0.001)')
                    elif 'LateDecayLR' in label_text:
                        new_labels.append('Adaptive Decay')
                    else:
                        new_labels.append(label_text)
                axes[i].set_yticklabels(new_labels)
                
                axes[i].set_title(f"{titles[i]}: Effects vs Fixed Moderate", 
                                 fontsize=14, fontweight='bold')
                axes[i].axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlabel("Effect Size (σ units)", fontweight='bold')
                
            except Exception as e:
                print(f"   Warning: Could not create forest plot for {outcome}: {e}")
                axes[i].text(0.5, 0.5, f"Forest plot unavailable\nfor {titles[i]}", 
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                 facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"forest_plots_enhanced.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_signal_vs_noise_analysis(self, tr_reward: Any) -> None:
        """
        Create signal vs noise analysis with proper learning rate labels.
        
        NOW WITH PROPER LEARNING RATE LABELS!
        """
        print("   - Creating signal vs noise analysis with enhanced labels...")
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract posterior samples
        lr_effects = tr_reward.posterior["LearningRate"]
        
        try:
            # Try coordinate-based selection first
            lr0001_effect = lr_effects.sel(LearningRate_dim="LR0001").values.flatten()
            latedecay_effect = lr_effects.sel(LearningRate_dim="LateDecayLR").values.flatten()
            print("   Using coordinate-based selection for posterior samples")
        except:
            # Fallback to index-based selection
            print("   Using index-based selection - verifying order...")
            print(f"   Available coordinates: {lr_effects.coords}")
            lr0001_effect = lr_effects.isel(LearningRate_dim=0).values.flatten()
            latedecay_effect = lr_effects.isel(LearningRate_dim=1).values.flatten()
        
        seed_noise = tr_reward.posterior["1|PolicySeed_sigma"].values.flatten()
        
        # Left panel: Effect magnitude distributions
        ax[0].hist(np.abs(lr0001_effect), alpha=0.7, 
                  label="Fixed High (η=0.001)", density=True, bins=35, 
                  color='#E15759')
        ax[0].hist(np.abs(latedecay_effect), alpha=0.7, 
                  label="Adaptive Decay", density=True, bins=35, 
                  color='#70AD47')
        ax[0].hist(seed_noise, alpha=0.7, 
                  label="Seed Noise (σ)", density=True, bins=35, 
                  color='#4472C4')
        ax[0].axvline(0, color='red', linestyle='--', alpha=0.6, linewidth=2)
        ax[0].set_xlabel("Effect Magnitude (σ units)", fontweight='bold', fontsize=12)
        ax[0].set_ylabel("Density", fontweight='bold', fontsize=12)
        ax[0].set_title("Signal vs. Noise Comparison", fontsize=14, fontweight='bold')
        ax[0].legend(frameon=True, fancybox=True, shadow=True)
        ax[0].grid(True, alpha=0.3)
        
        # Right panel: Probability of exceeding noise
        prob_lr0001_exceeds = np.mean(np.abs(lr0001_effect) > seed_noise)
        prob_latedecay_exceeds = np.mean(np.abs(latedecay_effect) > seed_noise)
        
        bars = ax[1].bar(['Fixed High\nvs Seed Noise', 'Adaptive Decay\nvs Seed Noise'], 
                        [prob_lr0001_exceeds, prob_latedecay_exceeds],
                        color=['#E15759', '#70AD47'], alpha=0.8, width=0.6)
        ax[1].set_ylabel("Posterior Probability", fontweight='bold', fontsize=12)
        ax[1].set_title("Probability: LR Effect > Seed Variability", 
                       fontsize=14, fontweight='bold')
        ax[1].set_ylim([0, 1])
        ax[1].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, [prob_lr0001_exceeds, prob_latedecay_exceeds])):
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                      f'{prob:.1%}', ha='center', va='bottom', 
                      fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"signal_vs_noise_enhanced.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ═══════════════════════════════════════════════════════════════════
# 5. BAYESIAN MODELING MODULE
# ═══════════════════════════════════════════════════════════════════

class BayesianAnalyzer:
    """Handles Bayesian mixed-effects model fitting and diagnostics."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def fit_model(self, target: str, family: str, data: pd.DataFrame) -> Any:
        """
        Fit a Bayesian mixed-effects model for a specific target variable.
        
        Args:
            target: Target variable name
            family: Distribution family ('gaussian', 'beta', etc.)
            data: Input dataframe
            
        Returns:
            Fitted InferenceData object
        """
        print(f"   - Fitting {family} model for {target}...")
        
        # Model specification
        formula = (f"{target} ~ LearningRate * Climate "
                  "+ (1|PolicySeed) + (1|Climate:EvalSeed)")
        
        # Create and fit model
        model = bmb.Model(formula, data=data, family=family,
                         priors={"Intercept": bmb.Prior("Normal", mu=0, sigma=1)})
        
        idata = model.fit(**self.config.MCMC_PARAMS)
        
        # Save trace plot
        az.plot_trace(idata)
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/f"trace_{target}.png", dpi=300)
        plt.close()
        
        # Generate and save summary
        summary = az.summary(idata)
        self._print_diagnostics(target, summary)
        summary.to_csv(self.config.OUT_RES/f"summary_{target}.csv")
        
        return idata
    
    def fit_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit all four outcome models with FIXED key assignment.
        
        Returns:
            Dictionary of fitted models with correct keys
        """
        print("3) Fitting Bayesian mixed-effects models...")
        
        models = {}
        
        # Model specifications with EXPLICIT key mapping
        model_specs = [
            ("reward_mean", "gaussian", "Primary performance outcome", "reward"),
            ("log_power_mean", "gaussian", "Energy consumption", "energy"),  
            ("comfort_prop", "beta", "Comfort violations", "comfort"),
            ("log_reward_cv", "gaussian", "Performance stability", "cv")
        ]
        
        for target, family, description, model_key in model_specs:
            print(f"   {description}:")
            try:
                models[model_key] = self.fit_model(target, family, data)
                print(f"     ✓ Successfully fitted '{model_key}' model")
            except Exception as e:
                print(f"     ✗ ERROR fitting {target}: {e}")
                
        print(f"\n   Successfully fitted {len(models)}/4 models: {list(models.keys())}")
        return models
    
    def _print_diagnostics(self, target: str, summary: pd.DataFrame) -> None:
        """Print model convergence diagnostics."""
        print(f"     ✓ Max R-hat: {summary['r_hat'].max():.4f} (should be < 1.01)")
        print(f"     ✓ Min ESS bulk: {summary['ess_bulk'].min():.0f} (should be > 400)")
        print(f"     ✓ Min ESS tail: {summary['ess_tail'].min():.0f} (should be > 400)")

# ═══════════════════════════════════════════════════════════════════
# 6. POWER ANALYSIS MODULE (CORRECTED - POLICY SEED LEVEL)
# ═══════════════════════════════════════════════════════════════════

class PowerAnalysis:
    """Handles bootstrap power analysis at the policy seed level for learning rate comparisons."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def aggregate_to_policy_seed_level(self, epi: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate evaluation runs to policy seed level means.
        This is the correct unit of analysis for learning rate comparisons.
        
        Args:
            epi: Full evaluation dataset (90 rows)
            
        Returns:
            DataFrame with one row per policy seed × learning rate (15 rows)
        """
        # Aggregate across climates and eval seeds to get policy seed performance
        seed_level = (epi.groupby(['PolicySeed', 'LearningRate'], observed=True)
                      ['reward_mean'].mean().reset_index())
        seed_level.columns = ['PolicySeed', 'LearningRate', 'reward_mean_agg']
        
        print(f" - Aggregated from {len(epi)} evaluations to {len(seed_level)} policy seed means")
        print(f" - Sample size per learning rate: {len(seed_level) // len(epi['LearningRate'].unique())}")
        
        return seed_level
    
    def bootstrap_power_test_seeds(self, group1_data: np.ndarray, group2_data: np.ndarray, 
                                  effect_size: float, n_simulations: int = 1000, alpha: float = 0.05) -> float:
        """
        Bootstrap power analysis using policy seed level data (n=5 per group).
        
        Args:
            group1_data: Policy seed means for learning rate 1 (n=5)
            group2_data: Policy seed means for learning rate 2 (n=5)  
            effect_size: Hypothetical effect size to detect
            n_simulations: Number of bootstrap simulations
            alpha: Significance level
            
        Returns:
            Power estimate
        """
        n1, n2 = len(group1_data), len(group2_data)
        detections = 0
        
        for _ in range(n_simulations):
            # Apply effect size to group1 and bootstrap resample
            lifted_group1 = group1_data + effect_size
            bs1 = np.random.choice(lifted_group1, size=n1, replace=True)
            bs2 = np.random.choice(group2_data, size=n2, replace=True)
            
            # Bootstrap confidence interval for difference
            boot_diffs = []
            for _ in range(200):  # Inner bootstrap
                b1 = np.random.choice(bs1, size=n1, replace=True)
                b2 = np.random.choice(bs2, size=n2, replace=True)
                boot_diffs.append(np.mean(b1) - np.mean(b2))
            
            # Check if CI excludes zero
            ci_lower = np.percentile(boot_diffs, 100 * alpha/2)
            ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha/2))
            
            if ci_lower > 0 or ci_upper < 0:
                detections += 1
        
        return detections / n_simulations
    
    def run_policy_seed_power_analysis(self, epi: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive power analysis at policy seed level.
        
        Args:
            epi: Full evaluation dataset
            
        Returns:
            Power analysis results
        """
        print("5) Running CORRECTED bootstrap power analysis (policy seed level)...")
        
        # Aggregate to policy seed level
        seed_data = self.aggregate_to_policy_seed_level(epi)
        
        # Effect sizes to test
        effect_sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]  # 20% to 80% of SD
        
        # Learning rate pairs
        learning_rates = seed_data['LearningRate'].unique()
        lr_pairs = [(lr1, lr2) for i, lr1 in enumerate(learning_rates) 
                    for lr2 in learning_rates[i+1:]]
        
        results = {
            'effect_sizes': effect_sizes,
            'lr_pairs': [],
            'power_matrix': [],
            'summary': {},
            'n_per_group': len(seed_data) // len(learning_rates)
        }
        
        print(f" - Testing power with n={results['n_per_group']} policy seeds per learning rate...")
        
        for lr1, lr2 in lr_pairs:
            # Get short names for readability
            lr1_short = self.config.LR_MAPPING_SHORT[lr1]
            lr2_short = self.config.LR_MAPPING_SHORT[lr2]
            pair_name = f"{lr1_short} vs {lr2_short}"
            results['lr_pairs'].append(pair_name)
            
            # Extract policy seed level data
            group1_data = seed_data[seed_data['LearningRate'] == lr1]['reward_mean_agg'].values
            group2_data = seed_data[seed_data['LearningRate'] == lr2]['reward_mean_agg'].values
            
            # Calculate power for each effect size
            power_row = []
            for effect_size in effect_sizes:
                power = self.bootstrap_power_test_seeds(group1_data, group2_data, effect_size)
                power_row.append(power)
            
            results['power_matrix'].append(power_row)
            
            # Summary for key effect sizes
            power_30pct = power_row[1]  # 30% effect
            power_40pct = power_row[2]  # 40% effect
            power_50pct = power_row[3]  # 50% effect
            
            results['summary'][pair_name] = {
                '30pct_effect': power_30pct,
                '40pct_effect': power_40pct,
                '50pct_effect': power_50pct
            }
            
            print(f"   ✓ {pair_name}: Power = {power_30pct:.3f} (30%), {power_40pct:.3f} (40%), {power_50pct:.3f} (50%)")
        
        # Create visualizations
        self._create_policy_seed_power_plots(results)
        
        # Print summary with proper interpretation
        self._print_policy_seed_power_summary(results)
        
        return results
    
    def _create_policy_seed_power_plots(self, results: Dict[str, Any]) -> None:
        """Create power analysis visualizations for policy seed level analysis."""
        print(" - Creating policy seed level power visualizations...")
        
        power_matrix = np.array(results['power_matrix'])
        effect_sizes = results['effect_sizes']
        lr_pairs = results['lr_pairs']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: Power curves
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, pair in enumerate(lr_pairs):
            ax1.plot(effect_sizes, power_matrix[i], 'o-', 
                    color=colors[i % len(colors)], linewidth=3, markersize=8,
                    label=pair)
        
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label='80% Power Threshold')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, linewidth=1,
                   label='50% Power')
        
        ax1.set_xlabel('Effect Size (σ units)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Statistical Power', fontweight='bold', fontsize=12)
        ax1.set_title(f'Power Analysis: Policy Seed Level (n={results["n_per_group"]} per LR)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 1)
        
        # Right panel: Power at 40% effect size
        power_40pct = power_matrix[:, 2]  # 40% effect column
        
        bars = ax2.bar(range(len(lr_pairs)), power_40pct, 
                      color=['#2E8B57', '#CD853F', '#4682B4'][:len(lr_pairs)],
                      alpha=0.8)
        
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label='80% Power')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.6, linewidth=1,
                   label='50% Power')
        
        ax2.set_xticks(range(len(lr_pairs)))
        ax2.set_xticklabels(lr_pairs, rotation=45)
        ax2.set_ylabel('Statistical Power', fontweight='bold', fontsize=12)
        ax2.set_title('Power at 40% Effect Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # Add power values on bars
        for bar, power in zip(bars, power_40pct):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{power:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.config.OUT_FIG/"power_analysis_policy_seed_corrected.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_policy_seed_power_summary(self, results: Dict[str, Any]) -> None:
        """Print corrected power analysis summary."""
        print(f"\n=== CORRECTED POWER ANALYSIS SUMMARY (POLICY SEED LEVEL) ===")
        print(f"Unit of analysis: Policy seeds (n={results['n_per_group']} per learning rate)")
        print(f"Bootstrap simulations: 1000 per comparison")
        print(f"Research question: 'Are 5 policy seeds sufficient to detect LR differences?'")
        
        print(f"\nPower to detect moderate to large effects:")
        
        for pair, powers in results['summary'].items():
            power_30 = powers['30pct_effect']
            power_40 = powers['40pct_effect'] 
            power_50 = powers['50pct_effect']
            
            # Interpret power levels
            status_40 = "✓ Adequate" if power_40 >= 0.8 else "⚠ Moderate" if power_40 >= 0.5 else "✗ Low"
            status_50 = "✓ Adequate" if power_50 >= 0.8 else "⚠ Moderate" if power_50 >= 0.5 else "✗ Low"
            
            print(f"  {pair}:")
            print(f"    30% effect: {power_30:.3f}")
            print(f"    40% effect: {power_40:.3f} ({status_40})")
            print(f"    50% effect: {power_50:.3f} ({status_50})")
        
        # Overall assessment
        avg_power_40 = np.mean([powers['40pct_effect'] for powers in results['summary'].values()])
        avg_power_50 = np.mean([powers['50pct_effect'] for powers in results['summary'].values()])
        
        print(f"\nOverall averages:")
        print(f"40% effect: {avg_power_40:.3f}")
        print(f"50% effect: {avg_power_50:.3f}")
        
       
        print(f"📊 EXPERIMENTAL REALITY:")
        print(f"   • n=5 policy seeds represents substantial computational investment")
        print(f"   • Each seed required full PPO training (21 episodes × 3 LR conditions)")
        print(f"   • Sample size typical for high-quality DRL studies given computational constraints")
        
        if avg_power_50 >= 0.8:
            print(f"\n✓ CONCLUSION: Adequate power for detecting large effects (50%+ differences)")
        elif avg_power_40 >= 0.5:
            print(f"\n⚠ CONCLUSION: Moderate power - can detect substantial differences with reasonable confidence")
            print(f"   • Bayesian approach compensates by modeling uncertainty explicitly")
            print(f"   • Effects detected despite limited power are likely robust")
        else:
            print(f"\n⚠ CONCLUSION: Limited power for detecting moderate effects")
            print(f"   • Emphasizes value of Bayesian hierarchical modeling")
            print(f"   • Results should be interpreted as exploratory/pilot findings")
        

    def set_visualizer(self, visualizer):
        """Set reference to visualizer for configuration access."""
        self.visualizer = visualizer

# ═══════════════════════════════════════════════════════════════════
# 7. MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════

class PPOAnalysisPipeline:
    """Main analysis pipeline orchestrating all components."""
    
    def __init__(self):
        self.config = AnalysisConfig()
        self._setup_directories()
        
        # Initialize modules
        self.data_processor = DataProcessor(self.config)
        self.visualizer = VisualizationEngine(self.config)
        self.analyzer = BayesianAnalyzer(self.config)
        self.power_analyzer = PowerAnalysis(self.config)
        self.power_analyzer.set_visualizer(self.visualizer)

    def _setup_directories(self) -> None:
        """Create output directories."""
        for path in [self.config.OUT_DATA, self.config.OUT_FIG, self.config.OUT_RES]:
            path.mkdir(exist_ok=True)
    
    def _cleanup_previous_outputs(self) -> None:
        """Clean up any previous output files before running analysis."""
        import glob
        
        # Clean figures
        for file in glob.glob(str(self.config.OUT_FIG / "*.png")):
            os.remove(file)
        
        # Clean results  
        for file in glob.glob(str(self.config.OUT_RES / "*.csv")):
            os.remove(file)
            
        print("   Cleaned previous output files")

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*70)
        print("PPO LEARNING RATE ANALYSIS - ENHANCED MODULAR VERSION")
        print("="*70)
        
        # 1. Data Processing
        epi = self.data_processor.load_and_process_data()
        epi.to_csv(self.config.OUT_DATA / "EpisodeMeans.csv", index=False)
        
         # CLEAN PREVIOUS OUTPUTS
        self._cleanup_previous_outputs()

        # 2. Descriptive Visualizations  
        print("\n2) Creating comprehensive visualizations...")
        self.visualizer.create_enhanced_descriptives(epi)
        self.visualizer.create_marginal_means_plot(epi)
        climate_table = self.visualizer.create_climate_results_table(epi)
        winners = self.visualizer.create_win_rates_analysis(epi)
        
        
        # 3. Bayesian Analysis
        models = self.analyzer.fit_all_models(epi)
        
        # 4. Advanced Visualizations (with proper LR labels!)
        print("\n4) Creating advanced statistical visualizations...")
        self.visualizer.create_forest_plots(models)
        self.visualizer.create_signal_vs_noise_analysis(models['reward'])
        print("   - Creating alternative win rate visualizations...")
        self.visualizer.create_win_rates_heatmap(epi)
        print("\n4) Creating advanced statistical visualizations...")
        self.visualizer.create_forest_plots(models)
        self.visualizer.create_signal_vs_noise_analysis(models['reward'])
        self.visualizer.create_win_rates_heatmap(epi)
       
        




        # 5. Power Analysis
        power_results = self.power_analyzer.run_policy_seed_power_analysis(epi)     

        # 6. Summary
        self._print_completion_summary()
       
        return {
            'data': epi,
            'models': models,
            'winners': winners,
            'power_results': power_results,
            'config': self.config
        }
    
    
    def _print_completion_summary(self) -> None:
        """Print analysis completion summary."""
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE! Generated outputs:")
        print("="*70)
        print(f"📊 Data: {self.config.OUT_DATA}/EpisodeMeans.csv")
        print(f"📈 Figures: {len(list(self.config.OUT_FIG.glob('*.png')))} visualizations in {self.config.OUT_FIG}")
        print(f"📋 Results: {len(list(self.config.OUT_RES.glob('*.csv')))} model summaries in {self.config.OUT_RES}")


# ═══════════════════════════════════════════════════════════════════
# 8. EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Run complete analysis
    pipeline = PPOAnalysisPipeline()
    results = pipeline.run_complete_analysis()
    

    # Optional: Access results for further analysis
    # epi_data = results['data']
    # bayesian_models = results['models'] 
    # win_analysis = results['winners']
