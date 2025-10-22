"""
Motif analysis for DNA sequences in binary classification.

This module provides:
1. Motif discovery and visualization
2. Pattern frequency analysis
3. Class-specific motif identification
4. Motif importance scoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import re
from itertools import product
from scipy.stats import chi2_contingency, fisher_exact
import logomaker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score


class MotifAnalyzer:
    """Comprehensive motif analysis for DNA sequences."""
    
    def __init__(self, sequences: List[str], labels: List[int], motif_lengths: List[int] = [6, 7, 8, 9, 10, 11]):
        """
        Initialize motif analyzer.
        
        Args:
            sequences: List of DNA sequences
            labels: List of binary labels (0 or 1)
            motif_lengths: List of motif lengths to analyze
        """
        self.sequences = sequences
        self.labels = np.array(labels)
        self.motif_lengths = motif_lengths
        
        # Separate sequences by class
        self.class_0_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]
        self.class_1_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]
        
        print(f"Initialized motif analyzer:")
        print(f"  Total sequences: {len(sequences)}")
        print(f"  Class 0: {len(self.class_0_sequences)}")
        print(f"  Class 1: {len(self.class_1_sequences)}")
        print(f"  Motif lengths: {motif_lengths}")
    
    def extract_kmers(self, sequences: List[str], k: int) -> Counter:
        """Extract k-mers from sequences."""
        kmers = Counter()
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if all(base in 'ATGC' for base in kmer):  # Only valid nucleotides
                    kmers[kmer] += 1
        return kmers
    
    def find_discriminative_motifs(self, k: int, min_frequency: int = 5, 
                                 max_motifs: int = 50) -> List[Dict]:
        """Find motifs that discriminate between classes."""
        
        # Extract k-mers for each class
        class_0_kmers = self.extract_kmers(self.class_0_sequences, k)
        class_1_kmers = self.extract_kmers(self.class_1_sequences, k)
        
        # Get all unique k-mers
        all_kmers = set(class_0_kmers.keys()) | set(class_1_kmers.keys())
        
        motif_stats = []
        
        for kmer in all_kmers:
            count_0 = class_0_kmers.get(kmer, 0)
            count_1 = class_1_kmers.get(kmer, 0)
            total_count = count_0 + count_1
            
            if total_count < min_frequency:
                continue
            
            # Calculate frequencies
            freq_0 = count_0 / len(self.class_0_sequences)
            freq_1 = count_1 / len(self.class_1_sequences)
            
            # Statistical test (Fisher's exact test for small counts)
            if count_0 + count_1 >= 10:
                # Chi-square test
                observed = [[count_0, len(self.class_0_sequences) - count_0],
                           [count_1, len(self.class_1_sequences) - count_1]]
                try:
                    chi2, p_value = chi2_contingency(observed)[:2]
                except:
                    p_value = 1.0
            else:
                # Fisher's exact test
                try:
                    _, p_value = fisher_exact([[count_0, len(self.class_0_sequences) - count_0],
                                             [count_1, len(self.class_1_sequences) - count_1]])
                except:
                    p_value = 1.0
            
            # Calculate fold change
            fold_change = (freq_1 + 1e-8) / (freq_0 + 1e-8)
            
            motif_stats.append({
                'motif': kmer,
                'length': k,
                'count_class_0': count_0,
                'count_class_1': count_1,
                'freq_class_0': freq_0,
                'freq_class_1': freq_1,
                'fold_change': fold_change,
                'log2_fold_change': np.log2(fold_change),
                'p_value': p_value,
                'total_count': total_count
            })
        
        # Sort by significance and fold change
        motif_stats.sort(key=lambda x: (x['p_value'], -abs(x['log2_fold_change'])))
        
        return motif_stats[:max_motifs]
    
    def analyze_all_motif_lengths(self, min_frequency: int = 5, 
                                max_motifs_per_length: int = 20) -> Dict[int, List[Dict]]:
        """Analyze motifs for all specified lengths."""
        
        all_motifs = {}
        
        for k in self.motif_lengths:
            print(f"Analyzing {k}-mers...")
            motifs = self.find_discriminative_motifs(k, min_frequency, max_motifs_per_length)
            all_motifs[k] = motifs
            print(f"  Found {len(motifs)} significant {k}-mers")
        
        return all_motifs
    
    def create_position_weight_matrix(self, motifs: List[str]) -> np.ndarray:
        """Create position weight matrix from aligned motifs."""
        if not motifs:
            return np.array([])
        
        motif_length = len(motifs[0])
        nucleotides = ['A', 'T', 'G', 'C']
        pwm = np.zeros((4, motif_length))
        
        for pos in range(motif_length):
            counts = Counter(motif[pos] for motif in motifs if pos < len(motif))
            total = sum(counts.values())
            
            for i, nuc in enumerate(nucleotides):
                pwm[i, pos] = counts.get(nuc, 0) / total if total > 0 else 0.25
        
        return pwm
    
    def plot_motif_logo(self, motifs: List[str], title: str = "Motif Logo"):
        """Create sequence logo for motifs."""
        if not motifs:
            print("No motifs to plot")
            return
        
        # Create PWM
        pwm = self.create_position_weight_matrix(motifs)
        
        if pwm.size == 0:
            print("Could not create PWM")
            return
        
        # Convert to DataFrame for logomaker
        pwm_df = pd.DataFrame(pwm.T, columns=['A', 'T', 'G', 'C'])
        
        # Create logo
        fig, ax = plt.subplots(figsize=(max(8, len(motifs[0]) * 0.8), 4))
        logo = logomaker.Logo(pwm_df, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Position')
        ax.set_ylabel('Bits')
        plt.tight_layout()
        plt.show()
    
    def plot_motif_analysis_summary(self, all_motifs: Dict[int, List[Dict]]):
        """Plot comprehensive motif analysis summary."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Number of significant motifs by length
        ax = axes[0, 0]
        lengths = list(all_motifs.keys())
        counts = [len(motifs) for motifs in all_motifs.values()]
        
        bars = ax.bar(lengths, counts, color='lightblue', edgecolor='black')
        ax.set_title('Number of Significant Motifs by Length')
        ax.set_xlabel('Motif Length')
        ax.set_ylabel('Number of Motifs')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
        
        # Plot 2: Fold change distribution
        ax = axes[0, 1]
        all_fold_changes = []
        for motifs in all_motifs.values():
            all_fold_changes.extend([m['log2_fold_change'] for m in motifs])
        
        ax.hist(all_fold_changes, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Distribution of Log2 Fold Changes')
        ax.set_xlabel('Log2 Fold Change')
        ax.set_ylabel('Frequency')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: P-value distribution
        ax = axes[0, 2]
        all_p_values = []
        for motifs in all_motifs.values():
            all_p_values.extend([m['p_value'] for m in motifs])
        
        ax.hist(np.log10(np.array(all_p_values) + 1e-10), bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Distribution of P-values (log10)')
        ax.set_xlabel('Log10 P-value')
        ax.set_ylabel('Frequency')
        
        # Plot 4: Top motifs by fold change
        ax = axes[1, 0]
        top_motifs = []
        for motifs in all_motifs.values():
            top_motifs.extend(motifs[:3])  # Top 3 from each length
        
        top_motifs.sort(key=lambda x: abs(x['log2_fold_change']), reverse=True)
        top_motifs = top_motifs[:10]  # Top 10 overall
        
        motif_names = [m['motif'] for m in top_motifs]
        fold_changes = [m['log2_fold_change'] for m in top_motifs]
        
        colors = ['red' if fc > 0 else 'blue' for fc in fold_changes]
        bars = ax.barh(range(len(motif_names)), fold_changes, color=colors, alpha=0.7)
        ax.set_yticks(range(len(motif_names)))
        ax.set_yticklabels(motif_names)
        ax.set_xlabel('Log2 Fold Change')
        ax.set_title('Top Discriminative Motifs')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 5: Motif frequency comparison
        ax = axes[1, 1]
        if len(all_motifs) > 0:
            # Get top motifs from first length
            first_length = min(all_motifs.keys())
            top_motifs_freq = all_motifs[first_length][:8]
            
            motifs = [m['motif'] for m in top_motifs_freq]
            freq_0 = [m['freq_class_0'] for m in top_motifs_freq]
            freq_1 = [m['freq_class_1'] for m in top_motifs_freq]
            
            x = np.arange(len(motifs))
            width = 0.35
            
            ax.bar(x - width/2, freq_0, width, label='Class 0', alpha=0.7)
            ax.bar(x + width/2, freq_1, width, label='Class 1', alpha=0.7)
            
            ax.set_xlabel('Motifs')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Motif Frequencies ({first_length}-mers)')
            ax.set_xticks(x)
            ax.set_xticklabels(motifs, rotation=45)
            ax.legend()
        
        # Plot 6: Nucleotide composition
        ax = axes[1, 2]
        nucleotide_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        
        for seq in self.sequences:
            for nuc in seq:
                if nuc in nucleotide_counts:
                    nucleotide_counts[nuc] += 1
        
        total = sum(nucleotide_counts.values())
        nucleotide_freqs = {nuc: count/total for nuc, count in nucleotide_counts.items()}
        
        ax.bar(nucleotide_freqs.keys(), nucleotide_freqs.values(), 
               color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        ax.set_title('Overall Nucleotide Composition')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Nucleotide')
        
        plt.tight_layout()
        plt.show()
    
    def get_motif_summary_table(self, all_motifs: Dict[int, List[Dict]], 
                              top_n_per_length: int = 5) -> pd.DataFrame:
        """Create summary table of top motifs."""
        
        summary_data = []
        
        for length, motifs in all_motifs.items():
            for i, motif in enumerate(motifs[:top_n_per_length]):
                summary_data.append({
                    'Length': length,
                    'Rank': i + 1,
                    'Motif': motif['motif'],
                    'Class_0_Freq': f"{motif['freq_class_0']:.4f}",
                    'Class_1_Freq': f"{motif['freq_class_1']:.4f}",
                    'Log2_FC': f"{motif['log2_fold_change']:.3f}",
                    'P_value': f"{motif['p_value']:.2e}",
                    'Total_Count': motif['total_count'],
                    'Enriched_in': 'Class_1' if motif['log2_fold_change'] > 0 else 'Class_0'
                })
        
        return pd.DataFrame(summary_data)
    
    def analyze_motif_positions(self, motifs: List[str], sequences: List[str]) -> Dict:
        """Analyze positional preferences of motifs."""
        
        position_data = defaultdict(list)
        
        for motif in motifs:
            positions = []
            for seq in sequences:
                for match in re.finditer(motif, seq):
                    # Normalize position by sequence length
                    rel_pos = match.start() / len(seq)
                    positions.append(rel_pos)
            
            if positions:
                position_data[motif] = {
                    'positions': positions,
                    'mean_position': np.mean(positions),
                    'std_position': np.std(positions),
                    'count': len(positions)
                }
        
        return dict(position_data)
    
    def plot_motif_positions(self, motifs: List[str], max_motifs: int = 6):
        """Plot positional distribution of motifs."""
        
        # Analyze positions for both classes
        class_0_positions = self.analyze_motif_positions(motifs, self.class_0_sequences)
        class_1_positions = self.analyze_motif_positions(motifs, self.class_1_sequences)
        
        # Select top motifs by total count
        motif_counts = {}
        for motif in motifs:
            count_0 = class_0_positions.get(motif, {}).get('count', 0)
            count_1 = class_1_positions.get(motif, {}).get('count', 0)
            motif_counts[motif] = count_0 + count_1
        
        top_motifs = sorted(motif_counts.keys(), key=lambda x: motif_counts[x], reverse=True)[:max_motifs]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, motif in enumerate(top_motifs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot histograms
            if motif in class_0_positions:
                ax.hist(class_0_positions[motif]['positions'], bins=20, alpha=0.6, 
                       label='Class 0', color='blue', density=True)
            
            if motif in class_1_positions:
                ax.hist(class_1_positions[motif]['positions'], bins=20, alpha=0.6, 
                       label='Class 1', color='red', density=True)
            
            ax.set_title(f'Motif: {motif}')
            ax.set_xlabel('Relative Position')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(top_motifs), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()


def run_comprehensive_motif_analysis(sequences: List[str], labels: List[int], 
                                   motif_lengths: List[int] = [6, 7, 8, 9, 10, 11]):
    """Run comprehensive motif analysis."""
    
    print("="*60)
    print("COMPREHENSIVE MOTIF ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MotifAnalyzer(sequences, labels, motif_lengths)
    
    # Analyze all motif lengths
    all_motifs = analyzer.analyze_all_motif_lengths(min_frequency=10, max_motifs_per_length=20)
    
    # Create visualizations
    analyzer.plot_motif_analysis_summary(all_motifs)
    
    # Create summary table
    summary_table = analyzer.get_motif_summary_table(all_motifs, top_n_per_length=3)
    print("\nTop Discriminative Motifs:")
    print(summary_table.to_string(index=False))
    
    # Analyze motif positions for most significant motifs
    if all_motifs:
        first_length = min(all_motifs.keys())
        top_motifs = [m['motif'] for m in all_motifs[first_length][:6]]
        print(f"\nAnalyzing positional preferences for {first_length}-mers...")
        analyzer.plot_motif_positions(top_motifs)
    
    # Create sequence logos for top motifs
    print("\nCreating sequence logos...")
    for length, motifs in all_motifs.items():
        if motifs:
            # Class 1 enriched motifs
            class_1_motifs = [m['motif'] for m in motifs if m['log2_fold_change'] > 0][:5]
            if class_1_motifs:
                analyzer.plot_motif_logo(class_1_motifs, f"Class 1 Enriched {length}-mers")
            
            # Class 0 enriched motifs  
            class_0_motifs = [m['motif'] for m in motifs if m['log2_fold_change'] < 0][:5]
            if class_0_motifs:
                analyzer.plot_motif_logo(class_0_motifs, f"Class 0 Enriched {length}-mers")
    
    return analyzer, all_motifs, summary_table


if __name__ == "__main__":
    # Example usage with sample data
    print("Motif analysis module loaded successfully!")
    print("Use run_comprehensive_motif_analysis(sequences, labels) to analyze your data.")
