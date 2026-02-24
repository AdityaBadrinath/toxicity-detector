import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix_placeholder():
    """Visual placeholder for dashboard metrics."""
    fig, ax = plt.subplots(figsize=(6, 3))
    # Simple bar chart for demo visualization
    categories = ['Toxic', 'Non-Toxic']
    scores = [0, 100] # Placeholder
    
    colors = ['#ff4b4b', '#21c354']
    ax.barh(categories, scores, color=colors)
    ax.set_xlim(0, 100)
    ax.set_title("Confidence Distribution")
    return fig

def create_summary_stats(df):
    """Generate summary stats from batch predictions."""
    total = len(df)
    toxic_count = df[df['Prediction'] == 'Toxic'].shape[0]
    non_toxic_count = total - toxic_count
    
    stats = {
        "Total Comments": total,
        "Toxic %": round((toxic_count / total) * 100, 2),
        "Non-Toxic %": round((non_toxic_count / total) * 100, 2),
        "Toxic Count": toxic_count
    }
    return stats