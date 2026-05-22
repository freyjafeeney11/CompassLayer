import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_performance_graphs(csv_file="performance_log.csv"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Could not find {csv_file}. Run your main app first!")
        return

    # Use seaborn for a beautiful modern theme
    sns.set_theme(style="whitegrid", palette="pastel")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle('CompassLayer Performance Analysis', fontsize=18, fontweight='bold', y=0.98)

    cpu_count = os.cpu_count() or 1
    app_cpu_norm = df['App_CPU_Percent'] / cpu_count
    base_cpu = (df['Sys_CPU_Percent'] - app_cpu_norm).clip(lower=0)
    
    # Beautiful colors for the stackplot
    colors = sns.color_palette("muted")
    ax1.stackplot(df['Time_s'], base_cpu, app_cpu_norm, 
                  labels=['Base System (Game + OS)', 'CompassLayer Overhead'],
                  colors=[colors[0], colors[3]], alpha=0.85)
    
    ax1.set_xlabel('Time Elapsed (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System CPU Usage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('CPU Footprint: Game vs. CompassLayer (%)', fontsize=15, pad=15, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.set_ylim(0, max(100, df['Sys_CPU_Percent'].max() + 5))
    ax1.margins(x=0)
    sns.lineplot(data=df, x='Time_s', y='Latency_ms', ax=ax2, color=sns.color_palette("flare")[2], linewidth=2.5)
    ax2.set_xlabel('Time Elapsed (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frame Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('CompassLayer Internal Processing Latency (ms)', fontsize=15, pad=15, fontweight='bold')
    avg_lat = df['Latency_ms'].mean()
    ax2.axhline(avg_lat, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_lat:.1f} ms')
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_ylim(0, df['Latency_ms'].max() * 1.2)
    ax2.margins(x=0)

    sns.lineplot(data=df, x='Time_s', y='Memory_MB', ax=ax3, color=sns.color_palette("crest")[2], linewidth=2.5)
    
    ax3.fill_between(df['Time_s'], df['Memory_MB'], color=sns.color_palette("crest")[2], alpha=0.3)
    
    ax3.set_xlabel('Time Elapsed (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Allocated RAM (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('CompassLayer Memory Consumption (MB)', fontsize=15, pad=15, fontweight='bold')
    ax3.set_ylim(0, df['Memory_MB'].max() * 1.5)
    ax3.margins(x=0)

    sns.despine(left=True, bottom=True)
    fig.tight_layout(pad=4.0, h_pad=5.0, w_pad=4.0, rect=[0, 0, 1, 0.93])
   
    output_filename = 'performance_graphs.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graphs successfully generated and saved as '{output_filename}'")
   
    plt.show()

if __name__ == '__main__':
    generate_performance_graphs()